from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import litellm
import networkx as nx
from rank_bm25 import BM25Okapi

from src.config.cfg import Config
from .base import BaseMemory

logger = logging.getLogger(__name__)


def _normalize_predicate(raw: str, allowed: frozenset[str]) -> str:
    """Map a raw LLM-generated predicate to the nearest allowed predicate.

    Strategy (in order of preference):
    1. Exact match — return as-is.
    2. Canonical aliases — common LLM paraphrases mapped to the intended predicate.
    3. Token-overlap — split both strings on '_' and count shared tokens; take the
       allowed predicate with the highest overlap (ties broken by string length).
    4. Fallback — return 'related_to' if no overlap is found.

    This runs post-LLM so invented free-text predicates ("mastered_recordings_for",
    "was a case of", "executive produced") are silently normalised rather than
    polluting the graph with a long tail of hapax predicates.
    """
    p = raw.strip().lower().replace(" ", "_")

    # 1. Exact match
    if p in allowed:
        return p

    # 2. Hard-coded aliases for common LLM paraphrases
    _ALIASES: dict[str, str] = {
        "works_at":          "educated_at",
        "worked_at":         "educated_at",
        "studied_at":        "educated_at",
        "studies_at":        "educated_at",
        "works_for":         "works_as",
        "worked_as":         "works_as",
        "occupation":        "works_as",
        "profession":        "works_as",
        "is_an":             "is_a",
        "was_a":             "is_a",
        "is_also_known_as":  "known_as",
        "also_known_as":     "known_as",
        "nickname":          "known_as",
        "alias":             "known_as",
        "published_by":      "produced_by",
        "produced":          "produced_by",
        "executive_produced": "produced_by",
        "co-produced":       "produced_by",
        "wrote":             "written_by",
        "authored":          "authored_by",
        "directed":          "directed_by",
        "co-directed":       "directed_by",
        "starring":          "starred_in",
        "stars":             "starred_in",
        "plays_role_of":     "starred_in",
        "played_role_of":    "starred_in",
        "guest_starred_in":  "appeared_in",
        "appeared_as":       "appeared_in",
        "plays_in":          "appeared_in",
        "played_in":         "appeared_in",
        "released":          "released_in",
        "released_on":       "released_in",
        "published":         "published_in",
        "published_on":      "published_in",
        "founded":           "founded_by",
        "co-founded":        "founded_by",
        "established":       "founded_by",
        "serves":            "serves_as",
        "served_as":         "serves_as",
        "led":               "led_by",
        "leads":             "led_by",
        "manages":           "led_by",
        "coached":           "coached_by",
        "trained_by":        "coached_by",
        "competed_at":       "competed_in",
        "participated_in":   "competed_in",
        "plays_for":         "played_for",
        "represented":       "played_for",
        "is_part_of":        "part_of",
        "is_member_of":      "member_of",
        "is_related_to":     "related_to",
        "has_relation_to":   "related_to",
        "associated_with":   "affiliated_with",
        "collaborated_with": "affiliated_with",
        "worked_with":       "affiliated_with",
        "worked_on":         "affiliated_with",
        "owns":              "affiliated_with",
        "owned_by":          "affiliated_with",
        "operates":          "affiliated_with",
        "located_near":      "located_in",
        "based_in":          "located_in",
        "borders":           "located_in",
        "relocated_to":      "located_in",
        "moved_to":          "located_in",
        "capital":           "capital_of",
        "married":           "married_to",
        "spouse_of":         "married_to",
        "born":              "born_in",
        "died":              "died_in",
        "lost_to":           "involved_in",
        "defeated":          "involved_in",
        "contains":          "has_property",
        "includes":          "has_property",
        "features":          "has_property",
        "consists_of":       "has_property",
        "composed_of":       "has_property",
        "noted_for":         "known_as",
        "is_also_called":    "known_as",
        "also_called":       "known_as",
        "named_after":       "known_as",
        "specializes_in":    "works_as",
        "graduate_of":       "educated_at",
        "attended":          "educated_at",
        "studied_under":     "taught_by",
        "mentored_by":       "taught_by",
        "co-starred":        "appeared_in",
        "co_starred":        "appeared_in",
        "created":           "founded_by",
        "created_by":        "founded_by",
        "replaced_by":       "followed_by",
        "succeeded_by":      "followed_by",
        "succeeded":         "preceded_by",
        "recorded":          "released_in",
        "aired_on":          "released_in",
        "broadcast_on":      "released_in",
        "distributed_by":    "produced_by",
        "distributed":       "produced_by",
    }
    if p in _ALIASES:
        return _ALIASES[p]

    # 3. Token-overlap scoring
    # Stop-tokens are excluded from overlap to avoid spurious matches driven by
    # common words ("for", "a", "of", "in", "by") that carry no semantic content.
    _STOP_TOKENS = frozenset({"a", "an", "the", "of", "for", "in", "by", "at",
                               "to", "is", "was", "be", "been", "as", "on"})
    p_tokens = set(p.split("_")) - _STOP_TOKENS
    best_pred, best_score = "related_to", 0
    for a in allowed:
        a_tokens = set(a.split("_")) - _STOP_TOKENS
        score = len(p_tokens & a_tokens)
        if score > best_score or (score == best_score and len(a) < len(best_pred)):
            best_pred, best_score = a, score

    # Only accept overlap ≥ 1 to avoid nonsense matches
    return best_pred if best_score >= 1 else "related_to"

# --- Predicate whitelist ---
# Why: unconstrained LLMs invent free-text predicates ("is located on the large
# natural bay of", "is also known by her nickname") that fragment the graph into
# disconnected clusters. A shared vocabulary ensures that the same relationship
# extracted from two different documents uses the same predicate string, making
# cross-document entity linking possible.
#
# Design principle: predicates are derived from semantic categories, not from
# observed data. Each category covers a distinct relation class; a predicate is
# added when it is conceptually irreducible to existing ones.
# Source taxonomy: ConceptNet relation types + Wikidata property categories,
# restricted to the domains present in HotpotQA (biography, film/media,
# sport, geography, science, history, politics).
#
# Categories and their predicates:
#
#   [BIOGRAPHICAL — identity & life events]
#     born_in        where a person was born (city/country)
#     born_on        when a person was born (date/year)
#     died_in        where a person died
#     died_on        when a person died
#     nationality    citizenship or ethnic identity of a person
#     known_as       alias, nickname, pen name, stage name
#     works_as       occupation or profession (NEW: distinct from is_a — role, not type)
#
#   [SOCIAL — interpersonal relations]
#     child_of       parent-child (NEW: common bridge in biography questions)
#     married_to     spousal relationship
#     educated_at    institution where a person studied (NEW: academic bridge)
#     taught_by      academic mentor/supervisor (NEW: named-person bridge)
#
#   [CLASSIFICATION — type & membership]
#     is_a           entity type or category
#     part_of        structural component of a larger whole
#     member_of      membership in a group, team, or organisation
#     affiliated_with loose association — when member_of is too strong (NEW)
#
#   [GEOGRAPHY — spatial relations]
#     located_in     physical location of an entity
#     capital_of     a city that is the capital of a country/region (NEW)
#
#   [ORGANISATION — founding & leadership]
#     founded_by     person or group who established an entity
#     founded_in     year or place of founding
#     led_by         current or historical head/leader (NEW: broader than coached_by)
#
#   [CREATIVE WORKS — authorship & production]
#     directed_by    film/theatre director
#     produced_by    film/record producer
#     written_by     author of script, book, article
#     authored_by    book author (distinct from written_by: longer-form works)
#     composed_by    musical composer (NEW: distinct from written_by)
#     based_on       adaptation source — novel, true story, earlier work (NEW)
#     appeared_in    participation broader than a lead role (NEW: vs. starred_in)
#     starred_in     lead acting role
#     released_in    publication or release year/location
#     published_in   for books, journals, periodicals (NEW: distinct from released_in)
#
#   [AWARDS & RECOGNITION]
#     won            award or title won
#     nominated_for  award nomination without win
#
#   [SPORT]
#     played_for     team or club a player represented
#     coached_by     coach or manager of a team/athlete
#     competed_in    event or competition participated in (NEW)
#
#   [SEQUENTIAL — ordering & succession]
#     followed_by    successor (person, work, event)
#     preceded_by    predecessor
#
#   [ROLES — contextual assignments]
#     serves_as      role held by a person in an organisation or context (NEW)
#
#   [FALLBACK — when no specific predicate fits]
#     Use these only if no specific predicate above applies.
#     Ensures that entities are still added to the graph (and reachable via BFS)
#     even when the exact relationship type cannot be expressed precisely.
#     Ordered from most to least specific:
#     has_property   entity has an attribute or characteristic
#                    (e.g. "Novel X has_property Genre Mystery")
#     involved_in    entity participates in an event, process or situation
#                    (e.g. "Person X involved_in Conflict Y")
#     related_to     last-resort generic link — use only when nothing else fits
#
_ALLOWED_PREDICATES = ", ".join([
    # Biographical
    "born_in", "born_on", "died_in", "died_on", "nationality", "known_as", "works_as",
    # Social
    "child_of", "married_to", "educated_at", "taught_by",
    # Classification
    "is_a", "part_of", "member_of", "affiliated_with",
    # Geography
    "located_in", "capital_of",
    # Organisation
    "founded_by", "founded_in", "led_by",
    # Creative works
    "directed_by", "produced_by", "written_by", "authored_by",
    "composed_by", "based_on", "appeared_in", "starred_in", "released_in", "published_in",
    # Awards
    "won", "nominated_for",
    # Sport
    "played_for", "coached_by", "competed_in",
    # Sequential
    "followed_by", "preceded_by",
    # Roles
    "serves_as",
    # Fallback (use only when no specific predicate fits)
    "has_property", "involved_in", "related_to",
])

# Frozenset for O(1) membership tests in _normalize_predicate
_ALLOWED_PREDICATES_SET: frozenset[str] = frozenset(
    p.strip() for p in _ALLOWED_PREDICATES.split(",")
)

_SYSTEM_PROMPT = (
    "Extract the most important entities and relationships from the text as triples. "
    "Return a JSON object with a 'triples' key containing an array of at most 20 objects, "
    "each with 'subject', 'predicate', and 'object' string fields.\n"
    "Rules:\n"
    "  - subject and object: 1-5 words, Title Case (e.g. 'Clara Voss', 'Lake Arven').\n"
    f"  - predicate: snake_case, choose the closest specific match from: {_ALLOWED_PREDICATES}.\n"
    "    Use has_property / involved_in / related_to only as last resort — prefer specific predicates.\n"
    "  - NEVER put a list in object. One entity per triple.\n"
    "    Bad:  {\"subject\": \"Film X\", \"predicate\": \"starred_in\", "
    "\"object\": \"Actor A, Actor B, Actor C\"}\n"
    "    Good: three separate triples, one per actor.\n"
    'Return {"triples": []} if no clear relationships exist. '
    "Output only the JSON object, no other text."
)

TRIPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject":   {"type": "string"},
                    "predicate": {"type": "string"},
                    "object":    {"type": "string"},
                },
                "required": ["subject", "predicate", "object"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["triples"],
    "additionalProperties": False,
}

# Fallback: extract triples from truncated JSON via regex
_TRIPLE_RE = re.compile(
    r'\{\s*"subject"\s*:\s*"(?P<s>[^"]+)"\s*,\s*"predicate"\s*:\s*"(?P<p>[^"]+)"\s*,'
    r'\s*"object"\s*:\s*"(?P<o>[^"]+)"\s*\}'
)


def _split_list_objects(triples: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Post-processing fallback: if the LLM returns a comma-separated list as
    the object despite the prompt instruction, split it into individual triples.
    Heuristic: split only when ≥3 comma-separated parts each ≤40 chars.
    Short pairs like "New York, USA" (2 parts) are kept intact.
    """
    result: list[dict[str, str]] = []
    for t in triples:
        parts = [p.strip() for p in t["object"].split(",") if p.strip()]
        if len(parts) >= 3 and all(len(p) <= 40 for p in parts):
            for part in parts:
                result.append({"subject": t["subject"], "predicate": t["predicate"], "object": part})
        else:
            result.append(t)
    return result


def parse_triples(raw: str) -> list[dict[str, str]]:
    """Parse an LLM response into normalised (subject, predicate, object) triples.

    Shared by both graph variants (GraphMemory and LlamaIndexGraphMemory) so they
    apply identical parsing: JSON-first with a regex fallback for truncated output,
    list-object splitting, and predicate normalisation against the shared whitelist.
    Keeps entity casing verbatim (unlike LlamaIndex's default parser, which
    lowercases via .capitalize() and drops comma-valued objects entirely).
    """
    raw = raw or "{}"
    try:
        parsed = json.loads(raw)
        triples = parsed.get("triples", []) if isinstance(parsed, dict) else []
    except json.JSONDecodeError:
        matches = list(_TRIPLE_RE.finditer(raw))
        if matches:
            logger.debug("parse_triples: JSON truncated, recovered %d triples via regex", len(matches))
            triples = [{"subject": m.group("s"), "predicate": m.group("p"), "object": m.group("o")} for m in matches]
        else:
            logger.warning("parse_triples: JSON parse failed: %r", raw[:200])
            return []

    valid = [
        t for t in triples
        if isinstance(t, dict)
        and all(isinstance(t.get(k), str) and t.get(k, "").strip() for k in ("subject", "predicate", "object"))
    ]
    split = _split_list_objects(valid)

    # Normalise predicates: map LLM-invented free-text predicates to the
    # nearest entry in _ALLOWED_PREDICATES_SET so the graph stays clean.
    normalised = []
    for t in split:
        raw_p = t["predicate"]
        norm_p = _normalize_predicate(raw_p, _ALLOWED_PREDICATES_SET)
        if norm_p != raw_p:
            logger.debug("parse_triples: predicate %r → %r", raw_p, norm_p)
        normalised.append({**t, "predicate": norm_p})

    logger.debug(
        "parse_triples: %d valid → %d after list-split → %d after normalisation",
        len(valid), len(split), len(normalised),
    )
    return normalised


class GraphMemory(BaseMemory):
    """
    Knowledge-graph memory backend backed by a NetworkX MultiDiGraph.

    Design decisions (see EXPERIMENT_PLAN.md § Graph-Variante):
      1. MultiDiGraph — allows multiple predicates between the same node pair
         without silent data loss.
      2. Predicate whitelist — shared vocabulary across documents enables
         cross-document entity linking.
      3. BM25 entity linking — maps natural-language query tokens to graph
         nodes without an LLM call, preserving the zero-cost retrieval
         property that distinguishes GraphRAG from Vector RAG.

    Ingest (Phase A):
        Each document is passed to the LLM (Ollama JSON Schema) which extracts
        (subject, predicate, object) triples. Post-processing splits any
        list-valued objects into individual triples. Results are added as
        directed edges. Graph is persisted via nx.node_link_data().

    Retrieval (Phase B):
        search() uses BM25 over node names to find the best-matching entry
        nodes for the query (entity linking), then runs a manual per-seed
        hop-by-hop BFS (undirected, radius=max_hops) from those nodes to collect
        context triples. The BFS is level-by-level (not nx.ego_graph) so that
        1-hop edges are emitted before 2-hop edges and survive the top_k
        truncation. No LLM call at retrieval time.
    """

    def __init__(self, config: Config) -> None:
        self._llm_cfg = config.llm.ingest
        self._max_hops: int = config.graph.max_hops
        # Graph uses its own top_k (graph.top_k) because triples are ~8 words
        # vs. text passages (~50 words). Needs more results to cover 2-hop paths.
        self._top_k: int = config.graph.top_k
        self._rerank_model: str = config.graph.rerank_model
        self._rerank_top_n: int = config.graph.rerank_top_n
        self._storage_path: Path = Path(config.stores.graph)
        self._checkpoint_path: Path = self._storage_path.with_suffix(".checkpoint")
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._bm25: BM25Okapi | None = None
        self._node_list: list[str] = []
        # Cross-encoder loaded lazily on first search — no cost during ingest runs.
        self._reranker = None
        if self._storage_path.exists():
            self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        data = json.loads(self._storage_path.read_text(encoding="utf-8"))
        # Force multigraph so that files saved by the old DiGraph code are
        # transparently upgraded to MultiDiGraph on first load.
        data["multigraph"] = True
        self._graph = nx.node_link_graph(data, directed=True, multigraph=True)
        self._bm25 = None
        logger.debug(
            "GraphMemory: loaded %d nodes, %d edges from %s",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
            self._storage_path,
        )

    def _save(self, checkpoint: int | None = None) -> None:
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._storage_path.write_text(
            json.dumps(nx.node_link_data(self._graph), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if checkpoint is not None:
            self._checkpoint_path.write_text(str(checkpoint), encoding="utf-8")

    def read_checkpoint(self) -> int:
        if self._checkpoint_path.exists():
            try:
                return int(self._checkpoint_path.read_text(encoding="utf-8").strip())
            except ValueError:
                return 0
        return 0

    # ── BM25 entity-linking index ─────────────────────────────────────────────

    # Common English stopwords that produce false-positive BM25 matches.
    # Example without this list: "Who designed the hotel..." → "who" matches
    # band node "The Who" → BFS returns music facts instead of hotel facts.
    _STOPWORDS = frozenset({
        "who", "what", "when", "where", "which", "how", "why",
        "the", "a", "an", "and", "or", "but", "in", "on", "at",
        "to", "of", "for", "by", "with", "from", "is", "was",
        "are", "were", "be", "been", "has", "had", "have", "do",
        "did", "does", "not", "it", "its", "that", "this", "as",
    })

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        tokens = re.sub(r"[^a-z0-9]", " ", text.lower()).split()
        # Keep tokens ≥3 chars that are not stopwords.
        # ≥3 (not >3) preserves short but meaningful tokens like "vh1", "bbc", "nba".
        return [t for t in tokens if len(t) >= 3 and t not in cls._STOPWORDS]

    def _build_node_index(self) -> None:
        self._node_list = list(self._graph.nodes())
        corpus = [self._tokenize(str(n)) for n in self._node_list]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    # ── Cross-encoder reranking ────────────────────────────────────────────────

    def _rerank(self, query: str, candidates: list[str]) -> list[str]:
        """Score candidate triple strings against the query with a cross-encoder
        and return the top `rerank_top_n` by relevance.

        Replaces the previous BFS-order truncation (seed-rank + hop-distance),
        which never scored candidates against the query. Runs on CPU with no LLM
        call, preserving the zero-cost-retrieval property. Loaded lazily so ingest
        runs never pay the model-load cost.
        """
        if not candidates:
            return []
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self._rerank_model, device="cpu")
        scores = self._reranker.predict([(query, c) for c in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[: self._rerank_top_n]]

    # ── LLM triple extraction ─────────────────────────────────────────────────

    def _extract_triples(self, text: str, phase: str, run_id: str) -> list[dict[str, str]]:
        response = litellm.completion(
            model=self._llm_cfg.model,
            api_base=self._llm_cfg.base_url,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=self._llm_cfg.max_tokens,
            response_format={"type": "json_schema", "json_schema": {
                "name": "triples",
                "schema": TRIPLE_SCHEMA,
            }},
            metadata={
                "phase": phase,
                "actor": "graph_extract",
                "variant_name": "graph",
                "run_id": run_id,
            },
        )
        raw = response.choices[0].message.content or "{}"
        triples = parse_triples(raw)
        logger.debug("GraphMemory: extracted %d triples from %d-char text", len(triples), len(text))
        return triples

    # ── BaseMemory interface ──────────────────────────────────────────────────

    def ingest_documents(self, documents: list[str], start_from: int = 0) -> None:
        non_empty = [d for d in documents if d.strip()]
        total = len(non_empty)
        for i, doc in enumerate(non_empty[start_from:], start_from + 1):
            triples = self._extract_triples(doc, phase="ingest", run_id="ingest")
            for t in triples:
                self._graph.add_edge(
                    t["subject"].strip(),
                    t["object"].strip(),
                    predicate=t["predicate"].strip(),
                )
            if i % 10 == 0 or i == total:
                print(f"  [{i:4d}/{total}]  nodes={self._graph.number_of_nodes()}  edges={self._graph.number_of_edges()}", flush=True)
            if i % 50 == 0:
                self._save(checkpoint=i)
                self._bm25 = None
        self._save(checkpoint=total)
        self._bm25 = None

    def search(self, query: str) -> list[str]:
        if not query.strip() or self._graph.number_of_edges() == 0:
            return []

        if self._bm25 is None:
            self._build_node_index()

        if not self._node_list:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        top_indices = scores.argsort()[::-1][:10]
        seed_nodes = [self._node_list[i] for i in top_indices if scores[i] > 0]

        if not seed_nodes:
            return []

        # Hop-by-hop BFS so that 1-hop edges (directly relevant) appear before
        # 2- and 3-hop edges. Seeds are ordered by BM25 score, so the best-
        # matching entity's immediate neighborhood is returned first.
        seen_edges: set[tuple[str, str, str]] = set()
        results: list[str] = []

        for seed in seed_nodes:
            if seed not in self._graph:
                continue
            visited: set[str] = {seed}
            frontier: set[str] = {seed}
            for _ in range(self._max_hops):
                next_frontier: set[str] = set()
                for node in frontier:
                    for s, o, data in self._graph.out_edges(node, data=True):
                        p = data.get("predicate", "related_to")
                        key = (s, p, o)
                        if key not in seen_edges:
                            seen_edges.add(key)
                            results.append(f"{s} {p.replace('_', ' ')} {o}")
                        if o not in visited:
                            next_frontier.add(o)
                    for s, o, data in self._graph.in_edges(node, data=True):
                        p = data.get("predicate", "related_to")
                        key = (s, p, o)
                        if key not in seen_edges:
                            seen_edges.add(key)
                            results.append(f"{s} {p.replace('_', ' ')} {o}")
                        if s not in visited:
                            next_frontier.add(s)
                visited |= next_frontier
                frontier = next_frontier
                if not frontier:
                    break

        reranked = self._rerank(query, results)
        logger.debug(
            "GraphMemory: query=%r → %d seeds → %d candidates → rerank top %d",
            query, len(seed_nodes), len(results), len(reranked),
        )
        return reranked

    def update_fact(self, fact: str) -> None:
        if not fact.strip():
            return
        triples = self._extract_triples(fact, phase="agent_reasoning", run_id="update_fact")
        for t in triples:
            self._graph.add_edge(
                t["subject"].strip(), t["object"].strip(), predicate=t["predicate"].strip()
            )
        self._bm25 = None
        self._save()

    def reset(self) -> None:
        self._graph = nx.MultiDiGraph()
        self._bm25 = None
        self._node_list = []
        if self._storage_path.exists():
            self._storage_path.unlink()
        if self._checkpoint_path.exists():
            self._checkpoint_path.unlink()
        logger.debug("GraphMemory: reset")

    def get_backend_name(self) -> str:
        return "graph"

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()
