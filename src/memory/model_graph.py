from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import litellm
import networkx as nx

from src.config.cfg import Config
from .base import BaseMemory

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Extract the most important entities and relationships from the text as triples. "
    "Return a JSON object with a 'triples' key containing an array of at most 8 objects, "
    "each with 'subject', 'predicate', and 'object' string fields. "
    "Use title case for entity names (1-4 words) and snake_case for predicates "
    "(e.g. born_in, located_in, discovered_by). "
    'Return {"triples": []} if no clear relationships exist. '
    "Output only the JSON object, no other text."
)

# Fallback: extract complete triple objects from truncated JSON via regex
_TRIPLE_RE = re.compile(
    r'\{\s*"subject"\s*:\s*"(?P<s>[^"]+)"\s*,\s*"predicate"\s*:\s*"(?P<p>[^"]+)"\s*,\s*"object"\s*:\s*"(?P<o>[^"]+)"\s*\}'
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


class GraphMemory(BaseMemory):
    """
    Knowledge-graph memory backend backed by a NetworkX DiGraph.

    Ingestion (Phase A):
        Each document is passed to the LLM (Ollama JSON Schema mode) which
        extracts (subject, predicate, object) triples. Results are added as
        directed edges and the graph is persisted via nx.node_link_data().
        Every extraction call is tagged: phase="ingest", actor="graph_extract".

    Retrieval (Phase B):
        search() uses nx.ego_graph (BFS, radius=max_hops, undirected=True) to
        find all edges reachable from seed nodes that match the query tokens.
        No LLM call at retrieval time.

    update_fact() (Phase B):
        Tagged: phase="agent_reasoning", actor="graph_extract".
    """

    def __init__(self, config: Config) -> None:
        self._llm_cfg = config.llm.ingest
        self._max_hops: int = config.graph.max_hops
        self._top_k: int = config.retrieval.top_k
        self._storage_path: Path = Path(config.stores.graph)
        self._graph: nx.DiGraph = nx.DiGraph()
        if self._storage_path.exists():
            self._load()

    # ---- Persistence ----

    def _load(self) -> None:
        data = json.loads(self._storage_path.read_text(encoding="utf-8"))
        self._graph = nx.node_link_graph(data, directed=True, multigraph=False)
        logger.debug(
            "GraphMemory: loaded %d nodes, %d edges from %s",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
            self._storage_path,
        )

    def _save(self) -> None:
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._storage_path.write_text(
            json.dumps(nx.node_link_data(self._graph), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ---- LLM triple extraction ----

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
        try:
            parsed = json.loads(raw)
            triples = parsed.get("triples", []) if isinstance(parsed, dict) else []
        except json.JSONDecodeError:
            # Truncated JSON: extract complete triple objects via regex
            matches = list(_TRIPLE_RE.finditer(raw))
            if matches:
                logger.debug("GraphMemory: JSON truncated, recovered %d triples via regex", len(matches))
                return [{"subject": m.group("s"), "predicate": m.group("p"), "object": m.group("o")} for m in matches]
            logger.warning("GraphMemory: JSON parse failed and no triples recoverable: %r", raw[:200])
            return []

        valid = [
            t for t in triples
            if isinstance(t, dict)
            and all(isinstance(t.get(k), str) and t.get(k, "").strip() for k in ("subject", "predicate", "object"))
        ]
        logger.debug("GraphMemory: %d valid triples from %d-char text", len(valid), len(text))
        return valid

    # ---- BaseMemory interface ----

    def ingest_documents(self, documents: list[str]) -> None:
        for doc in documents:
            if not doc.strip():
                continue
            triples = self._extract_triples(doc, phase="ingest", run_id="ingest")
            for t in triples:
                self._graph.add_edge(
                    t["subject"].strip(),
                    t["object"].strip(),
                    predicate=t["predicate"].strip(),
                )
        self._save()
        logger.debug(
            "GraphMemory: ingested %d docs → %d nodes, %d edges",
            len(documents),
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
        )

    def search(self, query: str) -> list[str]:
        if not query.strip() or self._graph.number_of_edges() == 0:
            return []

        tokens = [t.lower() for t in query.split() if len(t) > 2]
        if not tokens:
            return []

        seed_nodes = [
            n for n in self._graph.nodes()
            if any(tok in str(n).lower() for tok in tokens)
        ]
        if not seed_nodes:
            return []

        seen: set[tuple[str, str, str]] = set()
        results: list[str] = []
        for seed in seed_nodes:
            try:
                sub = nx.ego_graph(self._graph, seed, radius=self._max_hops, undirected=True)
            except nx.NetworkXError:
                continue
            for s, o, data in sub.edges(data=True):
                p = data.get("predicate", "related_to")
                key = (s, p, o)
                if key not in seen:
                    seen.add(key)
                    results.append(f"{s} {p.replace('_', ' ')} {o}")

        logger.debug(
            "GraphMemory: query=%r → %d seeds → %d edges",
            query, len(seed_nodes), len(results),
        )
        return results[:self._top_k]

    def update_fact(self, fact: str) -> None:
        if not fact.strip():
            return
        triples = self._extract_triples(fact, phase="agent_reasoning", run_id="update_fact")
        for t in triples:
            self._graph.add_edge(
                t["subject"].strip(), t["object"].strip(), predicate=t["predicate"].strip()
            )
        self._save()

    def reset(self) -> None:
        self._graph = nx.DiGraph()
        if self._storage_path.exists():
            self._storage_path.unlink()
        logger.debug("GraphMemory: reset")

    def get_backend_name(self) -> str:
        return "graph"

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()
