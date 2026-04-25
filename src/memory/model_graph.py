from __future__ import annotations

import json
import logging
from pathlib import Path

import litellm

from src.config.settings import ExperimentConfig

from .base import BaseMemory

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """\
Extract all factual relationships from the text below as (subject, predicate, object) triples.

Rules:
- Keep entity names short and consistent (1-4 words, title case).
- Use snake_case for predicates (e.g. born_in, located_in, discovered_by).
- Return ONLY a JSON array — no explanation, no markdown fences.
- Return [] if no clear relationships exist.

Text: {text}

Example output:
[{{"subject": "Marie Curie", "predicate": "born_in", "object": "Warsaw"}},
 {{"subject": "Warsaw", "predicate": "capital_of", "object": "Poland"}}]"""


class GraphMemory(BaseMemory):
    """
    Knowledge-graph memory backend.

    Ingestion (Phase A):
        Each document is passed to the configured LLM which extracts
        (subject, predicate, object) triples. The resulting graph is persisted
        as a JSON file so the QA phase can read it without re-ingesting.

        Every LLM extraction call is tagged:
            phase="ingest", actor="graph_extract"

    Retrieval (Phase B):
        search() performs keyword-based seed-node matching followed by a
        breadth-first traversal up to ``_MAX_HOPS`` hops. No LLM call is made
        at retrieval time — this is the core advantage being measured: once the
        graph is built, retrieval is free.

    update_fact() (Phase B):
        A new fact string is sent through the same extraction pipeline and the
        resulting triples are added to the live graph. The LLM call is tagged:
            phase="agent_reasoning", actor="graph_extract"
        to distinguish test-time extraction overhead from offline ingestion.

    Graph storage format (JSON):
        {
            "nodes": ["Entity A", "Entity B", ...],
            "edges": [
                {"subject": "Entity A", "predicate": "rel", "object": "Entity B"},
                ...
            ]
        }
    """

    _MAX_HOPS: int = 2

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self._storage_path: Path | None = (
            Path(config.memory_path) if config.memory_path else None
        )
        self._nodes: set[str] = set()
        self._edges: list[dict[str, str]] = []
        if self._storage_path and self._storage_path.exists():
            self._load()

    # ---- Persistence helpers ----

    def _load(self) -> None:
        assert self._storage_path is not None
        data = json.loads(self._storage_path.read_text(encoding="utf-8"))
        self._nodes = set(data.get("nodes", []))
        self._edges = data.get("edges", [])
        logger.debug(
            "GraphMemory: loaded %d nodes, %d edges from %s",
            len(self._nodes),
            len(self._edges),
            self._storage_path,
        )

    def _save(self) -> None:
        if self._storage_path is None:
            return
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._storage_path.write_text(
            json.dumps(
                {"nodes": sorted(self._nodes), "edges": self._edges},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.debug(
            "GraphMemory: saved %d nodes, %d edges to %s",
            len(self._nodes),
            len(self._edges),
            self._storage_path,
        )

    # ---- LLM extraction (shared by ingest and update_fact) ----

    def _extract_triples(
        self, text: str, phase: str, run_id: str
    ) -> list[dict[str, str]]:
        """
        Call the configured LLM to extract (subject, predicate, object) triples.

        Args:
            text:    Raw text to extract from.
            phase:   Telemetry phase tag ("ingest" or "agent_reasoning").
            run_id:  Telemetry run identifier for joining with QA results.

        Returns:
            List of triple dicts. Empty list on parse failure or empty document.
        """
        prompt = _EXTRACTION_PROMPT.format(text=text)
        response = litellm.completion(
            model=self._config.llm.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
            metadata={
                "phase": phase,
                "actor": "graph_extract",
                "variant_name": self._config.variant_name,
                "run_id": run_id,
            },
        )
        raw = response.choices[0].message.content or "[]" # type: ignore
        try:
            triples = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(
                "GraphMemory: could not parse LLM response as JSON: %r", raw[:200]
            )
            return []

        if not isinstance(triples, list):
            logger.warning("GraphMemory: expected JSON array, got %s", type(triples))
            return []

        valid = [
            t
            for t in triples
            if isinstance(t, dict)
            and all(k in t for k in ("subject", "predicate", "object"))
            and all(isinstance(t[k], str) and t[k].strip() for k in ("subject", "predicate", "object"))
        ]
        logger.debug(
            "GraphMemory: extracted %d valid triples from %d-char document",
            len(valid),
            len(text),
        )
        return valid

    def _add_triples(self, triples: list[dict[str, str]]) -> None:
        seen: set[tuple[str, str, str]] = {
            (e["subject"], e["predicate"], e["object"]) for e in self._edges
        }
        for t in triples:
            key = (t["subject"].strip(), t["predicate"].strip(), t["object"].strip())
            if key not in seen:
                seen.add(key)
                self._nodes.add(key[0])
                self._nodes.add(key[2])
                self._edges.append(
                    {"subject": key[0], "predicate": key[1], "object": key[2]}
                )

    # ---- BaseMemory interface ----

    def ingest_documents(self, documents: list[str]) -> None:
        """
        Extract triples from each document via LLM and persist the graph.

        LLM calls tagged: phase="ingest", actor="graph_extract".
        """
        for doc in documents:
            if not doc.strip():
                continue
            triples = self._extract_triples(doc, phase="ingest", run_id="ingest")
            self._add_triples(triples)
        self._save()
        logger.debug(
            "GraphMemory: ingested %d documents → %d nodes, %d edges",
            len(documents),
            len(self._nodes),
            len(self._edges),
        )

    def search(self, query: str) -> list[str]:
        """
        Keyword-seeded BFS over the knowledge graph. No LLM call.

        Finds graph nodes whose names contain any non-trivial query token,
        then traverses outward up to ``_MAX_HOPS`` hops, collecting all edges
        encountered. Returns edges formatted as natural-language strings
        ("Subject relation Object"), capped at config.retrieval.top_k.
        """
        if not query.strip() or not self._edges:
            return []

        query_tokens = [t.lower() for t in query.split() if len(t) > 2]
        if not query_tokens:
            return []

        # Seed: nodes whose name contains any query token
        seed_nodes: set[str] = {
            node
            for node in self._nodes
            if any(token in node.lower() for token in query_tokens)
        }
        if not seed_nodes:
            return []

        visited: set[str] = set(seed_nodes)
        frontier: set[str] = set(seed_nodes)
        seen_edges: set[tuple[str, str, str]] = set()
        collected: list[dict[str, str]] = []

        for _ in range(self._MAX_HOPS):
            next_frontier: set[str] = set()
            for edge in self._edges:
                s, p, o = edge["subject"], edge["predicate"], edge["object"]
                edge_key = (s, p, o)
                if (s in frontier or o in frontier) and edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    collected.append(edge)
                    neighbor = o if s in frontier else s
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                        visited.add(neighbor)
            frontier = next_frontier
            if not frontier:
                break

        results = [
            f"{e['subject']} {e['predicate'].replace('_', ' ')} {e['object']}"
            for e in collected
        ]
        logger.debug(
            "GraphMemory: query=%r → %d seed nodes → %d edges returned",
            query,
            len(seed_nodes),
            len(results),
        )
        return results[: self._config.retrieval.top_k]

    def update_fact(self, fact: str) -> None:
        """
        Parse a new fact into triples and add to the live graph.

        LLM call tagged: phase="agent_reasoning", actor="graph_extract".
        Triples are available to subsequent search() calls immediately.
        """
        if not fact.strip():
            return
        triples = self._extract_triples(
            fact, phase="agent_reasoning", run_id="update_fact"
        )
        self._add_triples(triples)
        self._save()
        logger.debug(
            "GraphMemory: update_fact added %d triples, graph now %d edges",
            len(triples),
            len(self._edges),
        )

    def reset(self) -> None:
        """Clear all nodes and edges and delete the persistence file."""
        self._nodes = set()
        self._edges = []
        if self._storage_path and self._storage_path.exists():
            self._storage_path.unlink()
        logger.debug("GraphMemory: reset — graph cleared")

    def get_backend_name(self) -> str:
        return "graph"

    # ---- Introspection (useful in tests and notebooks) ----

    @property
    def node_count(self) -> int:
        """Number of unique entities in the graph."""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Number of (subject, predicate, object) triples in the graph."""
        return len(self._edges)
