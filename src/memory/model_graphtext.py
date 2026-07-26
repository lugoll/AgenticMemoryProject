from __future__ import annotations

import logging
import time

from src.telemetry import record_retrieval_event
from .model_graph import GraphMemory

logger = logging.getLogger(__name__)


class GraphTextMemory(GraphMemory):
    """
    BM25-entity-seeded graph retrieval that returns source passages.

    Identical seeding and traversal to GraphMemory (BM25 entity linking →
    flat hop-by-hop BFS), but instead of returning the traversed triples it
    returns the source chunks those entities were extracted from
    (BaseMemory._collect_chunks: mention-overlap-ranked, capped at
    ``retrieval.rerank_fetch_k``), cross-encoder reranked to
    ``retrieval.top_k``. Exists to isolate the seeding signal: graphtext vs.
    vectorgraphtext differ *only* in how the BFS is seeded (BM25 entity
    linking vs. chunk-vector cosine), everything downstream is shared.
    No LLM call at retrieval time.
    """

    def search(self, query: str) -> list[str]:
        if not query.strip():
            return []

        t0 = time.perf_counter()
        seed_nodes = self._bm25_seeds(query)
        if not seed_nodes:
            return []
        t_seed = time.perf_counter()

        _triples, visited, stats = self._expand_triples(seed_nodes)
        t_bfs = time.perf_counter()

        chunks = self._collect_chunks(visited)
        t_collect = time.perf_counter()

        reranked = self._reranker.rerank(query, chunks)
        t_rerank = time.perf_counter()

        record_retrieval_event(
            "entity_link", self.get_backend_name(),
            duration_ms=(t_seed - t0) * 1000, seeds=len(seed_nodes),
        )
        record_retrieval_event(
            "graph_bfs", self.get_backend_name(),
            duration_ms=(t_bfs - t_seed) * 1000,
            seeds=len(seed_nodes), visited=len(visited), **stats,
        )
        record_retrieval_event(
            "chunk_collect", self.get_backend_name(),
            duration_ms=(t_collect - t_bfs) * 1000,
            entities=len(visited), returned=len(chunks),
        )
        record_retrieval_event(
            "rerank", self.get_backend_name(),
            duration_ms=(t_rerank - t_collect) * 1000,
            candidates=len(chunks), returned=len(reranked),
        )
        logger.debug(
            "GraphTextMemory: query=%r → %d seeds → %d visited → "
            "%d chunks → rerank top %d",
            query, len(seed_nodes), len(visited), len(chunks), len(reranked),
        )
        return reranked

    def get_backend_name(self) -> str:
        return "graphtext"

    @property
    def store_size(self) -> int:
        """Number of chunks currently stored (shared-store view)."""
        return self.chunk_count
