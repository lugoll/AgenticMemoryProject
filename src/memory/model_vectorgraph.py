from __future__ import annotations

import logging
import time

from src.config.cfg import Config
from src.telemetry import record_retrieval_event
from .base import BaseMemory
from .reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class VectorGraphMemory(BaseMemory):
    """
    Chunk-anchored graph retrieval view over the shared Neo4j store.

    Output is triples-only (comparable to GraphMemory); only the entity
    seeding differs. Entity-name vectors proved a weak semantic anchor for
    queries (a nickname or paraphrase rarely matches the embedding of the
    bare entity name), so seeds are found through the chunks instead:

      1. Embed the query with the shared embedder and match it against the
         ``chunk_vector`` index (the same signal Vector RAG retrieves with).
      2. Hop ``(:Chunk)-[:MENTIONS]->(:__Entity__)`` to collect seed entities.
      3. Grow reasoning chains via the shared beam traversal
         (BaseMemory._expand_beam): whole chains are scored and a per-node-capped
         top-K beam is kept; the top chains' triples are returned. The eval settled
         on the beam for vectorgraph's noisy chunk-anchored seeds.

    No LLM call at retrieval time.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        # Chunk anchoring reuses the Vector RAG parameters so the entry-point
        # signal is identical to the vector variant.
        self._chunk_top_k: int = config.retrieval.top_k
        self._cutoff: float = config.retrieval.similarity_cutoff
        self._reranker = CrossEncoderReranker(config)

    def search(self, query: str) -> list[str]:
        if not query.strip():
            return []

        t0 = time.perf_counter()
        query_embedding = self.embed_model.get_text_embedding(query)

        try:
            rows = self._cypher(
                "CALL db.index.vector.queryNodes($index, $top_k, $embedding) "
                "YIELD node, score "
                "WHERE score >= $cutoff "
                "MATCH (node)-[:MENTIONS]->(e:__Entity__) "
                "RETURN DISTINCT e.name AS name",
                index=self._vector_index,
                top_k=self._chunk_top_k,
                embedding=query_embedding,
                cutoff=self._cutoff,
            )
        except Exception as exc:
            # Index missing (ingest not run yet) — behave like an empty store.
            logger.warning("VectorGraphMemory: chunk seeding failed: %s", exc)
            return []

        seed_nodes = [r["name"] for r in rows if r["name"]]
        t_seed = time.perf_counter()

        if not seed_nodes:
            logger.debug("VectorGraphMemory: query=%r → no seed entities", query)
            return []

        # Beam traversal scores whole chains (no separate final rerank).
        chains, stats = self._expand_beam(query, seed_nodes, self._reranker)

        record_retrieval_event(
            "chunk_seed", self.get_backend_name(),
            duration_ms=(t_seed - t0) * 1000, seeds=len(seed_nodes),
        )
        record_retrieval_event(
            "graph_bfs", self.get_backend_name(),
            duration_ms=stats["bfs_ms"], seeds=len(seed_nodes),
            cypher_queries=stats["cypher_queries"], paths_scored=stats["paths_scored"],
        )
        record_retrieval_event(
            "rerank", self.get_backend_name(),
            duration_ms=stats["rerank_ms"],
            candidates=stats["paths_scored"], returned=stats["paths_kept"],
        )
        logger.debug(
            "VectorGraphMemory: query=%r → %d seeds → %d paths scored → %d chains",
            query, len(seed_nodes), stats["paths_scored"], stats["paths_kept"],
        )
        return chains

    def get_backend_name(self) -> str:
        return "vectorgraph"
