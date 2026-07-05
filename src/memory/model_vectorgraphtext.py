from __future__ import annotations

import logging
import time

from src.config.cfg import Config
from src.telemetry import record_retrieval_event
from .base import BaseMemory
from .reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class VectorGraphTextMemory(BaseMemory):
    """
    Vector-anchored graph retrieval view that returns source passages.

    Same LlamaIndex VectorContextRetriever path as the pre-rework vectorgraph
    (query embedded with the shared embedder, matched against the entity-node
    vector index, expanded to ``graph.max_hops``) but with ``include_text=True``:
    each retrieved subgraph node carries the source chunk text it was extracted
    from (via the MENTIONS links), not just the bare triples. Exists to measure
    the token/accuracy tradeoff of passages vs. triples-only graph context.
    A cross-encoder (shared CrossEncoderReranker) reranks before truncation.
    No LLM call at retrieval time.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._max_hops: int = config.graph.max_hops
        self._top_k: int = config.graph.top_k
        self._reranker = CrossEncoderReranker(config)

    def search(self, query: str) -> list[str]:
        from llama_index.core.indices.property_graph import VectorContextRetriever

        if not query.strip():
            return []

        t0 = time.perf_counter()
        retriever = VectorContextRetriever(
            self.graph_store,
            embed_model=self.embed_model,
            include_text=True,
            similarity_top_k=self._top_k,
            # Traverse to the same radius as the other graph variants
            # (config.graph.max_hops) so all pull comparable neighborhoods.
            path_depth=self._max_hops,
        )
        nodes = retriever.retrieve(query)
        candidates = [c for n in nodes if (c := n.get_content().strip())]
        t_retrieve = time.perf_counter()

        reranked = self._reranker.rerank(query, candidates)
        t_rerank = time.perf_counter()

        record_retrieval_event(
            "entity_vector_retrieve", self.get_backend_name(),
            duration_ms=(t_retrieve - t0) * 1000, candidates=len(candidates),
        )
        record_retrieval_event(
            "rerank", self.get_backend_name(),
            duration_ms=(t_rerank - t_retrieve) * 1000,
            candidates=len(candidates), returned=len(reranked),
        )
        logger.debug(
            "VectorGraphTextMemory: query=%r → %d candidates → rerank top %d",
            query, len(candidates), len(reranked),
        )
        return reranked

    def get_backend_name(self) -> str:
        return "vectorgraphtext"
