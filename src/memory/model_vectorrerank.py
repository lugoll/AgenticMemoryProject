from __future__ import annotations

import logging
import time

from src.config.cfg import Config
from src.telemetry import record_retrieval_event
from .base import BaseMemory
from .reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class VectorRerankMemory(BaseMemory):
    """
    Vector RAG retrieval view with a cross-encoder rerank stage.

    Same native Neo4j chunk vector index and local query embedding as
    ``VectorMemory``, but over-fetches a larger candidate pool
    (``retrieval.rerank_fetch_k``) and reranks it with the shared
    ``CrossEncoderReranker`` before truncating to ``retrieval.top_k``.

    Exists to isolate the effect of reranking: it puts pure Vector RAG through
    the *identical* ranking stage the graph variants use (same cross-encoder,
    same final top-n), so any accuracy/token difference between vector and graph
    is not confounded by one path reranking and the other not. Like every
    variant the rerank stage issues no LLM call → zero retrieval tokens; only
    the wall-clock cost is reported via retrieval_overhead telemetry.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._fetch_k: int = config.retrieval.rerank_fetch_k
        self._cutoff: float = config.retrieval.similarity_cutoff
        self._reranker = CrossEncoderReranker(config)

    def search(self, query: str) -> list[str]:
        """Over-fetch top candidates by cosine, then cross-encoder rerank to top_n."""
        if not query.strip():
            return []

        t0 = time.perf_counter()
        # get_text_embedding (not get_query_embedding) to avoid any
        # instruction-prefix formatting — matches VectorMemory's raw behaviour.
        query_embedding = self.embed_model.get_text_embedding(query)

        try:
            rows = self._cypher(
                "CALL db.index.vector.queryNodes($index, $top_k, $embedding) "
                "YIELD node, score "
                "RETURN node.text AS text, score",
                index=self._vector_index,
                top_k=self._fetch_k,
                embedding=query_embedding,
            )
        except Exception as exc:
            # Index missing (ingest not run yet) — behave like an empty store.
            logger.warning("VectorRerankMemory: vector query failed: %s", exc)
            return []

        candidates = [
            r["text"]
            for r in rows
            if r["score"] >= self._cutoff and r["text"] and r["text"].strip()
        ]
        t_retrieve = time.perf_counter()

        reranked = self._reranker.rerank(query, candidates)
        t_rerank = time.perf_counter()

        record_retrieval_event(
            "vector_query", self.get_backend_name(),
            duration_ms=(t_retrieve - t0) * 1000,
            candidates=len(rows), returned=len(candidates),
        )
        record_retrieval_event(
            "rerank", self.get_backend_name(),
            duration_ms=(t_rerank - t_retrieve) * 1000,
            candidates=len(candidates), returned=len(reranked),
        )
        logger.debug(
            "VectorRerankMemory: query=%r → %d/%d above cutoff %.2f → rerank top %d",
            query, len(candidates), len(rows), self._cutoff, len(reranked),
        )
        return reranked

    def get_backend_name(self) -> str:
        return "vectorrerank"

    @property
    def store_size(self) -> int:
        """Number of chunks currently stored (shared-store view)."""
        return self.chunk_count
