from __future__ import annotations

import logging
import time

from src.config.cfg import Config
from src.telemetry import record_retrieval_event
from .base import BaseMemory

logger = logging.getLogger(__name__)


class VectorMemory(BaseMemory):
    """
    Vector RAG retrieval view over the shared Neo4j store.

    Queries the native Neo4j vector index over ``Chunk.embedding`` (cosine).
    The query is embedded locally with the shared sentence-transformer — no
    LLM call, no API key.

    Score semantics: Neo4j's cosine vector index reports
    ``score = (1 + cos) / 2`` in [0, 1] — identical to the previous ChromaDB
    conversion ``1 - distance/2``, so ``retrieval.similarity_cutoff`` keeps
    its meaning unchanged.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._top_k: int = config.retrieval.top_k
        self._cutoff: float = config.retrieval.similarity_cutoff

    def search(self, query: str) -> list[str]:
        """Return up to top_k chunks by cosine similarity, filtered by cutoff."""
        if not query.strip():
            return []

        t0 = time.perf_counter()
        # get_text_embedding (not get_query_embedding) to avoid any
        # instruction-prefix formatting — matches the previous raw
        # SentenceTransformer.encode behaviour.
        query_embedding = self.embed_model.get_text_embedding(query)

        try:
            rows = self._cypher(
                "CALL db.index.vector.queryNodes($index, $top_k, $embedding) "
                "YIELD node, score "
                "RETURN node.text AS text, score",
                index=self._vector_index,
                top_k=self._top_k,
                embedding=query_embedding,
            )
        except Exception as exc:
            # Index missing (ingest not run yet) — behave like an empty store.
            logger.warning("VectorMemory: vector query failed: %s", exc)
            return []

        passages = [
            r["text"]
            for r in rows
            if r["score"] >= self._cutoff and r["text"] and r["text"].strip()
        ]
        record_retrieval_event(
            "vector_query", self.get_backend_name(),
            duration_ms=(time.perf_counter() - t0) * 1000,
            candidates=len(rows), returned=len(passages),
        )
        logger.debug(
            "VectorMemory: query=%r → %d/%d results above cutoff %.2f",
            query, len(passages), len(rows), self._cutoff,
        )
        return passages

    def get_backend_name(self) -> str:
        return "vector"

    @property
    def store_size(self) -> int:
        """Number of chunks currently stored (shared-store view)."""
        return self.chunk_count
