from __future__ import annotations

import logging
import re
import time

from src.config.cfg import Config
from src.telemetry import record_retrieval_event
from .base import BaseMemory

logger = logging.getLogger(__name__)

# Characters with special meaning in Lucene query syntax (Neo4j full-text
# indexes are Lucene-backed): + - && || ! ( ) { } [ ] ^ " ~ * ? : \ /
_LUCENE_SPECIAL = re.compile(r'[+\-!(){}\[\]^"~*?:\\/&|]')


def _sanitise_query(raw: str) -> str:
    """Turn a natural-language query into a safe Lucene query string.

    Strategy: strip Lucene operators, keep only non-empty tokens, and join
    them with OR so that chunks matching *any* query term are returned
    (Lucene's BM25 scoring still ranks multi-term matches higher).
    """
    cleaned = _LUCENE_SPECIAL.sub(" ", raw)
    tokens = [t for t in cleaned.split() if t]
    if not tokens:
        return ""
    # Wrap each token in double quotes to treat it as a literal phrase,
    # then combine with OR for broad recall.
    return " OR ".join(f'"{t}"' for t in tokens)


class BM25Memory(BaseMemory):
    """
    BM25 retrieval view over the shared Neo4j store.

    Searches the ``:Chunk`` nodes written by the unified ingest through the
    Lucene full-text index (BM25-family scoring). No embeddings and no LLM
    calls at retrieval time.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._top_k: int = config.retrieval.top_k

    def search(self, query: str) -> list[str]:
        """Return up to top_k chunk texts ranked by Lucene BM25 relevance."""
        if not query.strip():
            return []

        fts_query = _sanitise_query(query)
        if not fts_query:
            return []

        t0 = time.perf_counter()
        try:
            rows = self._cypher(
                "CALL db.index.fulltext.queryNodes($index, $fts_query) "
                "YIELD node, score "
                "RETURN node.text AS text "
                "ORDER BY score DESC LIMIT $top_k",
                index=self._fulltext_index,
                fts_query=fts_query,
                top_k=self._top_k,
            )
        except Exception as exc:
            # Index missing (ingest not run yet) — behave like an empty store.
            logger.warning("BM25Memory: full-text query failed: %s", exc)
            return []

        results = [r["text"] for r in rows if r["text"] and r["text"].strip()]
        record_retrieval_event(
            "fulltext_query", self.get_backend_name(),
            duration_ms=(time.perf_counter() - t0) * 1000, returned=len(results),
        )
        logger.debug("BM25Memory: query=%r returned %d results", query, len(results))
        return results

    def get_backend_name(self) -> str:
        return "bm25"

    @property
    def store_size(self) -> int:
        """Number of chunks currently stored (shared-store view)."""
        return self.chunk_count
