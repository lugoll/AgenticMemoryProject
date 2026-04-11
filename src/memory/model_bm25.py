from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path

from .base import BaseMemory

logger = logging.getLogger(__name__)

# Characters that have special meaning in FTS5 query syntax.
_FTS5_SPECIAL = re.compile(r'["\*\(\)\-\+\^:]')


def _sanitise_query(raw: str) -> str:
    """Turn a natural-language query into a safe FTS5 query string.

    Strategy: strip FTS5 operators, keep only non-empty tokens, and join
    them with OR so that documents matching *any* query term are returned
    (BM25 still ranks multi-term matches higher).
    """
    cleaned = _FTS5_SPECIAL.sub(" ", raw)
    tokens = [t for t in cleaned.split() if t]
    if not tokens:
        return ""
    # Wrap each token in double quotes to treat it as a literal phrase,
    # then combine with OR for broad recall.
    return " OR ".join(f'"{t}"' for t in tokens)


class BM25Memory(BaseMemory):
    """
    SQLite FTS5 memory backend with BM25 ranking.

    Uses a single FTS5 virtual table for storage and retrieval. Documents are
    inserted as rows; search queries are ranked by the built-in BM25 scoring
    function. No embeddings or LLM calls are required at any phase.

    When storage_path is provided the SQLite database is persisted to that file
    so that the ingestion step and the QA step can share the same memory state.
    If storage_path is None the database lives only in memory.
    """

    def __init__(self, top_k: int = 5, storage_path: Path | None = None) -> None:
        self._top_k: int = top_k
        self._storage_path: Path | None = storage_path
        self._conn: sqlite3.Connection = self._open()
        self._ensure_table()

    # ---- Connection helpers ----

    def _open(self) -> sqlite3.Connection:
        if self._storage_path is not None:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._storage_path))
        else:
            conn = sqlite3.connect(":memory:")
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _ensure_table(self) -> None:
        self._conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS documents USING fts5(content);"
        )
        self._conn.commit()

    # ---- BaseMemory interface ----

    def ingest_documents(self, documents: list[str]) -> None:
        """Insert non-empty documents into the FTS5 table."""
        rows = [(doc.strip(),) for doc in documents if doc.strip()]
        if rows:
            self._conn.executemany("INSERT INTO documents(content) VALUES (?);", rows)
            self._conn.commit()
        logger.debug(
            "BM25Memory: ingested %d documents, store size=%d",
            len(rows),
            self.store_size,
        )

    def search(self, query: str) -> list[str]:
        """Return up to top_k passages ranked by BM25 relevance."""
        if not query.strip():
            return []

        fts_query = _sanitise_query(query)
        if not fts_query:
            return []

        cursor = self._conn.execute(
            "SELECT content FROM documents WHERE documents MATCH ? "
            "ORDER BY bm25(documents) LIMIT ?;",
            (fts_query, self._top_k),
        )
        results = [row[0] for row in cursor.fetchall()]
        logger.debug("BM25Memory: query=%r returned %d results", query, len(results))
        return results

    def update_fact(self, fact: str) -> None:
        """Insert a single fact, making it immediately searchable."""
        stripped = fact.strip()
        if stripped:
            self._conn.execute("INSERT INTO documents(content) VALUES (?);", (stripped,))
            self._conn.commit()
            logger.debug(
                "BM25Memory: updated fact=%r, store size=%d",
                stripped,
                self.store_size,
            )

    def reset(self) -> None:
        """Drop the FTS5 table and recreate it. Delete the DB file if file-backed."""
        self._conn.execute("DROP TABLE IF EXISTS documents;")
        self._conn.commit()
        self._ensure_table()

        if self._storage_path and self._storage_path.exists():
            self._conn.close()
            self._storage_path.unlink()
            self._conn = self._open()
            self._ensure_table()

        logger.debug("BM25Memory: reset — store cleared")

    def get_backend_name(self) -> str:
        return "bm25"

    # ---- Introspection ----

    @property
    def store_size(self) -> int:
        """Number of documents currently stored."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM documents;")
        return cursor.fetchone()[0]
