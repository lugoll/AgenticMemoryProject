from __future__ import annotations

import json
import logging
from pathlib import Path

from .base import BaseMemory

logger = logging.getLogger(__name__)


class DummyMemory(BaseMemory):
    """
    In-memory dummy backend for testing agent logic without real LLMs.

    Uses a Python list as the document store and token-count-scored substring
    matching for retrieval. Not suitable for production use.

    When storage_path is provided, the store is persisted as a JSON file so
    that the ingestion step and the QA step (separate processes/runs) can share
    the same memory state. If storage_path is None the store lives only in
    memory for the lifetime of the object.

    Attributes:
        _store: Ordered list of all ingested text passages and facts.
        _top_k: Maximum number of results returned by search().
        _storage_path: Optional file path for persistence.
    """

    def __init__(self, top_k: int = 5, storage_path: Path | None = None) -> None:
        self._top_k: int = top_k
        self._storage_path: Path | None = storage_path
        self._store: list[str] = self._load() if storage_path else []

    # ---- Persistence helpers ----

    def _load(self) -> list[str]:
        """Load store from file. Returns empty list if file does not exist."""
        assert self._storage_path is not None
        if self._storage_path.exists():
            data = json.loads(self._storage_path.read_text(encoding="utf-8"))
            logger.debug("DummyMemory: loaded %d entries from %s", len(data), self._storage_path)
            return data
        return []

    def _save(self) -> None:
        """Persist store to file."""
        if self._storage_path is None:
            return
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._storage_path.write_text(
            json.dumps(self._store, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.debug("DummyMemory: saved %d entries to %s", len(self._store), self._storage_path)

    # ---- BaseMemory interface ----

    def ingest_documents(self, documents: list[str]) -> None:
        """Append non-empty documents to the store and persist."""
        for doc in documents:
            stripped = doc.strip()
            if stripped:
                self._store.append(stripped)
        self._save()
        logger.debug(
            "DummyMemory: ingested %d documents, store size=%d",
            len(documents),
            len(self._store),
        )

    def search(self, query: str) -> list[str]:
        """
        Return up to top_k passages ranked by how many query tokens they contain.

        Each passage scores +1 per query token found (case-insensitive).
        Zero-score passages are excluded entirely.
        """
        if not query.strip():
            return []

        query_tokens = [t.lower() for t in query.split() if t]
        scored: list[tuple[int, str]] = []

        for passage in self._store:
            passage_lower = passage.lower()
            score = sum(1 for token in query_tokens if token in passage_lower)
            if score > 0:
                scored.append((score, passage))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [passage for _, passage in scored[: self._top_k]]

        logger.debug(
            "DummyMemory: query=%r returned %d results", query, len(results)
        )
        return results

    def update_fact(self, fact: str) -> None:
        """Append the fact to the store, making it immediately searchable."""
        stripped = fact.strip()
        if stripped:
            self._store.append(stripped)
            self._save()
            logger.debug(
                "DummyMemory: updated fact=%r, store size=%d",
                stripped,
                len(self._store),
            )

    def reset(self) -> None:
        """Clear the store in memory and delete the persistence file if present."""
        self._store = []
        if self._storage_path and self._storage_path.exists():
            self._storage_path.unlink()
        logger.debug("DummyMemory: reset — store cleared")

    def get_backend_name(self) -> str:
        return "dummy"

    @property
    def store_size(self) -> int:
        """Number of passages currently held. Useful in tests."""
        return len(self._store)
