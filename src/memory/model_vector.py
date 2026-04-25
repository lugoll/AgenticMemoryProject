from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from .base import BaseMemory

if TYPE_CHECKING:
    from src.config.settings import ExperimentConfig

logger = logging.getLogger(__name__)


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping word-boundary chunks.

    chunk_size and chunk_overlap are measured in words, not characters,
    which is more stable across different paragraph lengths.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - chunk_overlap

    return chunks


class VectorMemory(BaseMemory):
    """
    ChromaDB HTTP-backed memory with sentence-transformer embeddings.

    Connects to a ChromaDB HTTP server (separate Docker container) so the
    vector index survives container restarts and is isolated from the app process.

    Embeddings are computed locally inside the app container via
    sentence-transformers — no LLM call required, no API key.
    Vectors are passed pre-computed to ChromaDB (chromadb-client, no C++ needed).

    Collection name comes from config.memory_path so variants coexist on the
    same ChromaDB server without collision.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        # Lazy imports: chromadb and sentence-transformers are Docker-only deps.
        # Importing here (not at module level) lets other modules import
        # model_vector without these packages being installed locally.
        import chromadb
        from sentence_transformers import SentenceTransformer

        self._config = config
        self._collection_name: str = config.memory_path or "vector_default"

        # Strip "huggingface/" prefix — SentenceTransformer expects bare model ID.
        raw_model: str = config.embedding.model
        model_id = raw_model.removeprefix("huggingface/")

        self._encoder = SentenceTransformer(model_id, device="cpu")

        chroma_host = config.embedding.chroma_host or "http://localhost:8000"
        self._client = chromadb.HttpClient(host=chroma_host)

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.debug(
            "VectorMemory: connected to %s, collection=%r, size=%d",
            chroma_host,
            self._collection_name,
            self._collection.count(),
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings and return as plain Python float lists."""
        vectors = self._encoder.encode(texts, show_progress_bar=False)
        return vectors.tolist()

    # ---- BaseMemory interface ----

    def ingest_documents(self, documents: list[str]) -> None:
        """Chunk documents, embed, and upsert into ChromaDB in batches."""
        chunk_size = self._config.ingestion.chunk_size
        chunk_overlap = self._config.ingestion.chunk_overlap
        batch_size = self._config.embedding.batch_size

        all_chunks: list[str] = []
        for doc in documents:
            stripped = doc.strip()
            if stripped:
                all_chunks.extend(_chunk_text(stripped, chunk_size, chunk_overlap))

        if not all_chunks:
            logger.warning("VectorMemory: no chunks produced from %d documents", len(documents))
            return

        for batch_start in range(0, len(all_chunks), batch_size):
            batch = all_chunks[batch_start : batch_start + batch_size]
            embeddings = self._embed(batch)
            ids = [str(uuid.uuid4()) for _ in batch]
            self._collection.upsert(documents=batch, embeddings=embeddings, ids=ids)

        logger.debug(
            "VectorMemory: ingested %d documents → %d chunks, store size=%d",
            len(documents),
            len(all_chunks),
            self._collection.count(),
        )

    def search(self, query: str) -> list[str]:
        """Return up to top_k passages by cosine similarity, filtered by cutoff."""
        if not query.strip():
            return []

        count = self._collection.count()
        if count == 0:
            return []

        top_k = self._config.retrieval.top_k
        cutoff = self._config.retrieval.similarity_cutoff

        query_embedding = self._embed([query])[0]
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            include=["documents", "distances"],
        )

        passages: list[str] = []
        docs = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, dist in zip(docs, distances):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite.
            # Convert to similarity: similarity = 1 - (distance / 2).
            similarity = 1.0 - (dist / 2.0)
            if similarity >= cutoff:
                passages.append(doc)

        logger.debug(
            "VectorMemory: query=%r → %d/%d results above cutoff %.2f",
            query,
            len(passages),
            len(docs),
            cutoff,
        )
        return passages

    def update_fact(self, fact: str) -> None:
        """Embed and insert a single fact, making it immediately searchable."""
        stripped = fact.strip()
        if stripped:
            embedding = self._embed([stripped])[0]
            self._collection.upsert(
                documents=[stripped],
                embeddings=[embedding],
                ids=[str(uuid.uuid4())],
            )
            logger.debug("VectorMemory: updated fact=%r", stripped)

    def reset(self) -> None:
        """Delete the collection and recreate it empty."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.debug("VectorMemory: reset — collection %r recreated", self._collection_name)

    def get_backend_name(self) -> str:
        return "vector"

    @property
    def store_size(self) -> int:
        return self._collection.count()
