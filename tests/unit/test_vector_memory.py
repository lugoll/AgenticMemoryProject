"""
Unit tests for VectorMemory.

chromadb and sentence-transformers are Docker-only dependencies and are not
installed in the local dev environment. All external calls are mocked so these
tests run without any network connection or C++ build tools.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.config.settings import (
    EmbeddingConfig,
    ExperimentConfig,
    IngestionConfig,
    LLMConfig,
    RetrievalConfig,
    TelemetryConfig,
)
from src.memory.model_vector import _chunk_text


# ── Config helper ─────────────────────────────────────────────────────────────

def _make_config(
    top_k: int = 3,
    chunk_size: int = 10,
    chunk_overlap: int = 2,
    similarity_cutoff: float = 0.0,
    collection: str = "test_collection",
) -> ExperimentConfig:
    return ExperimentConfig(
        variant_name="test_vector",
        agent_type="vector",
        data_file="dummy.json",
        memory_path=collection,
        llm=LLMConfig(model="ollama/test", api_base="http://localhost:11434"),
        embedding=EmbeddingConfig(
            model="huggingface/BAAI/bge-base-en-v1.5",
            batch_size=8,
            chroma_host="http://localhost:8000",
        ),
        retrieval=RetrievalConfig(top_k=top_k, similarity_cutoff=similarity_cutoff),
        ingestion=IngestionConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        telemetry=TelemetryConfig(),
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_collection():
    col = MagicMock()
    col.count.return_value = 0
    return col


@pytest.fixture()
def mock_client(mock_collection):
    client = MagicMock()
    client.get_or_create_collection.return_value = mock_collection
    return client


@pytest.fixture()
def memory(mock_client, mock_collection):
    """VectorMemory with all external deps (chromadb, sentence_transformers) mocked.

    chromadb and sentence_transformers are Docker-only; they are not installed
    locally. We inject mocks via sys.modules so that the lazy `import chromadb`
    inside VectorMemory.__init__ picks up our stubs instead of the real packages.
    """
    import sys

    dummy_vector = [0.1] * 768

    mock_chroma = MagicMock()
    mock_chroma.HttpClient.return_value = mock_client

    mock_encoder = MagicMock()
    mock_encoder.encode.return_value = MagicMock(tolist=lambda: [dummy_vector])
    mock_st_module = MagicMock()
    mock_st_module.SentenceTransformer.return_value = mock_encoder

    with patch.dict(sys.modules, {
        "chromadb": mock_chroma,
        "sentence_transformers": mock_st_module,
    }):
        from src.memory.model_vector import VectorMemory
        vm = VectorMemory(config=_make_config())
        vm._mock_collection = mock_collection
        vm._mock_encoder = mock_encoder
        yield vm


# ── _chunk_text unit tests ────────────────────────────────────────────────────

def test_chunk_text_single_chunk():
    result = _chunk_text("one two three", chunk_size=10, chunk_overlap=2)
    assert result == ["one two three"]


def test_chunk_text_produces_overlap():
    text = " ".join(str(i) for i in range(10))  # "0 1 2 3 4 5 6 7 8 9"
    chunks = _chunk_text(text, chunk_size=6, chunk_overlap=2)
    assert chunks[0].split()[:6] == ["0", "1", "2", "3", "4", "5"]
    assert chunks[1].split()[:2] == ["4", "5"]


def test_chunk_text_empty_input():
    assert _chunk_text("", chunk_size=10, chunk_overlap=2) == []


def test_chunk_text_whitespace_only():
    assert _chunk_text("   ", chunk_size=10, chunk_overlap=2) == []


# ── VectorMemory tests ────────────────────────────────────────────────────────

def test_store_size_delegates_to_collection(memory):
    memory._mock_collection.count.return_value = 42
    assert memory.store_size == 42


def test_ingest_calls_upsert(memory):
    memory._mock_collection.count.return_value = 0
    memory.ingest_documents(["The Eiffel Tower is in Paris."])
    assert memory._mock_collection.upsert.called


def test_ingest_skips_empty_strings(memory):
    memory.ingest_documents(["", "   "])
    memory._mock_collection.upsert.assert_not_called()


def test_search_returns_empty_on_empty_query(memory):
    assert memory.search("") == []
    assert memory.search("   ") == []


def test_search_returns_empty_when_store_empty(memory):
    memory._mock_collection.count.return_value = 0
    assert memory.search("anything") == []


def test_search_filters_by_cutoff(memory):
    memory._mock_collection.count.return_value = 3
    # distance=0.8 → similarity=0.6, below default cutoff=0.0 passes; set cutoff=0.7
    memory._config.retrieval.similarity_cutoff = 0.7
    memory._mock_collection.query.return_value = {
        "documents": [["doc_a", "doc_b"]],
        "distances": [[0.4, 0.8]],  # similarities: 0.8 and 0.6
    }
    results = memory.search("some query")
    assert results == ["doc_a"]  # only doc_a passes cutoff 0.7


def test_search_returns_up_to_top_k(memory):
    memory._mock_collection.count.return_value = 10
    memory._mock_collection.query.return_value = {
        "documents": [["a", "b", "c"]],
        "distances": [[0.1, 0.2, 0.3]],
    }
    results = memory.search("query")
    assert len(results) <= _make_config().retrieval.top_k


def test_update_fact_calls_upsert(memory):
    memory.update_fact("Napoleon was exiled to Saint Helena.")
    assert memory._mock_collection.upsert.called


def test_update_fact_ignores_empty(memory):
    memory.update_fact("")
    memory.update_fact("   ")
    memory._mock_collection.upsert.assert_not_called()


def test_reset_deletes_and_recreates_collection(memory):
    memory.reset()
    assert memory._client.delete_collection.called
    assert memory._client.get_or_create_collection.call_count >= 2  # init + reset


def test_get_backend_name(memory):
    assert memory.get_backend_name() == "vector"
