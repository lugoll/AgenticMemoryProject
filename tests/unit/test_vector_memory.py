"""Unit tests for VectorMemory — all external deps (chromadb, sentence-transformers) mocked."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.config.cfg import (
    Config, LLMCfg, LLMsCfg, EmbeddingCfg,
    RetrievalCfg, IngestionCfg, GraphCfg, StoresCfg, TelemetryCfg,
)
from src.memory.model_vector import _chunk_text


def _make_config(
    top_k: int = 3,
    chunk_size: int = 10,
    chunk_overlap: int = 2,
    similarity_cutoff: float = 0.0,
    collection: str = "test_collection",
) -> Config:
    _llm = LLMCfg(model="ollama/test", base_url="http://localhost:11434",
                  temperature=0.0, max_tokens=256)
    return Config(
        llm=LLMsCfg(agent=_llm, ingest=_llm, judge=_llm),
        embedding=EmbeddingCfg(model="BAAI/bge-base-en-v1.5", batch_size=8,
                               chroma_host="http://localhost:8000"),
        retrieval=RetrievalCfg(top_k=top_k, similarity_cutoff=similarity_cutoff),
        ingestion=IngestionCfg(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        graph=GraphCfg(max_hops=3),
        stores=StoresCfg(bm25="", graph="", vector=collection),
        telemetry=TelemetryCfg(output_dir="evaluations/"),
    )


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
    dummy_vector = [0.1] * 768

    mock_chroma = MagicMock()
    mock_chroma.HttpClient.return_value = mock_client

    mock_encoder = MagicMock()
    mock_encoder.encode.return_value = MagicMock(tolist=lambda: [dummy_vector])
    mock_st = MagicMock()
    mock_st.SentenceTransformer.return_value = mock_encoder

    with patch.dict(sys.modules, {"chromadb": mock_chroma, "sentence_transformers": mock_st}):
        from src.memory.model_vector import VectorMemory
        vm = VectorMemory(config=_make_config())
        vm._mock_collection = mock_collection
        vm._mock_encoder = mock_encoder
        yield vm


# ── _chunk_text ───────────────────────────────────────────────────────────────

def test_chunk_text_single_chunk():
    assert _chunk_text("one two three", chunk_size=10, chunk_overlap=2) == ["one two three"]


def test_chunk_text_produces_overlap():
    text = " ".join(str(i) for i in range(10))
    chunks = _chunk_text(text, chunk_size=6, chunk_overlap=2)
    assert chunks[0].split()[:6] == ["0", "1", "2", "3", "4", "5"]
    assert chunks[1].split()[:2] == ["4", "5"]


def test_chunk_text_empty_input():
    assert _chunk_text("", chunk_size=10, chunk_overlap=2) == []


def test_chunk_text_whitespace_only():
    assert _chunk_text("   ", chunk_size=10, chunk_overlap=2) == []


# ── VectorMemory ──────────────────────────────────────────────────────────────

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
    memory._config.retrieval.similarity_cutoff = 0.7
    memory._mock_collection.query.return_value = {
        "documents": [["doc_a", "doc_b"]],
        "distances": [[0.4, 0.8]],  # similarities: 0.8 and 0.6
    }
    assert memory.search("some query") == ["doc_a"]


def test_search_returns_up_to_top_k(memory):
    memory._mock_collection.count.return_value = 10
    memory._mock_collection.query.return_value = {
        "documents": [["a", "b", "c"]],
        "distances": [[0.1, 0.2, 0.3]],
    }
    assert len(memory.search("query")) <= _make_config().retrieval.top_k


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
    assert memory._client.get_or_create_collection.call_count >= 2


def test_get_backend_name(memory):
    assert memory.get_backend_name() == "vector"
