from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from src.memory.base import BaseMemory
from src.memory.model_lightrag import (
    LightRAGMemory,
    _TELEMETRY,
    _ingest_with_context,
    _query_with_context,
    _update_with_context,
)


# ---- Helpers ----

def _make_config(memory_path: str | None = None, top_k: int = 5) -> MagicMock:
    config = MagicMock()
    config.memory_path = memory_path
    config.retrieval.top_k = top_k
    config.llm.model = "ollama/test-model"
    config.llm.temperature = 0.0
    config.llm.max_tokens = 512
    config.variant_name = "test_variant"
    config.lightrag = MagicMock()
    config.lightrag.query_mode = "hybrid"
    config.lightrag.embedding_dim = 768
    config.lightrag.max_token_size = 8192
    return config


def _make_mock_rag(query_result: str = "some answer") -> MagicMock:
    mock = MagicMock()
    mock.initialize_storages = AsyncMock(return_value=None)
    mock.ainsert = AsyncMock(return_value=None)
    mock.aquery = AsyncMock(return_value=query_result)
    return mock


def _build_mem(mock_rag_cls, tmp_path: Path, top_k: int = 5, query_result: str = "result") -> LightRAGMemory:
    mock_rag = _make_mock_rag(query_result=query_result)
    mock_rag_cls.return_value = mock_rag
    config = _make_config(memory_path=str(tmp_path / "rag"), top_k=top_k)
    return LightRAGMemory(config=config), mock_rag


# ---- Contract ----

class TestLightRAGMemoryContract:
    def test_is_subclass_of_base_memory(self) -> None:
        assert issubclass(LightRAGMemory, BaseMemory)

    @patch("src.memory.model_lightrag.LightRAG")
    def test_get_backend_name(self, mock_rag_cls, tmp_path) -> None:
        mem, _ = _build_mem(mock_rag_cls, tmp_path)
        assert mem.get_backend_name() == "lightrag"


# ---- Ingest ----

class TestLightRAGMemoryIngest:
    @patch("src.memory.model_lightrag.LightRAG")
    def test_ingest_calls_ainsert_with_documents(self, mock_rag_cls, tmp_path) -> None:
        mem, mock_rag = _build_mem(mock_rag_cls, tmp_path)
        mem.ingest_documents(["doc one", "doc two"])
        mock_rag.ainsert.assert_called_once_with(["doc one", "doc two"])

    @patch("src.memory.model_lightrag.LightRAG")
    def test_ingest_skips_empty_strings(self, mock_rag_cls, tmp_path) -> None:
        mem, mock_rag = _build_mem(mock_rag_cls, tmp_path)
        mem.ingest_documents(["", "  ", "\t"])
        mock_rag.ainsert.assert_not_called()

    @patch("src.memory.model_lightrag.LightRAG")
    def test_ingest_filters_out_whitespace_only(self, mock_rag_cls, tmp_path) -> None:
        mem, mock_rag = _build_mem(mock_rag_cls, tmp_path)
        mem.ingest_documents(["  ", "real doc", ""])
        mock_rag.ainsert.assert_called_once_with(["real doc"])

    @patch("src.memory.model_lightrag.LightRAG")
    def test_ingest_empty_list_is_noop(self, mock_rag_cls, tmp_path) -> None:
        mem, mock_rag = _build_mem(mock_rag_cls, tmp_path)
        mem.ingest_documents([])
        mock_rag.ainsert.assert_not_called()


# ---- Search ----

class TestLightRAGMemorySearch:
    @patch("src.memory.model_lightrag.LightRAG")
    def test_search_returns_list_with_result(self, mock_rag_cls, tmp_path) -> None:
        mem, _ = _build_mem(mock_rag_cls, tmp_path, query_result="answer text")
        result = mem.search("who is marie curie?")
        assert result == ["answer text"]

    @patch("src.memory.model_lightrag.LightRAG")
    def test_search_empty_string_returns_empty(self, mock_rag_cls, tmp_path) -> None:
        mem, mock_rag = _build_mem(mock_rag_cls, tmp_path)
        result = mem.search("   ")
        assert result == []
        mock_rag.aquery.assert_not_called()

    @patch("src.memory.model_lightrag.LightRAG")
    def test_search_empty_rag_result_returns_empty(self, mock_rag_cls, tmp_path) -> None:
        mem, _ = _build_mem(mock_rag_cls, tmp_path, query_result="")
        result = mem.search("some query")
        assert result == []

    @patch("src.memory.model_lightrag.LightRAG")
    def test_search_whitespace_rag_result_returns_empty(self, mock_rag_cls, tmp_path) -> None:
        mem, _ = _build_mem(mock_rag_cls, tmp_path, query_result="   ")
        result = mem.search("some query")
        assert result == []

    @patch("src.memory.model_lightrag.LightRAG")
    def test_search_passes_query_mode_and_top_k(self, mock_rag_cls, tmp_path) -> None:
        from lightrag import QueryParam
        mem, mock_rag = _build_mem(mock_rag_cls, tmp_path, top_k=3)
        mem.search("query")
        args, kwargs = mock_rag.aquery.call_args
        param: QueryParam = args[1] if len(args) > 1 else kwargs.get("param")
        assert param.mode == "hybrid"
        assert param.top_k == 3


# ---- update_fact ----

class TestLightRAGMemoryUpdateFact:
    @patch("src.memory.model_lightrag.LightRAG")
    def test_update_fact_calls_ainsert(self, mock_rag_cls, tmp_path) -> None:
        mem, mock_rag = _build_mem(mock_rag_cls, tmp_path)
        mem.update_fact("Paris is the capital of France")
        mock_rag.ainsert.assert_called_once_with("Paris is the capital of France")

    @patch("src.memory.model_lightrag.LightRAG")
    def test_update_fact_empty_is_noop(self, mock_rag_cls, tmp_path) -> None:
        mem, mock_rag = _build_mem(mock_rag_cls, tmp_path)
        mem.update_fact("   ")
        mock_rag.ainsert.assert_not_called()

    @patch("src.memory.model_lightrag.LightRAG")
    def test_update_fact_empty_string_is_noop(self, mock_rag_cls, tmp_path) -> None:
        mem, mock_rag = _build_mem(mock_rag_cls, tmp_path)
        mem.update_fact("")
        mock_rag.ainsert.assert_not_called()


# ---- Reset ----

def _make_mock_rag_with_storages(query_result: str = "") -> MagicMock:
    """Build a mock LightRAG that also has the named storage attributes reset() iterates."""
    mock = _make_mock_rag(query_result=query_result)
    for attr in (
        "full_docs", "text_chunks", "full_entities", "full_relations",
        "entity_chunks", "relation_chunks", "llm_response_cache", "doc_status",
        "entities_vdb", "relationships_vdb", "chunks_vdb", "chunk_entity_relation_graph",
    ):
        storage = MagicMock()
        storage.drop = AsyncMock(return_value={"status": "success", "message": "data dropped"})
        setattr(mock, attr, storage)
    return mock


class TestLightRAGMemoryReset:
    @patch("src.memory.model_lightrag.LightRAG")
    def test_reset_calls_drop_on_all_storages(self, mock_rag_cls, tmp_path) -> None:
        mock_rag = _make_mock_rag_with_storages()
        mock_rag_cls.return_value = mock_rag
        config = _make_config(memory_path=str(tmp_path / "rag"))
        mem = LightRAGMemory(config=config)

        mem.reset()

        for attr in (
            "full_docs", "text_chunks", "full_entities", "full_relations",
            "entity_chunks", "relation_chunks", "llm_response_cache", "doc_status",
            "entities_vdb", "relationships_vdb", "chunks_vdb", "chunk_entity_relation_graph",
        ):
            getattr(mock_rag, attr).drop.assert_called_once()

    @patch("src.memory.model_lightrag.LightRAG")
    def test_reset_reuses_same_rag_instance(self, mock_rag_cls, tmp_path) -> None:
        mock_rag = _make_mock_rag_with_storages()
        mock_rag_cls.return_value = mock_rag
        config = _make_config(memory_path=str(tmp_path / "rag"))
        mem = LightRAGMemory(config=config)

        mem.reset()

        assert mem._rag is mock_rag

    @patch("src.memory.model_lightrag.LightRAG")
    def test_reset_then_search_returns_empty(self, mock_rag_cls, tmp_path) -> None:
        mock_rag = _make_mock_rag_with_storages(query_result="")
        mock_rag_cls.return_value = mock_rag
        config = _make_config(memory_path=str(tmp_path / "rag"))
        mem = LightRAGMemory(config=config)

        mem.reset()
        result = mem.search("anything")

        assert result == []


# ---- Telemetry dict ----

class TestLightRAGTelemetryDict:
    """Verify that each operation writes correct phase/actor values to _TELEMETRY.

    Tests run the coroutine wrappers directly with a mock LightRAG and inspect
    _TELEMETRY inside the mock's async call to mirror how llm_func reads it.
    """

    def test_ingest_sets_phase_and_actor(self) -> None:
        captured: dict = {}

        async def _run():
            mock_rag = MagicMock()

            async def capture_insert(docs):
                captured.update(_TELEMETRY)

            mock_rag.ainsert = capture_insert
            await _ingest_with_context(mock_rag, ["doc"], "test_variant")

        asyncio.run(_run())
        assert captured["phase"] == "ingest"
        assert captured["actor"] == "graph_extract"
        assert captured["run_id"] == "ingest"

    def test_query_sets_phase_and_actor(self) -> None:
        from lightrag import QueryParam

        captured: dict = {}

        async def _run():
            mock_rag = MagicMock()

            async def capture_query(query, param=None):
                captured.update(_TELEMETRY)
                return "result"

            mock_rag.aquery = capture_query
            await _query_with_context(mock_rag, "q", QueryParam(mode="hybrid"), run_id="abc123")

        asyncio.run(_run())
        assert captured["phase"] == "retrieval_overhead"
        assert captured["actor"] == "graph_cypher_gen"
        assert captured["run_id"] == "abc123"

    def test_update_fact_sets_phase_and_actor(self) -> None:
        captured: dict = {}

        async def _run():
            mock_rag = MagicMock()

            async def capture_insert(fact):
                captured.update(_TELEMETRY)

            mock_rag.ainsert = capture_insert
            await _update_with_context(mock_rag, "new fact")

        asyncio.run(_run())
        assert captured["phase"] == "agent_reasoning"
        assert captured["actor"] == "graph_extract"
        assert captured["run_id"] == "update_fact"

    def test_phases_are_distinct_across_operations(self) -> None:
        from lightrag import QueryParam

        phases: list[str] = []

        async def _run():
            mock_rag = MagicMock()

            async def record_ingest(docs):
                phases.append(_TELEMETRY["phase"])

            async def record_query(query, param=None):
                phases.append(_TELEMETRY["phase"])
                return ""

            async def record_update(fact):
                phases.append(_TELEMETRY["phase"])

            mock_rag.ainsert = record_ingest
            await _ingest_with_context(mock_rag, ["d"], "v")
            mock_rag.ainsert = record_update
            await _update_with_context(mock_rag, "fact")
            mock_rag.aquery = record_query
            await _query_with_context(mock_rag, "q", QueryParam(), run_id="x")

        asyncio.run(_run())
        assert phases == ["ingest", "agent_reasoning", "retrieval_overhead"]
