from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.memory.base import BaseMemory
from src.memory.model_graph import GraphMemory


# ---- Helpers ----

def _make_config(memory_path: str | None = None, top_k: int = 5) -> MagicMock:
    """Build a minimal ExperimentConfig mock."""
    config = MagicMock()
    config.memory_path = memory_path
    config.retrieval.top_k = top_k
    config.llm.model = "ollama/test-model"
    config.llm.temperature = 0.0
    config.llm.max_tokens = 512
    config.variant_name = "test_variant"
    return config


def _litellm_response(content: str) -> MagicMock:
    """Simulate a litellm.completion() return value."""
    resp = MagicMock()
    resp.choices[0].message.content = content
    return resp


SINGLE_TRIPLE = '[{"subject": "Marie Curie", "predicate": "born_in", "object": "Warsaw"}]'
TWO_TRIPLES = (
    '[{"subject": "Marie Curie", "predicate": "born_in", "object": "Warsaw"},'
    ' {"subject": "Warsaw", "predicate": "capital_of", "object": "Poland"}]'
)


# ---- Contract ----

class TestBaseMemoryContract:
    def test_graph_memory_is_subclass_of_base(self) -> None:
        assert issubclass(GraphMemory, BaseMemory)


# ---- Ingest ----

class TestGraphMemoryIngest:
    def test_ingest_calls_llm_per_document(self) -> None:
        config = _make_config()
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _litellm_response(SINGLE_TRIPLE)
            mem = GraphMemory(config=config)
            mem.ingest_documents(["doc one", "doc two"])
        assert mock_llm.call_count == 2

    def test_ingest_populates_nodes_and_edges(self) -> None:
        config = _make_config()
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _litellm_response(SINGLE_TRIPLE)
            mem = GraphMemory(config=config)
            mem.ingest_documents(["Marie Curie was born in Warsaw."])
        assert mem.node_count == 2
        assert mem.edge_count == 1

    def test_ingest_skips_empty_documents(self) -> None:
        config = _make_config()
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _litellm_response("[]")
            mem = GraphMemory(config=config)
            mem.ingest_documents(["", "   ", "\n"])
        assert mock_llm.call_count == 0

    def test_ingest_deduplicates_identical_triples(self) -> None:
        config = _make_config()
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _litellm_response(SINGLE_TRIPLE)
            mem = GraphMemory(config=config)
            mem.ingest_documents(["doc one", "doc two"])  # same triple returned twice
        assert mem.edge_count == 1

    def test_ingest_llm_tagged_correctly(self) -> None:
        config = _make_config()
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _litellm_response(SINGLE_TRIPLE)
            mem = GraphMemory(config=config)
            mem.ingest_documents(["some document"])
        call_kwargs = mock_llm.call_args.kwargs
        assert call_kwargs["metadata"]["phase"] == "ingest"
        assert call_kwargs["metadata"]["actor"] == "graph_extract"

    def test_ingest_invalid_json_from_llm_is_skipped(self) -> None:
        config = _make_config()
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _litellm_response("not valid json at all")
            mem = GraphMemory(config=config)
            mem.ingest_documents(["some document"])
        assert mem.edge_count == 0

    def test_ingest_non_array_json_from_llm_is_skipped(self) -> None:
        config = _make_config()
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _litellm_response('{"subject": "X"}')
            mem = GraphMemory(config=config)
            mem.ingest_documents(["some document"])
        assert mem.edge_count == 0

    def test_ingest_triples_missing_keys_are_skipped(self) -> None:
        config = _make_config()
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _litellm_response('[{"subject": "X", "predicate": "y"}]')
            mem = GraphMemory(config=config)
            mem.ingest_documents(["some document"])
        assert mem.edge_count == 0


# ---- Search ----

class TestGraphMemorySearch:
    def _mem_with_graph(self, top_k: int = 5) -> GraphMemory:
        """Ingest two triples: Curie-born_in-Warsaw, Warsaw-capital_of-Poland."""
        config = _make_config(top_k=top_k)
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _litellm_response(TWO_TRIPLES)
            mem = GraphMemory(config=config)
            mem.ingest_documents(["Marie Curie was born in Warsaw, the capital of Poland."])
        return mem

    def test_search_seed_node_returns_adjacent_edges(self) -> None:
        mem = self._mem_with_graph()
        results = mem.search("Marie Curie")
        assert len(results) >= 1
        assert any("Marie Curie" in r for r in results)

    def test_search_follows_two_hops(self) -> None:
        mem = self._mem_with_graph()
        # "Marie Curie" seeds node → 1 hop → Warsaw → 2 hops → Poland
        results = mem.search("Marie Curie")
        combined = " ".join(results)
        assert "Poland" in combined

    def test_search_returns_strings(self) -> None:
        mem = self._mem_with_graph()
        results = mem.search("Warsaw")
        assert all(isinstance(r, str) for r in results)

    def test_search_respects_top_k(self) -> None:
        mem = self._mem_with_graph(top_k=1)
        results = mem.search("Marie Curie")
        assert len(results) <= 1

    def test_search_empty_query_returns_empty(self) -> None:
        mem = self._mem_with_graph()
        assert mem.search("") == []

    def test_search_whitespace_query_returns_empty(self) -> None:
        mem = self._mem_with_graph()
        assert mem.search("   ") == []

    def test_search_no_matching_entity_returns_empty(self) -> None:
        mem = self._mem_with_graph()
        assert mem.search("Nikola Tesla") == []

    def test_search_on_empty_graph_returns_empty(self) -> None:
        config = _make_config()
        mem = GraphMemory(config=config)
        assert mem.search("anything") == []

    def test_search_result_format_is_readable(self) -> None:
        mem = self._mem_with_graph()
        results = mem.search("Warsaw")
        # Each result should be "Subject predicate Object" (no underscores)
        for r in results:
            assert "_" not in r  # snake_case predicates are converted to spaces


# ---- update_fact ----

class TestGraphMemoryUpdateFact:
    def test_update_fact_adds_to_graph(self) -> None:
        config = _make_config()
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _litellm_response(SINGLE_TRIPLE)
            mem = GraphMemory(config=config)
            mem.update_fact("Marie Curie was born in Warsaw.")
        assert mem.edge_count == 1

    def test_update_fact_immediately_searchable(self) -> None:
        config = _make_config()
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _litellm_response(SINGLE_TRIPLE)
            mem = GraphMemory(config=config)
            mem.update_fact("Marie Curie was born in Warsaw.")
            results = mem.search("Marie Curie")
        assert len(results) >= 1

    def test_update_fact_empty_string_ignored(self) -> None:
        config = _make_config()
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mem = GraphMemory(config=config)
            mem.update_fact("")
        assert mock_llm.call_count == 0
        assert mem.edge_count == 0

    def test_update_fact_llm_tagged_agent_reasoning(self) -> None:
        config = _make_config()
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _litellm_response(SINGLE_TRIPLE)
            mem = GraphMemory(config=config)
            mem.update_fact("Some new fact.")
        call_kwargs = mock_llm.call_args.kwargs
        assert call_kwargs["metadata"]["phase"] == "agent_reasoning"
        assert call_kwargs["metadata"]["actor"] == "graph_extract"


# ---- reset ----

class TestGraphMemoryReset:
    def test_reset_clears_nodes_and_edges(self) -> None:
        config = _make_config()
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _litellm_response(SINGLE_TRIPLE)
            mem = GraphMemory(config=config)
            mem.ingest_documents(["some document"])
        assert mem.edge_count > 0
        mem.reset()
        assert mem.node_count == 0
        assert mem.edge_count == 0

    def test_reset_makes_search_return_empty(self) -> None:
        config = _make_config()
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _litellm_response(SINGLE_TRIPLE)
            mem = GraphMemory(config=config)
            mem.ingest_documents(["some document"])
        mem.reset()
        assert mem.search("Marie Curie") == []

    def test_reset_on_empty_graph_is_no_op(self) -> None:
        config = _make_config()
        mem = GraphMemory(config=config)
        mem.reset()
        assert mem.node_count == 0

    def test_ingest_after_reset_works(self) -> None:
        config = _make_config()
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _litellm_response(SINGLE_TRIPLE)
            mem = GraphMemory(config=config)
            mem.ingest_documents(["old document"])
            mem.reset()
            mem.ingest_documents(["new document"])
        assert mem.edge_count == 1


# ---- backend name ----

class TestGraphMemoryBackendName:
    def test_backend_name_is_graph(self) -> None:
        config = _make_config()
        mem = GraphMemory(config=config)
        assert mem.get_backend_name() == "graph"


# ---- file persistence ----

class TestGraphMemoryFilePersistence:
    def test_graph_persists_across_instances(self, tmp_path: Path) -> None:
        storage = tmp_path / "graph.json"
        config1 = _make_config(memory_path=str(storage))
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _litellm_response(SINGLE_TRIPLE)
            mem1 = GraphMemory(config=config1)
            mem1.ingest_documents(["Marie Curie was born in Warsaw."])
        assert mem1.edge_count == 1

        config2 = _make_config(memory_path=str(storage))
        mem2 = GraphMemory(config=config2)
        assert mem2.edge_count == 1
        assert mem2.node_count == 2

    def test_file_created_on_ingest(self, tmp_path: Path) -> None:
        storage = tmp_path / "graph.json"
        config = _make_config(memory_path=str(storage))
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _litellm_response(SINGLE_TRIPLE)
            mem = GraphMemory(config=config)
            assert not storage.exists()
            mem.ingest_documents(["some document"])
        assert storage.exists()

    def test_reset_deletes_file(self, tmp_path: Path) -> None:
        storage = tmp_path / "graph.json"
        config = _make_config(memory_path=str(storage))
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _litellm_response(SINGLE_TRIPLE)
            mem = GraphMemory(config=config)
            mem.ingest_documents(["some document"])
        assert storage.exists()
        mem.reset()
        assert not storage.exists()

    def test_no_storage_path_no_file_created(self, tmp_path: Path) -> None:
        config = _make_config(memory_path=None)
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _litellm_response(SINGLE_TRIPLE)
            mem = GraphMemory(config=config)
            mem.ingest_documents(["some document"])
        assert list(tmp_path.iterdir()) == []

    def test_update_fact_persisted(self, tmp_path: Path) -> None:
        storage = tmp_path / "graph.json"
        config1 = _make_config(memory_path=str(storage))
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _litellm_response(SINGLE_TRIPLE)
            mem1 = GraphMemory(config=config1)
            mem1.update_fact("Marie Curie was born in Warsaw.")

        config2 = _make_config(memory_path=str(storage))
        mem2 = GraphMemory(config=config2)
        assert mem2.edge_count == 1
        assert len(mem2.search("Marie Curie")) >= 1
