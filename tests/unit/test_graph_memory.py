from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config.cfg import (
    Config, LLMCfg, LLMsCfg, EmbeddingCfg,
    RetrievalCfg, IngestionCfg, GraphCfg, StoresCfg, TelemetryCfg,
)
from src.memory.base import BaseMemory
from src.memory.model_graph import GraphMemory


def _make_config(graph_path: Path, top_k: int = 5) -> Config:
    _llm = LLMCfg(model="ollama/test", base_url="http://localhost:11434",
                  temperature=0.0, max_tokens=512)
    return Config(
        llm=LLMsCfg(agent=_llm, ingest=_llm, judge=_llm),
        embedding=EmbeddingCfg(model="BAAI/bge-base-en-v1.5", batch_size=8,
                               chroma_host="http://localhost:8000"),
        retrieval=RetrievalCfg(top_k=top_k, similarity_cutoff=0.5),
        ingestion=IngestionCfg(chunk_size=300, chunk_overlap=50),
        graph=GraphCfg(max_hops=3),
        stores=StoresCfg(bm25="", graph=str(graph_path), vector=""),
        telemetry=TelemetryCfg(output_dir="evaluations/"),
    )


def _llm_resp(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices[0].message.content = content
    return resp


# New response format: {"triples": [...]}
SINGLE_TRIPLE = '{"triples": [{"subject": "Marie Curie", "predicate": "born_in", "object": "Warsaw"}]}'
TWO_TRIPLES = (
    '{"triples": ['
    '{"subject": "Marie Curie", "predicate": "born_in", "object": "Warsaw"},'
    '{"subject": "Warsaw", "predicate": "capital_of", "object": "Poland"}]}'
)
EMPTY_TRIPLES = '{"triples": []}'


class TestBaseMemoryContract:
    def test_graph_memory_is_subclass_of_base(self) -> None:
        assert issubclass(GraphMemory, BaseMemory)


class TestGraphMemoryIngest:
    def test_ingest_calls_llm_per_document(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path / "g.json")
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _llm_resp(SINGLE_TRIPLE)
            mem = GraphMemory(config=cfg)
            mem.ingest_documents(["doc one", "doc two"])
        assert mock_llm.call_count == 2

    def test_ingest_populates_nodes_and_edges(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path / "g.json")
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _llm_resp(SINGLE_TRIPLE)
            mem = GraphMemory(config=cfg)
            mem.ingest_documents(["Marie Curie was born in Warsaw."])
        assert mem.node_count == 2
        assert mem.edge_count == 1

    def test_ingest_skips_empty_documents(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path / "g.json")
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _llm_resp(EMPTY_TRIPLES)
            mem = GraphMemory(config=cfg)
            mem.ingest_documents(["", "   ", "\n"])
        assert mock_llm.call_count == 0

    def test_ingest_deduplicates_identical_triples(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path / "g.json")
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _llm_resp(SINGLE_TRIPLE)
            mem = GraphMemory(config=cfg)
            mem.ingest_documents(["doc one", "doc two"])
        assert mem.edge_count == 1

    def test_ingest_llm_tagged_correctly(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path / "g.json")
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _llm_resp(SINGLE_TRIPLE)
            mem = GraphMemory(config=cfg)
            mem.ingest_documents(["some document"])
        call_kwargs = mock_llm.call_args.kwargs
        assert call_kwargs["metadata"]["phase"] == "ingest"
        assert call_kwargs["metadata"]["actor"] == "graph_extract"

    def test_ingest_invalid_json_from_llm_is_skipped(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path / "g.json")
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _llm_resp("not valid json at all !!!")
            mem = GraphMemory(config=cfg)
            mem.ingest_documents(["some document"])
        assert mem.edge_count == 0

    def test_ingest_empty_triples_array(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path / "g.json")
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _llm_resp(EMPTY_TRIPLES)
            mem = GraphMemory(config=cfg)
            mem.ingest_documents(["some document"])
        assert mem.edge_count == 0

    def test_ingest_triples_missing_object_key_are_skipped(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path / "g.json")
        incomplete = '{"triples": [{"subject": "X", "predicate": "y"}]}'
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _llm_resp(incomplete)
            mem = GraphMemory(config=cfg)
            mem.ingest_documents(["some document"])
        assert mem.edge_count == 0


class TestGraphMemorySearch:
    def _mem_with_graph(self, tmp_path: Path, top_k: int = 5) -> GraphMemory:
        cfg = _make_config(tmp_path / "g.json", top_k=top_k)
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _llm_resp(TWO_TRIPLES)
            mem = GraphMemory(config=cfg)
            mem.ingest_documents(["Marie Curie was born in Warsaw, the capital of Poland."])
        return mem

    def test_search_seed_node_returns_adjacent_edges(self, tmp_path: Path) -> None:
        mem = self._mem_with_graph(tmp_path)
        results = mem.search("Marie Curie")
        assert len(results) >= 1
        assert any("Marie Curie" in r for r in results)

    def test_search_follows_two_hops(self, tmp_path: Path) -> None:
        mem = self._mem_with_graph(tmp_path)
        results = mem.search("Marie Curie")
        combined = " ".join(results)
        assert "Poland" in combined

    def test_search_returns_strings(self, tmp_path: Path) -> None:
        mem = self._mem_with_graph(tmp_path)
        assert all(isinstance(r, str) for r in mem.search("Warsaw"))

    def test_search_respects_top_k(self, tmp_path: Path) -> None:
        mem = self._mem_with_graph(tmp_path, top_k=1)
        assert len(mem.search("Marie Curie")) <= 1

    def test_search_empty_query_returns_empty(self, tmp_path: Path) -> None:
        mem = self._mem_with_graph(tmp_path)
        assert mem.search("") == []

    def test_search_whitespace_query_returns_empty(self, tmp_path: Path) -> None:
        mem = self._mem_with_graph(tmp_path)
        assert mem.search("   ") == []

    def test_search_no_matching_entity_returns_empty(self, tmp_path: Path) -> None:
        mem = self._mem_with_graph(tmp_path)
        assert mem.search("Nikola Tesla") == []

    def test_search_on_empty_graph_returns_empty(self, tmp_path: Path) -> None:
        mem = GraphMemory(config=_make_config(tmp_path / "g.json"))
        assert mem.search("anything") == []

    def test_search_result_format_no_underscores(self, tmp_path: Path) -> None:
        mem = self._mem_with_graph(tmp_path)
        for r in mem.search("Warsaw"):
            assert "_" not in r


class TestGraphMemoryUpdateFact:
    def test_update_fact_adds_to_graph(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path / "g.json")
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _llm_resp(SINGLE_TRIPLE)
            mem = GraphMemory(config=cfg)
            mem.update_fact("Marie Curie was born in Warsaw.")
        assert mem.edge_count == 1

    def test_update_fact_immediately_searchable(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path / "g.json")
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _llm_resp(SINGLE_TRIPLE)
            mem = GraphMemory(config=cfg)
            mem.update_fact("Marie Curie was born in Warsaw.")
        assert len(mem.search("Marie Curie")) >= 1

    def test_update_fact_empty_string_ignored(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path / "g.json")
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mem = GraphMemory(config=cfg)
            mem.update_fact("")
        assert mock_llm.call_count == 0
        assert mem.edge_count == 0

    def test_update_fact_tagged_agent_reasoning(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path / "g.json")
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _llm_resp(SINGLE_TRIPLE)
            mem = GraphMemory(config=cfg)
            mem.update_fact("Some new fact.")
        call_kwargs = mock_llm.call_args.kwargs
        assert call_kwargs["metadata"]["phase"] == "agent_reasoning"
        assert call_kwargs["metadata"]["actor"] == "graph_extract"


class TestGraphMemoryReset:
    def test_reset_clears_graph(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path / "g.json")
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _llm_resp(SINGLE_TRIPLE)
            mem = GraphMemory(config=cfg)
            mem.ingest_documents(["some document"])
        assert mem.edge_count > 0
        mem.reset()
        assert mem.node_count == 0
        assert mem.edge_count == 0

    def test_reset_makes_search_empty(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path / "g.json")
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _llm_resp(SINGLE_TRIPLE)
            mem = GraphMemory(config=cfg)
            mem.ingest_documents(["some document"])
        mem.reset()
        assert mem.search("Marie Curie") == []

    def test_ingest_after_reset_works(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path / "g.json")
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _llm_resp(SINGLE_TRIPLE)
            mem = GraphMemory(config=cfg)
            mem.ingest_documents(["old document"])
            mem.reset()
            mem.ingest_documents(["new document"])
        assert mem.edge_count == 1


class TestGraphMemoryFilePersistence:
    def test_graph_persists_across_instances(self, tmp_path: Path) -> None:
        storage = tmp_path / "graph.json"
        cfg1 = _make_config(storage)
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _llm_resp(SINGLE_TRIPLE)
            mem1 = GraphMemory(config=cfg1)
            mem1.ingest_documents(["Marie Curie was born in Warsaw."])
        assert mem1.edge_count == 1

        mem2 = GraphMemory(config=_make_config(storage))
        assert mem2.edge_count == 1
        assert mem2.node_count == 2

    def test_file_created_on_ingest(self, tmp_path: Path) -> None:
        storage = tmp_path / "graph.json"
        cfg = _make_config(storage)
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _llm_resp(SINGLE_TRIPLE)
            mem = GraphMemory(config=cfg)
            assert not storage.exists()
            mem.ingest_documents(["some document"])
        assert storage.exists()

    def test_reset_deletes_file(self, tmp_path: Path) -> None:
        storage = tmp_path / "graph.json"
        cfg = _make_config(storage)
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _llm_resp(SINGLE_TRIPLE)
            mem = GraphMemory(config=cfg)
            mem.ingest_documents(["some document"])
        assert storage.exists()
        mem.reset()
        assert not storage.exists()

    def test_update_fact_persisted(self, tmp_path: Path) -> None:
        storage = tmp_path / "graph.json"
        with patch("src.memory.model_graph.litellm.completion") as mock_llm:
            mock_llm.return_value = _llm_resp(SINGLE_TRIPLE)
            mem1 = GraphMemory(config=_make_config(storage))
            mem1.update_fact("Marie Curie was born in Warsaw.")

        mem2 = GraphMemory(config=_make_config(storage))
        assert mem2.edge_count == 1
        assert len(mem2.search("Marie Curie")) >= 1


class TestGraphMemoryBackendName:
    def test_backend_name_is_graph(self, tmp_path: Path) -> None:
        mem = GraphMemory(config=_make_config(tmp_path / "g.json"))
        assert mem.get_backend_name() == "graph"
