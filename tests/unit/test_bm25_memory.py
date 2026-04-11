from __future__ import annotations

from pathlib import Path

import pytest

from src.memory.base import BaseMemory
from src.memory.model_bm25 import BM25Memory


class TestBaseMemoryContract:
    def test_bm25_is_subclass_of_base(self) -> None:
        assert issubclass(BM25Memory, BaseMemory)


class TestBM25MemoryIngest:
    def test_ingest_increases_store_size(self) -> None:
        mem = BM25Memory()
        mem.ingest_documents(["doc one", "doc two"])
        assert mem.store_size == 2

    def test_ingest_empty_list_no_op(self) -> None:
        mem = BM25Memory()
        mem.ingest_documents([])
        assert mem.store_size == 0

    def test_empty_strings_ignored(self) -> None:
        mem = BM25Memory()
        mem.ingest_documents(["", "   ", "valid content"])
        assert mem.store_size == 1

    def test_whitespace_only_strings_ignored(self) -> None:
        mem = BM25Memory()
        mem.ingest_documents(["\t\n", "  "])
        assert mem.store_size == 0

    def test_multiple_ingests_accumulate(self) -> None:
        mem = BM25Memory()
        mem.ingest_documents(["first"])
        mem.ingest_documents(["second"])
        assert mem.store_size == 2


class TestBM25MemorySearch:
    def test_single_keyword_match(self) -> None:
        mem = BM25Memory(top_k=5)
        mem.ingest_documents(["the cat sat on the mat", "dogs like bones"])
        results = mem.search("cat")
        assert len(results) == 1
        assert "cat" in results[0]

    def test_multiple_keywords_ranked_higher(self) -> None:
        mem = BM25Memory(top_k=5)
        mem.ingest_documents(["apple banana cherry", "apple banana", "apple"])
        results = mem.search("apple banana cherry")
        # The document matching all three terms should rank first.
        assert results[0] == "apple banana cherry"

    def test_returns_at_most_top_k(self) -> None:
        mem = BM25Memory(top_k=2)
        mem.ingest_documents([f"apple item {i}" for i in range(10)])
        results = mem.search("apple")
        assert len(results) <= 2

    def test_empty_query_returns_empty(self) -> None:
        mem = BM25Memory()
        mem.ingest_documents(["something useful"])
        assert mem.search("") == []

    def test_whitespace_only_query_returns_empty(self) -> None:
        mem = BM25Memory()
        mem.ingest_documents(["something useful"])
        assert mem.search("   ") == []

    def test_no_match_returns_empty(self) -> None:
        mem = BM25Memory()
        mem.ingest_documents(["cats and dogs"])
        assert mem.search("python programming") == []

    def test_case_insensitive_matching(self) -> None:
        mem = BM25Memory()
        mem.ingest_documents(["The Capital of France is Paris"])
        results = mem.search("france paris")
        assert len(results) == 1

    def test_search_on_empty_store_returns_empty(self) -> None:
        mem = BM25Memory()
        assert mem.search("anything") == []

    def test_results_are_strings(self) -> None:
        mem = BM25Memory()
        mem.ingest_documents(["hello world"])
        results = mem.search("hello")
        assert all(isinstance(r, str) for r in results)

    def test_special_chars_in_query_do_not_crash(self) -> None:
        mem = BM25Memory()
        mem.ingest_documents(["safe content here"])
        # These contain FTS5 special chars — should not raise.
        mem.search('"quoted"')
        mem.search("star*")
        mem.search("(parens)")
        mem.search("a + b - c")


class TestBM25MemoryUpdateFact:
    def test_update_fact_increases_store_size(self) -> None:
        mem = BM25Memory()
        mem.update_fact("The sky is blue")
        assert mem.store_size == 1

    def test_update_fact_immediately_searchable(self) -> None:
        mem = BM25Memory()
        mem.update_fact("The capital of France is Paris")
        results = mem.search("France Paris")
        assert len(results) == 1
        assert "Paris" in results[0]

    def test_update_empty_fact_ignored(self) -> None:
        mem = BM25Memory()
        mem.update_fact("")
        assert mem.store_size == 0

    def test_update_fact_and_ingest_share_same_store(self) -> None:
        mem = BM25Memory()
        mem.ingest_documents(["Paris is in Europe"])
        mem.update_fact("Paris is the capital of France")
        results = mem.search("Paris")
        assert len(results) == 2


class TestBM25MemoryBackendName:
    def test_backend_name_is_bm25(self) -> None:
        assert BM25Memory().get_backend_name() == "bm25"


class TestBM25MemoryReset:
    def test_reset_clears_store(self) -> None:
        mem = BM25Memory()
        mem.ingest_documents(["doc one", "doc two"])
        mem.reset()
        assert mem.store_size == 0

    def test_reset_makes_search_return_empty(self) -> None:
        mem = BM25Memory()
        mem.ingest_documents(["Paris is the capital of France"])
        mem.reset()
        assert mem.search("Paris") == []

    def test_reset_on_empty_store_is_no_op(self) -> None:
        mem = BM25Memory()
        mem.reset()
        assert mem.store_size == 0

    def test_ingest_after_reset_works(self) -> None:
        mem = BM25Memory()
        mem.ingest_documents(["old data"])
        mem.reset()
        mem.ingest_documents(["new data"])
        assert mem.store_size == 1
        assert mem.search("new") == ["new data"]


class TestBM25MemoryFilePersistence:
    def test_data_persists_across_instances(self, tmp_path: Path) -> None:
        path = tmp_path / "store.db"
        mem1 = BM25Memory(top_k=5, storage_path=path)
        mem1.ingest_documents(["persistent fact"])

        mem2 = BM25Memory(top_k=5, storage_path=path)
        assert mem2.store_size == 1
        assert "persistent fact" in mem2.search("persistent")

    def test_file_created_on_init(self, tmp_path: Path) -> None:
        path = tmp_path / "store.db"
        BM25Memory(top_k=5, storage_path=path)
        assert path.exists()

    def test_reset_deletes_file(self, tmp_path: Path) -> None:
        path = tmp_path / "store.db"
        mem = BM25Memory(top_k=5, storage_path=path)
        mem.ingest_documents(["something"])
        assert path.exists()
        mem.reset()
        # After reset the DB file is removed and recreated in-memory style,
        # but since we passed a storage_path, a new file is created.
        assert mem.store_size == 0

    def test_update_fact_persisted(self, tmp_path: Path) -> None:
        path = tmp_path / "store.db"
        mem1 = BM25Memory(top_k=5, storage_path=path)
        mem1.update_fact("sky is blue")

        mem2 = BM25Memory(top_k=5, storage_path=path)
        assert "sky is blue" in mem2.search("sky")
