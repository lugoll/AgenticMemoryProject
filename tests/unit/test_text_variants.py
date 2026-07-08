"""Unit tests for the chunk-text graph variants (graphtext / vectorgraphtext).

Everything runs against fakes — no Neo4j, no embedder, no cross-encoder.
Instances are built with ``object.__new__`` to skip BaseMemory.__init__
(which waits for a live Bolt connection) in the same spirit as the
traversal test doubles.
"""
from __future__ import annotations

from src.config.cfg import load_config
from src.memory.model_graphtext import GraphTextMemory
from src.memory.model_vectorgraphtext import VectorGraphTextMemory


CFG = load_config()


class _FakeReranker:
    """Order-preserving truncation to retrieval.top_k (like the real one with
    a constant-score model); records the candidate list it was given."""

    def __init__(self) -> None:
        self.seen: list[str] | None = None

    def rerank(self, query: str, candidates: list[str]) -> list[str]:
        self.seen = list(candidates)
        return candidates[: CFG.retrieval.top_k]


class _FakeEmbed:
    def get_text_embedding(self, text: str) -> list[float]:
        return [0.0, 1.0]


def _fake_cypher(seed_rows, edge_rows, chunk_rows, calls):
    """Dispatch on the Cypher text: vector seeding / BFS hop / chunk collection."""

    def cypher(query: str, /, **params):
        calls.append((query, params))
        if "db.index.vector.queryNodes" in query:
            return seed_rows
        if "MENTIONS" in query and "overlap" in query:
            return chunk_rows
        if "MATCH (n:__Entity__)-[r]-(m:__Entity__)" in query:
            # One populated hop, then empty frontiers end the BFS.
            rows, edge_rows[:] = edge_rows[:], []
            return rows
        raise AssertionError(f"unexpected cypher: {query}")

    return cypher


def _make_vgt(seed_rows, edge_rows, chunk_rows, calls) -> VectorGraphTextMemory:
    m = object.__new__(VectorGraphTextMemory)
    m._config = CFG
    m._vector_index = CFG.stores.chunk_vector_index
    m._top_k = CFG.retrieval.top_k
    # Hop-0 fetch decoupled from top_k (falls back to top_k when unset).
    m._hop0_k = CFG.retrieval.hop0_fetch_k or CFG.retrieval.top_k
    m._cutoff = CFG.retrieval.similarity_cutoff
    m._embed_model = _FakeEmbed()
    m._reranker = _FakeReranker()
    m._cypher = _fake_cypher(seed_rows, edge_rows, chunk_rows, calls)
    return m


class TestVectorGraphText:
    def test_merges_hop0_and_bridge_chunks_deduplicated(self) -> None:
        """Hop-0 chunks come first, bridge chunks follow, duplicates collapse."""
        seed_rows = [
            {"text": "chunk about Tiffany", "entities": ["Tiffany Pollard"]},
            {"text": "chunk about VH1", "entities": ["VH1"]},
        ]
        edge_rows = [
            {"s": "Tiffany Pollard", "p": "appeared_in", "o": "VH1", "neighbor": "VH1"},
        ]
        chunk_rows = [
            {"text": "chunk about Tiffany"},   # duplicate of a hop-0 chunk
            {"text": "bridge chunk"},
        ]
        calls: list = []
        m = _make_vgt(seed_rows, edge_rows, chunk_rows, calls)

        result = m.search("who appeared on VH1?")

        assert m._reranker.seen == [
            "chunk about Tiffany", "chunk about VH1", "bridge chunk",
        ]
        assert result == m._reranker.seen[: CFG.retrieval.top_k]

    def test_collect_chunks_uses_rerank_fetch_k_and_visited_entities(self) -> None:
        seed_rows = [{"text": "c1", "entities": ["A"]}]
        edge_rows = [{"s": "A", "p": "rel", "o": "B", "neighbor": "B"}]
        calls: list = []
        m = _make_vgt(seed_rows, edge_rows, [{"text": "c2"}], calls)

        m.search("q")

        collect_calls = [(q, p) for q, p in calls if "overlap" in q]
        assert len(collect_calls) == 1
        _, params = collect_calls[0]
        assert params["fetch_k"] == CFG.retrieval.rerank_fetch_k
        # Visited = seeds + BFS neighbours.
        assert params["entities"] == ["A", "B"]

    def test_no_seeds_returns_empty(self) -> None:
        calls: list = []
        m = _make_vgt([], [], [], calls)
        assert m.search("q") == []
        # Neither BFS nor chunk collection ran.
        assert all("queryNodes" in q for q, _ in calls)

    def test_final_size_is_retrieval_top_k(self) -> None:
        seed_rows = [
            {"text": f"chunk {i}", "entities": []} for i in range(10)
        ]
        m = _make_vgt(seed_rows, [], [], [])
        assert len(m.search("q")) == CFG.retrieval.top_k


class TestGraphText:
    def test_bm25_seeded_chunks_reranked(self) -> None:
        edge_rows = [{"s": "A", "p": "rel", "o": "B", "neighbor": "B"}]
        chunk_rows = [{"text": "passage 1"}, {"text": "passage 2"}]
        calls: list = []

        m = object.__new__(GraphTextMemory)
        m._config = CFG
        m._reranker = _FakeReranker()
        m._cypher = _fake_cypher([], edge_rows, chunk_rows, calls)
        m._bm25_seeds = lambda query: ["A"]

        result = m.search("q")

        assert result == ["passage 1", "passage 2"]
        assert m.get_backend_name() == "graphtext"

    def test_no_bm25_match_returns_empty(self) -> None:
        m = object.__new__(GraphTextMemory)
        m._config = CFG
        m._reranker = _FakeReranker()
        m._bm25_seeds = lambda query: []
        assert m.search("q") == []
