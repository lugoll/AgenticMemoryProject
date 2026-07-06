from __future__ import annotations

import logging
import re
import time

from rank_bm25 import BM25Okapi

from src.config.cfg import Config
from src.telemetry import record_retrieval_event
from .base import BaseMemory
from .reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class GraphMemory(BaseMemory):
    """
    Knowledge-graph retrieval view over the shared Neo4j store (hybrid port).

    Design decisions (see EXPERIMENT_PLAN.md § Graph-Variante):
      1. BM25 entity linking — maps natural-language query tokens to graph
         nodes without an LLM call, preserving the zero-cost retrieval
         property that distinguishes GraphRAG from Vector RAG. The rank_bm25
         index is kept in memory (built over entity names fetched from Neo4j)
         so tokenisation and scoring stay identical to the pre-migration
         NetworkX implementation.
      2. Global hop-by-hop BFS (BaseMemory._expand_triples) followed by a flat
         single-triple cross-encoder rerank — the frontier starts as all seeds
         and each hop is one batched Cypher query. The eval settled on this over
         chain-ranking, which regressed the graph variant's precise BM25 seeds.
         No LLM call at retrieval time.

    The graph itself is written by the unified ingest (BaseMemory) via the
    LlamaIndex extractor with the shared predicate whitelist.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._max_hops: int = config.graph.max_hops
        # Graph uses its own top_k (graph.top_k) because triples are ~8 words
        # vs. text passages (~50 words). Needs more results to cover 2-hop paths.
        self._top_k: int = config.graph.top_k
        self._bm25: BM25Okapi | None = None
        self._node_list: list[str] = []
        self._reranker = CrossEncoderReranker(config)

    # ── BM25 entity-linking index ─────────────────────────────────────────────

    # Common English stopwords that produce false-positive BM25 matches.
    # Example without this list: "Who designed the hotel..." → "who" matches
    # band node "The Who" → BFS returns music facts instead of hotel facts.
    _STOPWORDS = frozenset({
        "who", "what", "when", "where", "which", "how", "why",
        "the", "a", "an", "and", "or", "but", "in", "on", "at",
        "to", "of", "for", "by", "with", "from", "is", "was",
        "are", "were", "be", "been", "has", "had", "have", "do",
        "did", "does", "not", "it", "its", "that", "this", "as",
    })

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        tokens = re.sub(r"[^a-z0-9]", " ", text.lower()).split()
        # Keep tokens ≥3 chars that are not stopwords.
        # ≥3 (not >3) preserves short but meaningful tokens like "vh1", "bbc", "nba".
        return [t for t in tokens if len(t) >= 3 and t not in cls._STOPWORDS]

    def _build_node_index(self) -> None:
        rows = self._cypher(
            "MATCH (n:__Entity__) WHERE n.name IS NOT NULL "
            "RETURN n.name AS name ORDER BY name"
        )
        self._node_list = [r["name"] for r in rows]
        corpus = [self._tokenize(n) for n in self._node_list]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def search(self, query: str) -> list[str]:
        if not query.strip():
            return []

        if self._bm25 is None:
            self._build_node_index()

        if not self._node_list:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        t0 = time.perf_counter()
        scores = self._bm25.get_scores(query_tokens)
        top_indices = scores.argsort()[::-1][:10]
        seed_nodes = [self._node_list[i] for i in top_indices if scores[i] > 0]

        if not seed_nodes:
            return []

        results, stats = self._expand_triples(seed_nodes)
        t_expand = time.perf_counter()

        reranked = self._reranker.rerank(query, results)
        t_rerank = time.perf_counter()

        record_retrieval_event(
            "graph_bfs", self.get_backend_name(),
            duration_ms=(t_expand - t0) * 1000,
            seeds=len(seed_nodes), **stats,
        )
        record_retrieval_event(
            "rerank", self.get_backend_name(),
            duration_ms=(t_rerank - t_expand) * 1000,
            candidates=len(results), returned=len(reranked),
        )
        logger.debug(
            "GraphMemory: query=%r → %d seeds → %d candidates → rerank top %d",
            query, len(seed_nodes), len(results), len(reranked),
        )
        return reranked

    def get_backend_name(self) -> str:
        return "graph"
