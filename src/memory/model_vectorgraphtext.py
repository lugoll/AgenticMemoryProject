from __future__ import annotations

import logging
import time

from src.config.cfg import Config
from src.telemetry import record_retrieval_event
from .base import BaseMemory
from .reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class VectorGraphTextMemory(BaseMemory):
    """
    Chunk-vector-seeded graph retrieval that returns source passages.

    Strict superset of VectorRerankMemory's candidate pool: hop-0 candidates
    are the same chunks cosine retrieval finds, the graph traversal then adds
    multi-hop *bridge chunks* that mention entities 1..max_hops away — the
    passages pure vector similarity cannot see. Pipeline:

      1. Embed the query with the shared embedder and match it against the
         ``chunk_vector`` index (identical entry signal to the vector variant),
         pulling ``retrieval.hop0_fetch_k`` chunks (falls back to top_k); keep
         both the hop-0 chunk texts and their MENTIONS entities as seeds.
      2. Flat hop-by-hop BFS (BaseMemory._expand_triples) over the entity
         subgraph → visited entity set (max_frontier / max_candidates caps).
      3. BaseMemory._collect_chunks: chunks mentioning the visited entities,
         mention-overlap-ranked, capped at ``retrieval.rerank_fetch_k``.
      4. Hop-0 + collected chunks, deduplicated, cross-encoder reranked to
         ``retrieval.top_k`` (the shared final-context knob).

    Replaced the LlamaIndex VectorContextRetriever include_text path
    (entity-name-embedding seeds, duplicate combined strings per triplet);
    its eval results remain recorded in evaluations/.
    No LLM call at retrieval time.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._top_k: int = config.retrieval.top_k
        # Hop-0 chunk-vector fetch size, decoupled from the final-context top_k.
        # None → fall back to top_k (baseline behaviour).
        self._hop0_k: int = config.retrieval.hop0_fetch_k or config.retrieval.top_k
        self._cutoff: float = config.retrieval.similarity_cutoff
        self._reranker = CrossEncoderReranker(config)

    def search(self, query: str) -> list[str]:
        if not query.strip():
            return []

        t0 = time.perf_counter()
        query_embedding = self.embed_model.get_text_embedding(query)

        try:
            rows = self._cypher(
                "CALL db.index.vector.queryNodes($index, $top_k, $embedding) "
                "YIELD node, score "
                "WHERE score >= $cutoff "
                "OPTIONAL MATCH (node)-[:MENTIONS]->(e:__Entity__) "
                "RETURN node.text AS text, collect(DISTINCT e.name) AS entities",
                index=self._vector_index,
                top_k=self._hop0_k,
                embedding=query_embedding,
                cutoff=self._cutoff,
            )
        except Exception as exc:
            # Index missing (ingest not run yet) — behave like an empty store.
            logger.warning("VectorGraphTextMemory: chunk seeding failed: %s", exc)
            return []

        hop0_chunks = [r["text"] for r in rows if r["text"] and r["text"].strip()]
        seed_nodes = sorted({e for r in rows for e in r["entities"] if e})
        t_seed = time.perf_counter()

        if not hop0_chunks and not seed_nodes:
            logger.debug("VectorGraphTextMemory: query=%r → no seed chunks", query)
            return []

        _triples, visited, stats = self._expand_triples(seed_nodes)
        t_bfs = time.perf_counter()

        bridge_chunks = self._collect_chunks(visited)
        t_collect = time.perf_counter()

        # Hop-0 first (the vector signal), then bridge chunks; order-preserving
        # dedup so identical passages cannot occupy several rerank slots.
        candidates = list(dict.fromkeys(hop0_chunks + bridge_chunks))
        reranked = self._reranker.rerank(query, candidates)
        t_rerank = time.perf_counter()

        record_retrieval_event(
            "chunk_seed", self.get_backend_name(),
            duration_ms=(t_seed - t0) * 1000,
            returned=len(hop0_chunks), seeds=len(seed_nodes),
        )
        record_retrieval_event(
            "graph_bfs", self.get_backend_name(),
            duration_ms=(t_bfs - t_seed) * 1000,
            seeds=len(seed_nodes), visited=len(visited), **stats,
        )
        record_retrieval_event(
            "chunk_collect", self.get_backend_name(),
            duration_ms=(t_collect - t_bfs) * 1000,
            entities=len(visited), returned=len(bridge_chunks),
        )
        record_retrieval_event(
            "rerank", self.get_backend_name(),
            duration_ms=(t_rerank - t_collect) * 1000,
            candidates=len(candidates), returned=len(reranked),
        )
        logger.debug(
            "VectorGraphTextMemory: query=%r → %d hop-0 chunks / %d seeds → "
            "%d visited → %d bridge chunks → %d candidates → rerank top %d",
            query, len(hop0_chunks), len(seed_nodes), len(visited),
            len(bridge_chunks), len(candidates), len(reranked),
        )
        return reranked

    def get_backend_name(self) -> str:
        return "vectorgraphtext"

    @property
    def store_size(self) -> int:
        """Number of chunks currently stored (shared-store view)."""
        return self.chunk_count
