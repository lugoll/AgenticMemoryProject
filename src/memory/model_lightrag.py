from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import Any

import litellm
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

from src.config.settings import ExperimentConfig, LightRAGConfig

from .base import BaseMemory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level mutable dict for telemetry tag threading.
#
# ContextVar would be the natural choice here, but LightRAG's
# priority_limit_async_func_call decorator runs llm_func inside long-lived
# worker tasks created via asyncio.create_task(). create_task() snapshots the
# context at worker-creation time, so ContextVar mutations made later in
# _ingest_with_context / _query_with_context never reach the workers.
#
# A plain dict works safely instead: _run_sync() blocks until each operation
# fully completes before the next one starts, so there is no concurrent
# interleaving of phases across different ingest/query/update calls.
# ---------------------------------------------------------------------------
_TELEMETRY: dict[str, str] = {
    "phase": "ingest",
    "actor": "graph_extract",
    "run_id": "unknown",
}


# ---------------------------------------------------------------------------
# LiteLLM bridge functions
# ---------------------------------------------------------------------------

def _make_llm_func(config: ExperimentConfig):
    """Return an async LLM function that routes through LiteLLM with telemetry tags."""

    async def llm_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict] = [],
        keyword_extraction: bool = False,
        **kwargs: Any,
    ) -> str:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        response = await litellm.acompletion(
            model=config.llm.model,
            messages=messages,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            metadata={
                "phase": _TELEMETRY["phase"],
                "actor": _TELEMETRY["actor"],
                "variant_name": config.variant_name,
                "run_id": _TELEMETRY["run_id"],
            },
        )
        return response.choices[0].message.content or "" # type: ignore

    return llm_func


def _make_embedding_func(config: ExperimentConfig, lightrag_cfg: LightRAGConfig) -> EmbeddingFunc:
    """Return an EmbeddingFunc that routes embeddings through LiteLLM with telemetry tags."""

    async def embed_func(texts: list[str]) -> np.ndarray:
        response = await litellm.aembedding(
            model=config.embedding.model,
            input=texts,
            metadata={
                "phase": _TELEMETRY["phase"],
                "actor": "vector_embed",
                "variant_name": config.variant_name,
                "run_id": _TELEMETRY["run_id"],
            },
        )
        return np.array([item["embedding"] for item in response.data])

    return EmbeddingFunc(
        func=embed_func,
        embedding_dim=lightrag_cfg.embedding_dim,
        max_token_size=lightrag_cfg.max_token_size,
    )


# ---------------------------------------------------------------------------
# Coroutine wrappers that set the telemetry dict before delegating to LightRAG
# ---------------------------------------------------------------------------

async def _ingest_with_context(rag: LightRAG, documents: list[str], variant_name: str) -> None:
    _TELEMETRY.update({"phase": "ingest", "actor": "graph_extract", "run_id": "ingest"})
    await rag.ainsert(documents)


async def _query_with_context(
    rag: LightRAG, query: str, param: QueryParam, run_id: str
) -> str:
    _TELEMETRY.update({"phase": "retrieval_overhead", "actor": "graph_cypher_gen", "run_id": run_id})
    return await rag.aquery(query, param=param) # type: ignore


async def _update_with_context(rag: LightRAG, fact: str) -> None:
    _TELEMETRY.update({"phase": "agent_reasoning", "actor": "graph_extract", "run_id": "update_fact"})
    await rag.ainsert(fact)


# ---------------------------------------------------------------------------
# LightRAGMemory
# ---------------------------------------------------------------------------

class LightRAGMemory(BaseMemory):
    """
    LightRAG-backed memory using the lightrag-hku package.

    LightRAG builds a hybrid knowledge graph + vector index from documents
    during ingestion, then uses multi-hop graph traversal and vector search
    at query time to return raw context (entities, relations, text chunks).
    Final answer synthesis is delegated to the LangGraph agent's reason node.

    Async/sync bridge:
        LightRAG is async-first; BaseMemory is synchronous. A dedicated daemon
        thread runs a persistent asyncio event loop. All coroutines are
        dispatched via asyncio.run_coroutine_threadsafe(), which blocks the
        calling thread until completion and is safe to call from any context,
        including LangGraph's own sync-wrapped event loop.

    Telemetry:
        Phase/actor/run_id tags are stored in the module-level _TELEMETRY dict.
        Each operation sets the dict in its coroutine wrapper before calling into
        LightRAG; the llm_func closure reads it when LiteLLM is called. This is
        safe because _run_sync() serialises operations — no two phases are
        concurrent across separate ingest/query/update calls.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self._lightrag_cfg: LightRAGConfig = config.lightrag or LightRAGConfig()
        self._working_dir = Path(config.memory_path) if config.memory_path else Path("data/processed/lightrag_default")

        # Dedicated event loop in a daemon thread — isolated from any outer loop
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="lightrag-async"
        )
        self._thread.start()

        self._rag = self._build_rag()
        self._run_sync(self._rag.initialize_storages())
        logger.debug("LightRAGMemory: initialised, working_dir=%s", self._working_dir)

    def _build_rag(self) -> LightRAG:
        self._working_dir.mkdir(parents=True, exist_ok=True)
        return LightRAG(
            working_dir=str(self._working_dir),
            llm_model_func=_make_llm_func(self._config),
            embedding_func=_make_embedding_func(self._config, self._lightrag_cfg),
        )

    def _run_sync(self, coro) -> Any:
        """Submit a coroutine to the background loop and block until done."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    # ---- BaseMemory interface ----

    def ingest_documents(self, documents: list[str]) -> None:
        """
        Insert documents into LightRAG's knowledge graph + vector index.

        LLM calls tagged: phase="ingest", actor="graph_extract".
        Embedding calls tagged: phase="ingest", actor="vector_embed".
        """
        non_empty = [d for d in documents if d.strip()]
        if not non_empty:
            return
        self._run_sync(_ingest_with_context(self._rag, non_empty, self._config.variant_name))
        logger.debug("LightRAGMemory: ingested %d documents", len(non_empty))

    def search(self, query: str) -> list[str]:
        """
        Query LightRAG and return the raw assembled context as a single-element list.

        only_need_context=True skips LightRAG's internal LLM synthesis step.
        LightRAG returns the structured KG context (entities, relations, chunks)
        directly; the LangGraph reason node does the final answer synthesis.

        No LLM calls during retrieval.
        Embedding calls tagged: phase="retrieval_overhead", actor="vector_embed".
        """
        if not query.strip():
            return []

        param = QueryParam(
            mode=self._lightrag_cfg.query_mode,
            top_k=self._config.retrieval.top_k,
            only_need_context=True,
        )
        result: str = self._run_sync(
            _query_with_context(self._rag, query, param, run_id="search")
        )
        if not result or not result.strip():
            return []
        logger.debug(
            "LightRAGMemory: query=%r → %d-char response", query, len(result)
        )
        return [result]

    def update_fact(self, fact: str) -> None:
        """
        Insert a new fact into the live knowledge graph.

        LLM calls tagged: phase="agent_reasoning", actor="graph_extract".
        """
        if not fact.strip():
            return
        self._run_sync(_update_with_context(self._rag, fact))
        logger.debug("LightRAGMemory: update_fact inserted %d-char fact", len(fact))

    async def _drop_all_storages(self) -> None:
        # LightRAG uses process-global shared dicts keyed by namespace — simply
        # deleting the working directory does NOT clear those in-memory dicts, so
        # a fresh LightRAG instance would still see the old data via
        # get_namespace_data(). Calling drop() on every storage object clears
        # the shared dict AND writes the empty state to disk, giving us a true
        # clean slate without needing to destroy the LightRAG instance.
        for storage in [
            self._rag.full_docs,
            self._rag.text_chunks,
            self._rag.full_entities,
            self._rag.full_relations,
            self._rag.entity_chunks,
            self._rag.relation_chunks,
            self._rag.llm_response_cache,
            self._rag.doc_status,
            self._rag.entities_vdb,
            self._rag.relationships_vdb,
            self._rag.chunks_vdb,
            self._rag.chunk_entity_relation_graph,
        ]:
            await storage.drop()

    def reset(self) -> None:
        """
        Clear all LightRAG storage by calling drop() on every storage object.

        This clears both the process-global in-memory shared dicts and the
        on-disk files. The existing LightRAG instance and background loop are
        reused — no rebuild needed.
        """
        self._run_sync(self._drop_all_storages())
        logger.debug("LightRAGMemory: reset complete")

    def get_backend_name(self) -> str:
        return "lightrag"
