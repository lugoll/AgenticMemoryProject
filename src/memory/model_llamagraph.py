from __future__ import annotations

import asyncio
import logging
import time

from src.config.cfg import Config
from .base import BaseMemory

logger = logging.getLogger(__name__)


def _make_flushing_extractor(**kwargs):
    """
    SimpleLLMPathExtractor whose synchronous __call__ drains LiteLLM's async
    logging worker before its event loop is torn down.

    Why this exists:
        PropertyGraphIndex._insert_nodes runs the KG extractors via a throwaway
        ``asyncio.run(arun_transformations(...))`` loop, which awaits the
        extractor's ``acall``. Each per-node extraction is an async
        ``litellm.acompletion``; LiteLLM logs token usage by *enqueuing* the
        success callback onto a global background worker bound to that loop
        (litellm.litellm_core_utils.logging_worker.GLOBAL_LOGGING_WORKER) — it is
        NOT awaited inline (there is no inline logging path for async calls).
        When asyncio.run() closes the loop immediately after the final node, the
        last in-flight callback is cancelled before it executes, so the final
        node's telemetry record is silently dropped. Result: N documents ->
        N-1 logged LLM calls, always missing the last.

        We override ``acall`` (the method actually awaited inside that loop) to
        await GLOBAL_LOGGING_WORKER.flush() — which joins the worker queue —
        before returning, guaranteeing every callback has written its record
        while the loop is still alive.
    """
    from llama_index.core.indices.property_graph import SimpleLLMPathExtractor

    class _FlushingPathExtractor(SimpleLLMPathExtractor):
        async def acall(self, nodes, show_progress: bool = False, **call_kwargs):
            result = await super().acall(
                nodes, show_progress=show_progress, **call_kwargs
            )
            try:
                from litellm.litellm_core_utils.logging_worker import (
                    GLOBAL_LOGGING_WORKER,
                )

                # Bounded so a stuck callback can never hang ingestion.
                await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=30)
            except Exception as exc:  # never let telemetry drain break ingest
                logger.debug("Telemetry worker flush skipped: %s", exc)
            return result

    return _FlushingPathExtractor(**kwargs)


def _build_llm(llm_cfg, phase: str, actor: str, run_id: str):
    """
    Construct a LlamaIndex LiteLLM adapter with telemetry metadata baked in.

    additional_kwargs are merged into _model_kwargs and forwarded verbatim to
    every litellm.completion(**all_kwargs) call, so metadata lands in
    litellm_params["metadata"] where TelemetryTracker picks it up.
    """
    from llama_index.llms.litellm import LiteLLM

    return LiteLLM(
        model=llm_cfg.model,
        api_base=llm_cfg.base_url,
        temperature=llm_cfg.temperature,
        max_tokens=llm_cfg.max_tokens,
        additional_kwargs={
            "metadata": {
                "phase": phase,
                "actor": actor,
                "variant_name": "llamagraph",
                "run_id": run_id,
            }
        },
    )


class LlamaIndexGraphMemory(BaseMemory):
    """
    LlamaIndex PropertyGraphIndex backed by Neo4j.

    Ingest (Phase A):
        Documents are passed to PropertyGraphIndex.from_documents() with a
        SimpleLLMPathExtractor. The LLM (routed via the llama_index.llms.litellm
        adapter) extracts free-form (subject, predicate, object) triples which
        LlamaIndex stores as nodes and relationships in Neo4j.

    Retrieval (Phase B):
        VectorContextRetriever embeds the query with sentence-transformers and
        finds the most similar entity nodes in the Neo4j vector index. No LLM
        call at retrieval time.

    Telemetry:
        Two separate LiteLLM adapter instances are created at construction time,
        each with its telemetry metadata baked into additional_kwargs so every
        litellm.completion call carries the correct phase/actor tags without any
        dynamic state manipulation.
    """

    def __init__(self, config: Config) -> None:
        from llama_index.core import Settings
        from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        self._llm_cfg = config.llm.ingest
        self._max_hops: int = config.graph.max_hops
        self._top_k: int = config.graph.top_k
        self._neo4j_cfg = config.stores.neo4j

        # One client per phase — metadata is static and baked in at construction.
        self._llm_ingest = _build_llm(
            config.llm.ingest,
            phase="ingest",
            actor="graph_extract",
            run_id="ingest",
        )
        self._llm_agent = _build_llm(
            config.llm.ingest,
            phase="agent_reasoning",
            actor="graph_extract",
            run_id="update_fact",
        )

        self._embed_model = HuggingFaceEmbedding(
            model_name=config.embedding.model,
            device="cpu",
        )

        # LlamaIndex globals — ingest client is the default for index construction.
        Settings.llm = self._llm_ingest
        Settings.embed_model = self._embed_model

        self._wait_for_neo4j()

        self._graph_store = Neo4jPropertyGraphStore(
            username=self._neo4j_cfg.username,
            password=self._neo4j_cfg.password,
            url=self._neo4j_cfg.uri,
            database=self._neo4j_cfg.database,
        )

        # Reconstruct index from existing store if data is already present
        self._index = self._try_load_existing_index()

    # ── Neo4j readiness ───────────────────────────────────────────────────────

    def _wait_for_neo4j(self, timeout: int = 60) -> None:
        """Retry Bolt connectivity until Neo4j is ready (handles slow container start)."""
        import neo4j

        uri = self._neo4j_cfg.uri
        auth = (self._neo4j_cfg.username, self._neo4j_cfg.password)
        deadline = time.time() + timeout
        last_exc: Exception | None = None

        while time.time() < deadline:
            try:
                driver = neo4j.GraphDatabase.driver(uri, auth=auth)
                driver.verify_connectivity()
                driver.close()
                logger.debug("LlamaIndexGraphMemory: Neo4j ready at %s", uri)
                return
            except Exception as exc:
                last_exc = exc
                time.sleep(2)

        raise RuntimeError(
            f"Neo4j not reachable at {uri} after {timeout}s. "
            f"Is the container running?  docker compose up -d neo4j\n"
            f"Last error: {last_exc}"
        )

    # ── Index lifecycle ───────────────────────────────────────────────────────

    def _try_load_existing_index(self):
        """Return a PropertyGraphIndex from the existing Neo4j store, or None."""
        from llama_index.core import PropertyGraphIndex

        try:
            index = PropertyGraphIndex.from_existing(
                property_graph_store=self._graph_store,
                embed_model=self._embed_model,
                llm=self._llm_ingest,
            )
            count = self.node_count
            if count > 0:
                logger.debug(
                    "LlamaIndexGraphMemory: loaded existing index (%d nodes)", count
                )
                return index
        except Exception as exc:
            logger.debug("LlamaIndexGraphMemory: could not load existing index: %s", exc)
        return None

    # ── BaseMemory interface ──────────────────────────────────────────────────

    def ingest_documents(self, documents: list[str]) -> None:
        from llama_index.core import Document, PropertyGraphIndex

        non_empty = [d for d in documents if d.strip()]
        dropped = len(documents) - len(non_empty)
        if dropped:
            logger.warning(
                "LlamaIndexGraphMemory: dropped %d/%d empty or whitespace-only "
                "document(s) before ingest — %d will be processed",
                dropped,
                len(documents),
                len(non_empty),
            )
        if not non_empty:
            logger.warning("LlamaIndexGraphMemory: no documents to ingest")
            return

        li_docs = [Document(text=d) for d in non_empty]
        extractor = _make_flushing_extractor(
            llm=self._llm_ingest,
            max_paths_per_chunk=100,
            num_workers=1,
        )
        self._index = PropertyGraphIndex.from_documents(
            li_docs,
            kg_extractors=[extractor],
            embed_model=self._embed_model,
            property_graph_store=self._graph_store,
            show_progress=True,
        )

        logger.info(
            "LlamaIndexGraphMemory: ingested %d documents → %d nodes",
            len(non_empty),
            self.node_count,
        )

    def search(self, query: str) -> list[str]:
        from llama_index.core.indices.property_graph import VectorContextRetriever

        if not query.strip() or self._index is None:
            return []

        retriever = VectorContextRetriever(
            self._index.property_graph_store,
            embed_model=self._embed_model,
            include_text=False,
            similarity_top_k=self._top_k,
        )
        nodes = retriever.retrieve(query)
        results = [n.get_content() for n in nodes if n.get_content().strip()]
        logger.debug(
            "LlamaIndexGraphMemory: query=%r → %d results", query, len(results)
        )
        return results

    def update_fact(self, fact: str) -> None:
        from llama_index.core import Document, PropertyGraphIndex

        if not fact.strip() or self._index is None:
            return

        # Build a fresh extractor with the agent-reasoning-tagged LLM so the
        # telemetry for this single-fact update is tagged correctly.
        extractor = _make_flushing_extractor(
            llm=self._llm_agent,
            max_paths_per_chunk=20,
            num_workers=1,
        )
        index = PropertyGraphIndex.from_existing(
            property_graph_store=self._graph_store,
            kg_extractors=[extractor],
            embed_model=self._embed_model,
            llm=self._llm_agent,
        )
        index.insert(Document(text=fact))
        logger.debug("LlamaIndexGraphMemory: updated fact=%r", fact[:80])

    def reset(self) -> None:
        import neo4j

        uri = self._neo4j_cfg.uri
        auth = (self._neo4j_cfg.username, self._neo4j_cfg.password)

        with neo4j.GraphDatabase.driver(uri, auth=auth) as driver:
            with driver.session(database=self._neo4j_cfg.database) as session:
                session.run("MATCH (n) DETACH DELETE n")

        self._index = None
        logger.debug("LlamaIndexGraphMemory: reset — all nodes deleted")

    def get_backend_name(self) -> str:
        return "llamagraph"

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        import neo4j

        uri = self._neo4j_cfg.uri
        auth = (self._neo4j_cfg.username, self._neo4j_cfg.password)
        try:
            with neo4j.GraphDatabase.driver(uri, auth=auth) as driver:
                with driver.session(database=self._neo4j_cfg.database) as session:
                    result = session.run("MATCH (n) RETURN count(n) AS c")
                    return result.single()["c"]
        except Exception:
            return 0
