from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod

from src.config.cfg import Config, resolve_device
from .extraction import _SYSTEM_PROMPT, TRIPLE_SCHEMA, parse_triples
from .traversal import _BeamConfig, _beam_search

logger = logging.getLogger(__name__)


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping word-boundary chunks.

    chunk_size and chunk_overlap are measured in words, not characters,
    which is more stable across different paragraph lengths.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - chunk_overlap

    return chunks


def _make_extract_prompt():
    """PromptTemplate wrapping the shared extraction system prompt.

    SimpleLLMPathExtractor calls ``llm.apredict(prompt, text=..., max_knowledge_triplets=...)``
    on a completion-style template, so we append the ``{text}`` slot it fills per chunk.
    LlamaIndex's SafeFormatter substitutes only known keys, leaving the literal
    JSON braces in the prompt (e.g. ``{"triples": []}``) untouched — no escaping
    needed. ``{max_knowledge_triplets}`` is passed by the extractor but ignored
    here because the prompt already hard-caps the count.
    """
    from llama_index.core.prompts import PromptTemplate

    return PromptTemplate(_SYSTEM_PROMPT + "\n\nText:\n{text}\n\nJSON object:")


def _parse_triplets_fn(response: str) -> list[tuple[str, str, str]]:
    """Adapt the shared parser to the tuple shape the extractor expects.

    Replaces LlamaIndex's ``default_parse_triplets_fn``, which drops any object
    containing a comma and lowercases entity names via ``.capitalize()``. Delegating
    to ``parse_triples`` gives JSON parsing with regex fallback, list-object
    splitting, and predicate normalisation.
    """
    return [
        (t["subject"].strip(), t["predicate"].strip(), t["object"].strip())
        for t in parse_triples(response)
    ]


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


def _build_llm(llm_cfg, phase: str, actor: str, run_id: str, structured: bool = True):
    """
    Construct a LlamaIndex LiteLLM adapter with telemetry metadata baked in.

    additional_kwargs are merged into _model_kwargs and forwarded verbatim to
    every litellm.completion(**all_kwargs) call, so metadata lands in
    litellm_params["metadata"] where TelemetryTracker picks it up.

    With structured=True the shared TRIPLE_SCHEMA is forwarded as
    response_format so Ollama enforces the JSON grammar during extraction
    (parse_triples' regex fallback still covers providers that ignore it).
    """
    from llama_index.llms.litellm import LiteLLM

    additional_kwargs: dict = {
        "metadata": {
            "phase": phase,
            "actor": actor,
            "variant_name": "unified",
            "run_id": run_id,
        }
    }
    if structured:
        additional_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "triples", "schema": TRIPLE_SCHEMA},
        }

    return LiteLLM(
        model=llm_cfg.model,
        api_base=llm_cfg.base_url,
        temperature=llm_cfg.temperature,
        max_tokens=llm_cfg.max_tokens,
        additional_kwargs=additional_kwargs,
    )


class BaseMemory(ABC):
    """
    Neo4j-backed memory base: one store, one ingest, five retrieval views.

    Phase A (offline ingestion) is implemented ONCE here: documents are chunked,
    every chunk is stored as a ``(:Chunk {text, embedding})`` node, and a
    LlamaIndex PropertyGraphIndex extracts (subject, predicate, object) triples
    into ``(:__Entity__)`` nodes and typed relationships. Two chunk-level
    indexes (Lucene full-text and native vector) are created so the exact same
    ingested data serves all retrieval variants.

    Phase B: subclasses implement only ``search()`` as a retrieval view over
    the shared store. They must return plain strings, never backend-specific
    objects, and must not issue LLM calls at retrieval time unless tagged via
    the telemetry tracker.

    The LangGraph agent interacts with memory EXCLUSIVELY through this
    interface and must not know which retrieval view it is talking to.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._neo4j_cfg = config.stores.neo4j
        self._fulltext_index: str = config.stores.chunk_fulltext_index
        self._vector_index: str = config.stores.chunk_vector_index
        self._embed_model = None  # lazy HuggingFaceEmbedding (device from config)
        self._graph_store = None  # lazy Neo4jPropertyGraphStore
        self._driver = None  # lazy neo4j driver
        # Wall-clock split of the last ingest, read by 02_setup for the
        # per-variant cost attribution (chunk/embed = vector+bm25 share,
        # extraction = graph share).
        self.ingest_timings: dict[str, float] = {}
        self._wait_for_neo4j()

    # ── Neo4j plumbing ────────────────────────────────────────────────────────

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
                logger.debug("%s: Neo4j ready at %s", type(self).__name__, uri)
                return
            except Exception as exc:
                last_exc = exc
                time.sleep(2)

        raise RuntimeError(
            f"Neo4j not reachable at {uri} after {timeout}s. "
            f"Is the container running?  docker compose up -d neo4j\n"
            f"Last error: {last_exc}"
        )

    @property
    def driver(self):
        if self._driver is None:
            import neo4j

            self._driver = neo4j.GraphDatabase.driver(
                self._neo4j_cfg.uri,
                auth=(self._neo4j_cfg.username, self._neo4j_cfg.password),
            )
        return self._driver

    def _cypher(self, query: str, /, **params) -> list:
        """Run one Cypher statement and return the fully materialised records.

        ``query`` is positional-only so Cypher parameters may use that name.
        """
        with self.driver.session(database=self._neo4j_cfg.database) as session:
            return list(session.run(query, **params))

    @property
    def embed_model(self):
        """Shared local embedder (LlamaIndex HuggingFaceEmbedding), loaded lazily
        so views that never embed (bm25, graph) skip the model-load cost.
        Runs on ``config.device`` (auto → cuda when available); the model and
        weights are unchanged, so stored vectors remain comparable across devices."""
        if self._embed_model is None:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            device = resolve_device(self._config.device)
            logger.info(
                "%s: loading embedder %s on %s",
                type(self).__name__, self._config.embedding.model, device,
            )
            self._embed_model = HuggingFaceEmbedding(
                model_name=self._config.embedding.model,
                device=device,
            )
        return self._embed_model

    @property
    def graph_store(self):
        if self._graph_store is None:
            from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

            self._graph_store = Neo4jPropertyGraphStore(
                username=self._neo4j_cfg.username,
                password=self._neo4j_cfg.password,
                url=self._neo4j_cfg.uri,
                database=self._neo4j_cfg.database,
            )
        return self._graph_store

    # ── Phase B interface ─────────────────────────────────────────────────────

    @abstractmethod
    def search(self, query: str) -> list[str]:
        """
        Retrieve relevant text passages from the shared store.

        Args:
            query: Natural language query string from the agent.

        Returns:
            Text passages most relevant to the query, ranked by relevance.
            Returns an empty list if no relevant passages are found.
            Never raises on empty results.
        """
        ...

    def get_backend_name(self) -> str:
        """Returns the string identifier for this backend. Used for telemetry tagging."""
        return self.__class__.__name__.lower()

    def _expand_triples(self, seed_nodes: list[str]) -> tuple[list[str], set[str], dict]:
        """Global hop-by-hop BFS over the ``:__Entity__`` subgraph (graph variant).

        The frontier starts as *all* seed entities and each hop is one batched Cypher
        query (``max_hops`` queries total), so 1-hop edges are emitted before 2- and
        3-hop edges and survive downstream truncation. Returns a flat list of single
        triples for the caller to rerank — the configuration the eval settled on as
        best for the graph variant's precise BM25 seeds (chain-ranking regressed it).

        Hub explosion is bounded by ``graph.max_frontier`` (entities per hop) and
        ``graph.max_candidates`` (total triples); every cap hit is logged as a warning
        so truncated retrievals are visible in the logs.

        Returns:
            (triple strings "s p o", visited entity names — seeds plus every entity
            reached within max_hops, consumed by the *text variants'
            ``_collect_chunks`` — and a stats dict with cypher_queries / candidates /
            caps_hit for retrieval-overhead telemetry).
        """
        g = self._config.graph
        seen_edges: set[tuple[str, str, str]] = set()
        results: list[str] = []
        visited: set[str] = set(seed_nodes)
        frontier: list[str] = list(seed_nodes)
        caps_hit = False
        n_queries = 0

        for hop in range(1, g.max_hops + 1):
            if len(frontier) > g.max_frontier:
                logger.warning(
                    "_expand_triples: frontier cap hit at hop %d — %d entities "
                    "truncated to max_frontier=%d (seeds=%r...)",
                    hop, len(frontier), g.max_frontier, seed_nodes[:3],
                )
                caps_hit = True
                frontier = frontier[: g.max_frontier]

            rows = self._cypher(
                "MATCH (n:__Entity__)-[r]-(m:__Entity__) "
                "WHERE n.name IN $frontier "
                "RETURN DISTINCT startNode(r).name AS s, type(r) AS p, "
                "endNode(r).name AS o, m.name AS neighbor "
                "ORDER BY s, p, o LIMIT $row_limit",
                frontier=frontier,
                # +1 headroom so a cap hit is distinguishable from an exact fit.
                row_limit=g.max_candidates + 1,
            )
            n_queries += 1

            next_frontier: set[str] = set()
            for row in rows:
                s, p, o = row["s"], row["p"], row["o"]
                if s is None or o is None:
                    continue
                # Predicates are stored verbatim as relationship types (snake_case);
                # .lower() is a safety net in case a future store version normalises it.
                p = (p or "related_to").lower()
                key = (s, p, o)
                if key not in seen_edges:
                    seen_edges.add(key)
                    results.append(f"{s} {p.replace('_', ' ')} {o}")
                neighbor = row["neighbor"]
                if neighbor is not None and neighbor not in visited:
                    next_frontier.add(neighbor)

            if len(results) >= g.max_candidates:
                logger.warning(
                    "_expand_triples: candidate cap hit at hop %d — %d triples "
                    "truncated to max_candidates=%d (seeds=%r...)",
                    hop, len(results), g.max_candidates, seed_nodes[:3],
                )
                caps_hit = True
                results = results[: g.max_candidates]
                break

            visited |= next_frontier
            frontier = sorted(next_frontier)
            if not frontier:
                break

        stats = {
            "cypher_queries": n_queries,
            "candidates": len(results),
            "caps_hit": caps_hit,
        }
        return results, visited, stats

    def _collect_chunks(self, entities: set[str]) -> list[str]:
        """Source chunks mentioning the given (traversed) entities, best first.

        Shared by the text variants (graphtext / vectorgraphtext): after the BFS has
        produced a visited-entity set, this pulls the chunk passages those entities
        were extracted from. Ranked by a mention-overlap prior (chunks mentioning
        more traversed entities first — multi-hop bridge passages score highest) and
        capped at ``retrieval.rerank_fetch_k`` so a 3-hop entity explosion cannot
        flood the cross-encoder (latency guard; the reranker does the final cut).
        """
        if not entities:
            return []
        rows = self._cypher(
            "MATCH (c:Chunk)-[:MENTIONS]->(e:__Entity__) "
            "WHERE e.name IN $entities "
            "WITH c, count(DISTINCT e) AS overlap "
            "ORDER BY overlap DESC "
            "LIMIT $fetch_k "
            "RETURN c.text AS text",
            entities=sorted(entities),
            fetch_k=self._config.retrieval.rerank_fetch_k,
        )
        return [r["text"] for r in rows if r["text"] and r["text"].strip()]

    def _expand_beam(self, query: str, seed_nodes: list[str], reranker) -> tuple[list[str], dict]:
        """Beam-guided, chain-scored traversal of the ``:__Entity__`` subgraph (vectorgraph).

        Thin I/O wrapper: binds ``self._cypher`` and the reranker to the pure
        ``_beam_search`` algorithm (``src/memory/traversal.py``), which grows reasoning
        chains outward from the seeds, scores whole chains, keeps a per-node-capped
        top-K beam, and returns the deduplicated triples of the top chains. Used by the
        vectorgraph variant, whose noisy chunk-anchored seeds benefit from the caps;
        the graph variant uses the simpler flat ``_expand_triples`` + single-triple rerank.

        Returns:
            (deduplicated triples of the top-ranked chains, stats dict with
            cypher/scoring timings and counts for retrieval-overhead telemetry).
        """
        g = self._config.graph
        beam_cfg = _BeamConfig(
            max_hops=g.max_hops,
            beam_width=g.beam_width,
            max_per_tail=g.max_per_tail,
            # Final beam size = the shared final-context knob (retrieval.top_k),
            # so vectorgraph returns the same item count as every other variant.
            rerank_top_n=self._config.retrieval.top_k,
        )

        def fetch_edges(tails: list[str]):
            return self._cypher(
                "MATCH (n:__Entity__)-[r]-(m:__Entity__) "
                "WHERE n.name IN $frontier "
                "RETURN DISTINCT n.name AS anchor, startNode(r).name AS s, "
                "type(r) AS p, endNode(r).name AS o, m.name AS neighbor "
                "ORDER BY s, p, o LIMIT $row_limit",
                frontier=tails,
                # Per-hop DB safety bound (a beam tail may be a hub).
                row_limit=g.max_candidates,
            )

        return _beam_search(query, seed_nodes, beam_cfg, fetch_edges, reranker.score)

    # ── Phase A: unified ingest ───────────────────────────────────────────────

    def ingest_documents(self, documents: list[str], start_from: int = 0) -> None:
        """
        Chunk, extract, and persist documents into the shared Neo4j store.

        One run serves every retrieval view: chunks land as ``:Chunk`` nodes
        (embedded + full-text indexed), triples as ``:__Entity__`` nodes and
        typed relationships. ``start_from`` (a chunk index, see
        ``read_checkpoint``) resumes an interrupted extraction; chunking is
        deterministic, so indices are stable across runs.
        """
        from llama_index.core import Document, PropertyGraphIndex

        chunk_size = self._config.ingestion.chunk_size
        chunk_overlap = self._config.ingestion.chunk_overlap

        chunks: list[str] = []
        for doc in documents:
            stripped = doc.strip()
            if stripped:
                chunks.extend(_chunk_text(stripped, chunk_size, chunk_overlap))
        if not chunks:
            logger.warning("BaseMemory: no chunks produced from %d documents", len(documents))
            return

        llm = _build_llm(
            self._config.llm.ingest,
            phase="ingest",
            actor="graph_extract",
            run_id="ingest",
        )
        extractor = _make_flushing_extractor(
            llm=llm,
            extract_prompt=_make_extract_prompt(),
            parse_fn=_parse_triplets_fn,
            # Prompt hard-caps at 20 triples/chunk.
            max_paths_per_chunk=20,
            num_workers=1,
        )
        # Bind an index to the store and insert chunk by chunk so we get
        # incremental node/edge progress and checkpointable resume.
        index = PropertyGraphIndex.from_existing(
            property_graph_store=self.graph_store,
            kg_extractors=[extractor],
            embed_model=self.embed_model,
            llm=llm,
        )

        total = len(chunks)
        t0 = time.perf_counter()
        for i, text in enumerate(chunks[start_from:], start_from + 1):
            index.insert(Document(text=text))
            if i % 10 == 0 or i == total:
                print(
                    f"  [{i:4d}/{total}]  chunks={self.chunk_count}  "
                    f"entities={self.entity_count}  edges={self.edge_count}",
                    flush=True,
                )
            if i % 50 == 0:
                self._write_checkpoint(i)
        self._write_checkpoint(total)
        self.ingest_timings["graph_extract_s"] = round(time.perf_counter() - t0, 2)

        t0 = time.perf_counter()
        embedded = self._embed_missing_chunks()
        self.ingest_timings["chunk_embed_s"] = round(time.perf_counter() - t0, 2)
        self._create_chunk_indexes()

        logger.info(
            "BaseMemory: ingested %d chunks (%d embedded) → %d entities, %d edges",
            total, embedded, self.entity_count, self.edge_count,
        )

    def _embed_missing_chunks(self) -> int:
        """Set ``embedding`` on every :Chunk node that lacks one (vector view).

        Embeddings are computed locally (sentence-transformers via
        HuggingFaceEmbedding) — no LLM call, hence no telemetry record; the
        wall-clock cost is reported via ``ingest_timings``.
        """
        rows = self._cypher(
            "MATCH (c:Chunk) WHERE c.embedding IS NULL "
            "RETURN elementId(c) AS eid, c.text AS text"
        )
        if not rows:
            return 0

        batch_size = self._config.embedding.batch_size
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            embeddings = self.embed_model.get_text_embedding_batch(
                [r["text"] for r in batch]
            )
            self._cypher(
                "UNWIND $rows AS row "
                "MATCH (c:Chunk) WHERE elementId(c) = row.eid "
                "CALL db.create.setNodeVectorProperty(c, 'embedding', row.embedding) "
                "RETURN count(c)",
                rows=[
                    {"eid": r["eid"], "embedding": emb}
                    for r, emb in zip(batch, embeddings)
                ],
            )
        return len(rows)

    def _create_chunk_indexes(self) -> None:
        """Idempotently create the chunk-level full-text and vector indexes,
        plus the entity-name range index the BFS frontier lookups filter on."""
        self._cypher(
            f"CREATE FULLTEXT INDEX {self._fulltext_index} IF NOT EXISTS "
            f"FOR (c:Chunk) ON EACH [c.text]"
        )
        self._cypher(
            "CREATE INDEX entity_name IF NOT EXISTS "
            "FOR (n:__Entity__) ON (n.name)"
        )
        dimensions = len(self.embed_model.get_text_embedding("dimension probe"))
        self._cypher(
            f"CREATE VECTOR INDEX {self._vector_index} IF NOT EXISTS "
            f"FOR (c:Chunk) ON c.embedding "
            f"OPTIONS {{indexConfig: {{"
            f"`vector.dimensions`: {dimensions}, "
            f"`vector.similarity_function`: 'cosine'}}}}"
        )

    # ── Checkpoint (resume) ───────────────────────────────────────────────────

    def read_checkpoint(self) -> int:
        rows = self._cypher(
            "MATCH (m:Meta {key: 'ingest_checkpoint'}) RETURN m.value AS v"
        )
        return int(rows[0]["v"]) if rows else 0

    def _write_checkpoint(self, chunk_index: int) -> None:
        self._cypher(
            "MERGE (m:Meta {key: 'ingest_checkpoint'}) SET m.value = $i",
            i=chunk_index,
        )

    # ── Store lifecycle ───────────────────────────────────────────────────────

    def reset(self) -> None:
        """Wipe the shared store: all nodes/relationships and both chunk indexes.

        Only the unified ingest (02_setup) may call this — a reset from any
        retrieval view would destroy the data of every other variant.
        """
        self._cypher("MATCH (n) DETACH DELETE n")
        self._cypher(f"DROP INDEX {self._fulltext_index} IF EXISTS")
        self._cypher(f"DROP INDEX {self._vector_index} IF EXISTS")
        logger.debug("BaseMemory: reset — store wiped, chunk indexes dropped")

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def chunk_count(self) -> int:
        return self._count("MATCH (c:Chunk) RETURN count(c) AS c")

    @property
    def entity_count(self) -> int:
        return self._count("MATCH (n:__Entity__) RETURN count(n) AS c")

    @property
    def edge_count(self) -> int:
        """Entity-to-entity relationships (excludes chunk MENTIONS edges)."""
        return self._count(
            "MATCH (:__Entity__)-[r]->(:__Entity__) RETURN count(r) AS c"
        )

    def _count(self, cypher: str) -> int:
        try:
            return self._cypher(cypher)[0]["c"]
        except Exception:
            return 0


class UnifiedMemoryStore(BaseMemory):
    """Ingest-only handle on the shared store (used by 02_setup).

    Phase A is variant-independent, so the setup pipeline needs the base
    functionality without any retrieval view attached.
    """

    def search(self, query: str) -> list[str]:  # pragma: no cover - not a retrieval view
        raise NotImplementedError(
            "UnifiedMemoryStore is ingest-only; instantiate a retrieval view "
            "(BM25Memory, VectorMemory, VectorRerankMemory, GraphMemory, "
            "GraphTextMemory, VectorGraphMemory, VectorGraphTextMemory) to search."
        )

    def get_backend_name(self) -> str:
        return "unified"
