"""
Unified config loader — liest src/config/unified_config.yaml und gibt
typisierte Dataclass-Objekte zurück. Kein Pydantic, kein Overhead.

Env var overrides (useful for native Ollama on Mac, no Docker):
  OLLAMA_AGENT_URL  — overrides llm.agent.base_url and llm.ingest.base_url
  OLLAMA_JUDGE_URL  — overrides llm.judge.base_url
  NEO4J_URI         — overrides stores.neo4j.uri
  NEO4J_PASSWORD    — overrides stores.neo4j.password
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMCfg:
    model: str
    base_url: str
    temperature: float
    max_tokens: int


@dataclass
class LLMsCfg:
    agent: LLMCfg
    ingest: LLMCfg
    judge: LLMCfg


@dataclass
class EmbeddingCfg:
    model: str
    batch_size: int


@dataclass
class RetrievalCfg:
    # Single final-context-size knob for EVERY variant: the number of items
    # (passages / triples / chains) each backend returns after its final
    # truncation or cross-encoder rerank. Aligned across variants so token
    # differences reflect unit size (triple vs passage), not config asymmetry.
    top_k: int
    similarity_cutoff: float
    # Candidate pool the reranking vector variant (VectorRerankMemory) over-fetches
    # from the chunk vector index before the cross-encoder truncates to top_k.
    # Must exceed top_k or the rerank stage is a no-op. Plain VectorMemory ignores
    # this and returns top_k directly.
    rerank_fetch_k: int = 30
    # Hop-0 chunk-vector pool for VectorGraphTextMemory: how many chunks the initial
    # cosine seed pulls before graph expansion adds bridge chunks. Decoupled from the
    # final-context knob so seeding breadth is tunable independently of top_k.
    # None → fall back to top_k (baseline behaviour when the key is absent).
    hop0_fetch_k: int | None = None


@dataclass
class IngestionCfg:
    chunk_size: int
    chunk_overlap: int


@dataclass
class GraphCfg:
    max_hops: int
    # BM25 entity-linking seed count for the BFS (graph + graphtext variants):
    # how many best-matching entity names start the frontier. NOT a final-context
    # size — that is retrieval.top_k for every variant.
    seed_top_k: int = 10
    # Two retrieval paths share this store (eval-selected best per variant):
    #   graph       → flat hop-by-hop BFS (_expand_triples) + single-triple rerank.
    #                 Hub explosion bounded by max_frontier (entities/hop) and
    #                 max_candidates (total triples); cap hits are logged.
    #   vectorgraph → beam path traversal (traversal.py): chains scored whole, a
    #                 per-node-capped top-K beam kept (beam_width / max_per_tail).
    max_frontier: int = 200      # graph: max entities per hop frontier
    max_candidates: int = 1000   # max candidate triples / per-hop Cypher row bound
    beam_width: int = 10         # vectorgraph: chains kept per hop
    max_per_tail: int = 3        # vectorgraph: max chains sharing one tail/source node per hop
    # Cross-encoder reranker shared by all graph variants (src/memory/reranker.py).
    # No LLM call → preserves the zero-cost-retrieval property (device: see Config.device).
    # Final size after reranking is retrieval.top_k (the shared knob), not a graph field.
    rerank_model: str = "BAAI/bge-reranker-base"


@dataclass
class Neo4jCfg:
    uri: str
    username: str
    password: str
    database: str = "neo4j"


@dataclass
class StoresCfg:
    neo4j: Neo4jCfg
    # Chunk-level index names in the shared Neo4j store.
    chunk_fulltext_index: str = "chunk_fulltext"
    chunk_vector_index: str = "chunk_vector"


@dataclass
class TelemetryCfg:
    output_dir: str


@dataclass
class Config:
    llm: LLMsCfg
    embedding: EmbeddingCfg
    retrieval: RetrievalCfg
    ingestion: IngestionCfg
    graph: GraphCfg
    stores: StoresCfg
    telemetry: TelemetryCfg
    # Device for all local torch models (embedder + cross-encoder reranker):
    # "auto" | "cpu" | "cuda". Resolve via resolve_device().
    device: str = "auto"


def resolve_device(setting: str) -> str:
    """Map the config ``device`` setting to a concrete torch device string."""
    if setting != "auto":
        return setting
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_config(path: Path = Path("src/config/unified_config.yaml")) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    if url := os.environ.get("OLLAMA_AGENT_URL"):
        raw["llm"]["agent"]["base_url"] = url
        raw["llm"]["ingest"]["base_url"] = url
    if url := os.environ.get("OLLAMA_JUDGE_URL"):
        raw["llm"]["judge"]["base_url"] = url
    if uri := os.environ.get("NEO4J_URI"):
        raw["stores"]["neo4j"]["uri"] = uri
    if pw := os.environ.get("NEO4J_PASSWORD"):
        raw["stores"]["neo4j"]["password"] = pw

    return Config(
        llm=LLMsCfg(
            agent=LLMCfg(**raw["llm"]["agent"]),
            ingest=LLMCfg(**raw["llm"]["ingest"]),
            judge=LLMCfg(**raw["llm"]["judge"]),
        ),
        embedding=EmbeddingCfg(**raw["embedding"]),
        retrieval=RetrievalCfg(**raw["retrieval"]),
        ingestion=IngestionCfg(**raw["ingestion"]),
        graph=GraphCfg(**raw["graph"]),
        stores=StoresCfg(
            neo4j=Neo4jCfg(**raw["stores"]["neo4j"]),
            chunk_fulltext_index=raw["stores"].get("chunk_fulltext_index", "chunk_fulltext"),
            chunk_vector_index=raw["stores"].get("chunk_vector_index", "chunk_vector"),
        ),
        telemetry=TelemetryCfg(**raw["telemetry"]),
        device=raw.get("device", "auto"),
    )
