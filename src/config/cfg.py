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
    top_k: int
    similarity_cutoff: float


@dataclass
class IngestionCfg:
    chunk_size: int
    chunk_overlap: int


@dataclass
class GraphCfg:
    max_hops: int
    top_k: int = 10          # Override retrieval.top_k — triples are short, need more context
    # Safety caps against hub explosion during BFS expansion; every cap hit
    # is logged as a warning so truncated retrievals are visible in the logs.
    max_frontier: int = 200      # max entities per hop frontier
    max_candidates: int = 1000   # max candidate triples before reranking
    # Cross-encoder reranker shared by all graph variants (src/memory/reranker.py).
    # No LLM call → preserves the zero-cost-retrieval property (device: see Config.device).
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_top_n: int = 10   # Final context size after reranking (default = top_k)


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
