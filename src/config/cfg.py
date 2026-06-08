"""
Unified config loader — liest src/config/unified_config.yaml und gibt
typisierte Dataclass-Objekte zurück. Kein Pydantic, kein Overhead.

Env var overrides (useful for native Ollama on Mac, no Docker):
  OLLAMA_AGENT_URL  — overrides llm.agent.base_url and llm.ingest.base_url
  OLLAMA_JUDGE_URL  — overrides llm.judge.base_url
  CHROMA_HOST       — overrides embedding.chroma_host
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
    chroma_host: str


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


@dataclass
class Neo4jCfg:
    uri: str
    username: str
    password: str
    database: str = "neo4j"


@dataclass
class StoresCfg:
    bm25: str
    graph: str
    vector: str
    neo4j: Neo4jCfg = None  # type: ignore[assignment]


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


def load_config(path: Path = Path("src/config/unified_config.yaml")) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    if url := os.environ.get("OLLAMA_AGENT_URL"):
        raw["llm"]["agent"]["base_url"] = url
        raw["llm"]["ingest"]["base_url"] = url
    if url := os.environ.get("OLLAMA_JUDGE_URL"):
        raw["llm"]["judge"]["base_url"] = url
    if host := os.environ.get("CHROMA_HOST"):
        raw["embedding"]["chroma_host"] = host
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
            bm25=raw["stores"]["bm25"],
            graph=raw["stores"]["graph"],
            vector=raw["stores"]["vector"],
            neo4j=Neo4jCfg(**raw["stores"]["neo4j"]) if "neo4j" in raw["stores"] else None,
        ),
        telemetry=TelemetryCfg(**raw["telemetry"]),
    )
