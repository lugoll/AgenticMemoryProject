"""
Unified config loader — liest src/config/unified_config.yaml und gibt
typisierte Dataclass-Objekte zurück. Kein Pydantic, kein Overhead.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


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


@dataclass
class StoresCfg:
    bm25: str
    graph: str
    vector: str


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
        stores=StoresCfg(**raw["stores"]),
        telemetry=TelemetryCfg(**raw["telemetry"]),
    )
