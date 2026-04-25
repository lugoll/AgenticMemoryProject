from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, field_validator, model_validator


# ---- Leaf models ----

class LLMConfig(BaseModel):
    model: str
    temperature: float = 0.0
    max_tokens: int = 2048
    api_base: Optional[str] = None

    @field_validator("model")
    @classmethod
    def model_must_have_provider_prefix(cls, v: str) -> str:
        if "/" not in v:
            raise ValueError(
                f"LLM model '{v}' must use 'provider/model' format "
                f"(e.g., 'ollama/mixtral', 'openai/gpt-4o')"
            )
        return v

    @model_validator(mode="after")
    def resolve_api_base(self) -> "LLMConfig":
        if self.api_base is None:
            self.api_base = os.environ.get("OLLAMA_API_BASE")
        return self


class EmbeddingConfig(BaseModel):
    model: str
    batch_size: int = 32
    chroma_host: Optional[str] = None

    @field_validator("model")
    @classmethod
    def model_must_have_provider_prefix(cls, v: str) -> str:
        if "/" not in v:
            raise ValueError(
                f"Embedding model '{v}' must use 'provider/model' format"
            )
        return v

    @model_validator(mode="after")
    def resolve_chroma_host(self) -> "EmbeddingConfig":
        if self.chroma_host is None:
            self.chroma_host = os.environ.get("CHROMA_HOST")
        return self


class RetrievalConfig(BaseModel):
    top_k: int = 5
    similarity_cutoff: float = 0.7


class IngestionConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 64

    @model_validator(mode="after")
    def overlap_less_than_chunk(self) -> IngestionConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        return self


class TelemetryConfig(BaseModel):
    log_level: str = "INFO"
    output_dir: str = "evaluations/"


# ---- Top-level experiment config ----

AgentType = Literal["dummy", "vector", "graph", "bm25"]


class ExperimentConfig(BaseModel):
    """Fully resolved config for a single named experiment variant."""
    variant_name: str
    agent_type: AgentType
    data_file: str
    memory_path: Optional[str] = None
    llm: LLMConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    ingestion: IngestionConfig
    telemetry: TelemetryConfig


# ---- Loader ----

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Returns a new dict; does not mutate inputs."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def list_variants(config_path: Path | None = None) -> list[str]:
    """Return all variant names defined in the YAML config."""
    if config_path is None:
        config_path = Path(__file__).parent / "unified_config.yaml"
    with config_path.open("r") as f:
        raw = yaml.safe_load(f)
    return list(raw.get("variants", {}).keys())


def load_config(
    variant_name: str,
    config_path: Path | None = None,
) -> ExperimentConfig:
    """
    Load and resolve a named experiment variant from unified_config.yaml.

    Args:
        variant_name: Key into the `variants` block of the YAML.
        config_path: Override path to the YAML file. Defaults to
                     unified_config.yaml in the same directory as this file.

    Returns:
        A fully-validated ExperimentConfig for the requested variant.

    Raises:
        KeyError: If variant_name is not found in the YAML.
        pydantic.ValidationError: If the merged config fails validation.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "unified_config.yaml"

    with config_path.open("r") as f:
        raw = yaml.safe_load(f)

    defaults: dict = raw["default"]
    telemetry_raw: dict = raw.get("telemetry", {})
    variants: dict = raw.get("variants", {})

    if variant_name not in variants:
        available = list(variants.keys())
        raise KeyError(
            f"Variant '{variant_name}' not found. Available variants: {available}"
        )

    merged = _deep_merge(defaults, variants[variant_name])
    merged["variant_name"] = variant_name
    merged["telemetry"] = telemetry_raw

    return ExperimentConfig.model_validate(merged)
