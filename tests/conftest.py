from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config.settings import ExperimentConfig, load_config

# Minimal YAML used by all unit tests — avoids depending on the real unified_config.yaml
MINIMAL_YAML: dict = {
    "telemetry": {"log_level": "DEBUG", "output_dir": "evaluations/"},
    "default": {
        "agent_type": "dummy",
        "data_file": "test_dataset.json",
        "llm": {"model": "ollama/test-model", "temperature": 0.0, "max_tokens": 512},
        "embedding": {"model": "ollama/test-embed", "batch_size": 8},
        "retrieval": {"top_k": 3, "similarity_cutoff": 0.5},
        "ingestion": {"chunk_size": 256, "chunk_overlap": 32},
    },
    "variants": {
        "test_variant": {"agent_type": "dummy"},
        "test_vector": {"agent_type": "vector"},
        "test_graph": {"agent_type": "graph"},
    },
}


@pytest.fixture()
def sample_config_path(tmp_path: Path) -> Path:
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml.dump(MINIMAL_YAML))
    return config_file


@pytest.fixture()
def dummy_experiment_config(sample_config_path: Path) -> ExperimentConfig:
    return load_config("test_variant", config_path=sample_config_path)
