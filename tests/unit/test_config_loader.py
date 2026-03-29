from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.config.settings import ExperimentConfig, _deep_merge, load_config

# Reuse the fixture-friendly minimal YAML dict from conftest
from tests.conftest import MINIMAL_YAML


class TestDeepMerge:
    def test_nested_key_override(self) -> None:
        base = {"llm": {"model": "ollama/a", "temperature": 0.0}}
        override = {"llm": {"temperature": 1.0}}
        result = _deep_merge(base, override)
        assert result["llm"]["model"] == "ollama/a"
        assert result["llm"]["temperature"] == 1.0

    def test_top_level_key_added(self) -> None:
        base = {"llm": {"model": "ollama/a"}}
        override = {"memory_backend": "vector"}
        result = _deep_merge(base, override)
        assert result["memory_backend"] == "vector"
        assert result["llm"]["model"] == "ollama/a"

    def test_does_not_mutate_base(self) -> None:
        base = {"llm": {"model": "ollama/a"}}
        _deep_merge(base, {"llm": {"model": "ollama/b"}})
        assert base["llm"]["model"] == "ollama/a"

    def test_does_not_mutate_override(self) -> None:
        base = {"llm": {"model": "ollama/a"}}
        override = {"llm": {"model": "ollama/b"}}
        _deep_merge(base, override)
        assert override["llm"]["model"] == "ollama/b"

    def test_non_dict_value_replaced(self) -> None:
        base = {"top_k": 5}
        override = {"top_k": 10}
        assert _deep_merge(base, override)["top_k"] == 10


class TestLoadConfig:
    def test_loads_variant_with_correct_name(self, sample_config_path: Path) -> None:
        config = load_config("test_variant", config_path=sample_config_path)
        assert config.variant_name == "test_variant"

    def test_loads_default_llm_values(self, sample_config_path: Path) -> None:
        config = load_config("test_variant", config_path=sample_config_path)
        assert config.llm.model == "ollama/test-model"
        assert config.llm.temperature == 0.0
        assert config.llm.max_tokens == 512

    def test_loads_agent_type(self, sample_config_path: Path) -> None:
        config = load_config("test_variant", config_path=sample_config_path)
        assert config.agent_type == "dummy"

    def test_loads_data_file(self, sample_config_path: Path) -> None:
        config = load_config("test_variant", config_path=sample_config_path)
        assert config.data_file == "test_dataset.json"

    def test_variant_overrides_agent_type(self, sample_config_path: Path) -> None:
        config = load_config("test_vector", config_path=sample_config_path)
        assert config.agent_type == "vector"
        # LLM defaults should still be present
        assert config.llm.model == "ollama/test-model"

    def test_returns_experiment_config_instance(self, sample_config_path: Path) -> None:
        config = load_config("test_variant", config_path=sample_config_path)
        assert isinstance(config, ExperimentConfig)

    def test_unknown_variant_raises_key_error(self, sample_config_path: Path) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            load_config("nonexistent", config_path=sample_config_path)

    def test_error_message_lists_available_variants(self, sample_config_path: Path) -> None:
        with pytest.raises(KeyError) as exc_info:
            load_config("nonexistent", config_path=sample_config_path)
        assert "test_variant" in str(exc_info.value)

    def test_model_without_provider_prefix_raises(self, tmp_path: Path) -> None:
        import copy
        bad = copy.deepcopy(MINIMAL_YAML)
        bad["default"]["llm"]["model"] = "mixtral"  # missing provider/
        bad_path = tmp_path / "bad.yaml"
        bad_path.write_text(yaml.dump(bad))
        with pytest.raises(ValidationError, match="provider"):
            load_config("test_variant", config_path=bad_path)

    def test_embedding_without_provider_prefix_raises(self, tmp_path: Path) -> None:
        import copy
        bad = copy.deepcopy(MINIMAL_YAML)
        bad["default"]["embedding"]["model"] = "nomic-embed"  # missing provider/
        bad_path = tmp_path / "bad.yaml"
        bad_path.write_text(yaml.dump(bad))
        with pytest.raises(ValidationError, match="provider"):
            load_config("test_variant", config_path=bad_path)

    def test_overlap_equal_to_chunk_raises(self, tmp_path: Path) -> None:
        import copy
        bad = copy.deepcopy(MINIMAL_YAML)
        bad["default"]["ingestion"] = {"chunk_size": 100, "chunk_overlap": 100}
        bad_path = tmp_path / "bad.yaml"
        bad_path.write_text(yaml.dump(bad))
        with pytest.raises(ValidationError):
            load_config("test_variant", config_path=bad_path)

    def test_overlap_greater_than_chunk_raises(self, tmp_path: Path) -> None:
        import copy
        bad = copy.deepcopy(MINIMAL_YAML)
        bad["default"]["ingestion"] = {"chunk_size": 100, "chunk_overlap": 150}
        bad_path = tmp_path / "bad.yaml"
        bad_path.write_text(yaml.dump(bad))
        with pytest.raises(ValidationError):
            load_config("test_variant", config_path=bad_path)

    def test_default_config_path_resolves(self) -> None:
        # Smoke test: load from the real unified_config.yaml
        config = load_config("baseline_dummy")
        assert config.variant_name == "baseline_dummy"
        assert config.agent_type == "dummy"
