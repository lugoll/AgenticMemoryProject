from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import litellm
import pytest

from src.telemetry.tracker import TelemetryTracker, register_tracker


def _make_mock_response(prompt_tokens: int = 10, completion_tokens: int = 20) -> MagicMock:
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens
    response = MagicMock()
    response.usage = usage
    return response


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _read_last_record(path: Path) -> dict:
    """Read the last JSON record written to a telemetry JSONL file."""
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return json.loads(lines[-1])


@pytest.fixture(autouse=True)
def reset_litellm_callbacks():
    """Restore litellm.callbacks after each test to avoid cross-test pollution."""
    original = list(litellm.callbacks)
    yield
    litellm.callbacks = original


class TestTelemetryTrackerCallCount:
    def test_initial_call_count_is_zero(self, tmp_path: Path) -> None:
        tracker = TelemetryTracker(output_dir=tmp_path)
        assert tracker.call_count == 0

    def test_call_count_increments_on_success(self, tmp_path: Path) -> None:
        tracker = TelemetryTracker(output_dir=tmp_path)
        now = _now()
        tracker.log_success_event(
            kwargs={"model": "ollama/test", "metadata": {"phase": "ingest", "actor": "vector_embed"}},
            response_obj=_make_mock_response(),
            start_time=now,
            end_time=now,
        )
        assert tracker.call_count == 1

    def test_call_count_does_not_increment_on_failure(self, tmp_path: Path) -> None:
        tracker = TelemetryTracker(output_dir=tmp_path)
        now = _now()
        tracker.log_failure_event(
            kwargs={"model": "ollama/test", "metadata": {}, "exception": "timeout"},
            response_obj=None,
            start_time=now,
            end_time=now,
        )
        assert tracker.call_count == 0


class TestTelemetryTrackerOutput:
    def test_success_writes_to_jsonl_file(self, tmp_path: Path) -> None:
        tracker = TelemetryTracker(output_dir=tmp_path)
        now = _now()
        tracker.log_success_event(
            kwargs={"model": "ollama/test", "metadata": {"phase": "ingest", "actor": "graph_extract"}},
            response_obj=_make_mock_response(),
            start_time=now,
            end_time=now,
        )
        assert tracker.telemetry_path.exists()
        lines = tracker.telemetry_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

    def test_success_output_is_valid_json(self, tmp_path: Path) -> None:
        tracker = TelemetryTracker(output_dir=tmp_path)
        now = _now()
        tracker.log_success_event(
            kwargs={"model": "ollama/test", "metadata": {"phase": "ingest", "actor": "graph_extract"}},
            response_obj=_make_mock_response(),
            start_time=now,
            end_time=now,
        )
        record = _read_last_record(tracker.telemetry_path)
        assert isinstance(record, dict)

    def test_success_record_contains_required_fields(self, tmp_path: Path) -> None:
        tracker = TelemetryTracker(output_dir=tmp_path)
        now = _now()
        tracker.log_success_event(
            kwargs={
                "model": "openai/gpt-4o",
                "metadata": {
                    "phase": "agent_reasoning",
                    "actor": "langgraph_node",
                    "variant_name": "graph_rag_baseline",
                    "run_id": "abc12345",
                },
            },
            response_obj=_make_mock_response(prompt_tokens=100, completion_tokens=50),
            start_time=now,
            end_time=now,
        )
        record = _read_last_record(tracker.telemetry_path)

        assert record["event"] == "llm_call"
        assert record["phase"] == "agent_reasoning"
        assert record["actor"] == "langgraph_node"
        assert record["variant_name"] == "graph_rag_baseline"
        assert record["run_id"] == "abc12345"
        assert record["model"] == "openai/gpt-4o"
        assert record["prompt_tokens"] == 100
        assert record["completion_tokens"] == 50
        assert record["total_tokens"] == 150
        assert "timestamp" in record
        assert "latency_ms" in record
        assert record["call_index"] == 1

    def test_failure_record_contains_error_field(self, tmp_path: Path) -> None:
        tracker = TelemetryTracker(output_dir=tmp_path)
        now = _now()
        tracker.log_failure_event(
            kwargs={
                "model": "ollama/mixtral",
                "metadata": {"phase": "ingest", "actor": "graph_extract"},
                "exception": "Connection refused",
            },
            response_obj=None,
            start_time=now,
            end_time=now,
        )
        record = _read_last_record(tracker.telemetry_path)
        assert record["event"] == "llm_error"
        assert record["error"] == "Connection refused"
        assert record["phase"] == "ingest"

    def test_missing_metadata_uses_untagged(self, tmp_path: Path) -> None:
        tracker = TelemetryTracker(output_dir=tmp_path)
        now = _now()
        tracker.log_success_event(
            kwargs={"model": "ollama/test", "metadata": {}},
            response_obj=_make_mock_response(),
            start_time=now,
            end_time=now,
        )
        record = _read_last_record(tracker.telemetry_path)
        assert record["phase"] == "UNTAGGED"
        assert record["actor"] == "UNTAGGED"

    def test_none_metadata_uses_untagged(self, tmp_path: Path) -> None:
        tracker = TelemetryTracker(output_dir=tmp_path)
        now = _now()
        tracker.log_success_event(
            kwargs={"model": "ollama/test"},  # no metadata key at all
            response_obj=_make_mock_response(),
            start_time=now,
            end_time=now,
        )
        record = _read_last_record(tracker.telemetry_path)
        assert record["phase"] == "UNTAGGED"

    def test_multiple_calls_append_separate_lines(self, tmp_path: Path) -> None:
        tracker = TelemetryTracker(output_dir=tmp_path)
        now = _now()
        for _ in range(3):
            tracker.log_success_event(
                kwargs={"model": "ollama/test", "metadata": {"phase": "ingest", "actor": "vector_embed"}},
                response_obj=_make_mock_response(),
                start_time=now,
                end_time=now,
            )
        lines = tracker.telemetry_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        assert [json.loads(l)["call_index"] for l in lines] == [1, 2, 3]


class TestTelemetryTrackerWarnings:
    def test_untagged_call_logs_warning(self, tmp_path: Path, caplog) -> None:
        tracker = TelemetryTracker(output_dir=tmp_path)
        now = _now()
        with caplog.at_level(logging.WARNING, logger="src.telemetry.tracker"):
            tracker.log_success_event(
                kwargs={"model": "ollama/test", "metadata": {}},
                response_obj=_make_mock_response(),
                start_time=now,
                end_time=now,
            )
        assert "missing telemetry tags" in caplog.text

    def test_tagged_call_does_not_warn(self, tmp_path: Path, caplog) -> None:
        tracker = TelemetryTracker(output_dir=tmp_path)
        now = _now()
        with caplog.at_level(logging.WARNING, logger="src.telemetry.tracker"):
            tracker.log_success_event(
                kwargs={
                    "model": "ollama/test",
                    "metadata": {"phase": "evaluation", "actor": "llm_as_judge"},
                },
                response_obj=_make_mock_response(),
                start_time=now,
                end_time=now,
            )
        assert "UNTAGGED" not in caplog.text


class TestRegisterTracker:
    def test_register_tracker_returns_tracker_instance(self, tmp_path: Path) -> None:
        tracker = register_tracker(output_dir=tmp_path)
        assert isinstance(tracker, TelemetryTracker)

    def test_register_tracker_sets_litellm_callbacks(self, tmp_path: Path) -> None:
        tracker = register_tracker(output_dir=tmp_path)
        assert tracker in litellm.callbacks

    def test_register_tracker_telemetry_path_in_output_dir(self, tmp_path: Path) -> None:
        tracker = register_tracker(output_dir=tmp_path)
        assert tracker.telemetry_path.parent == tmp_path
