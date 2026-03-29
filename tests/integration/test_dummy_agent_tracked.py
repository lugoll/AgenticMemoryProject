"""
Integration tests for DummyAgent with real LiteLLM calls + TelemetryTracker.

Requires a live Ollama endpoint serving gpt-oss:20b.
Run with:  uv run pytest -m integration -v
Skip with: uv run pytest -m "not integration" -v
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest
import yaml

from src.agent.agent_dummy import DummyAgent
from src.config.settings import ExperimentConfig, load_config
from src.telemetry.tracker import TelemetryTracker, register_tracker

_OLLAMA_CONFIG: dict = {
    "telemetry": {"log_level": "DEBUG", "output_dir": "evaluations/"},
    "default": {
        "agent_type": "dummy",
        "data_file": "baseline_dummy.json",
        "llm": {"model": "ollama/gpt-oss:20b", "temperature": 0.0, "max_tokens": 256},
        "embedding": {"model": "ollama/nomic-embed-text", "batch_size": 4},
        "retrieval": {"top_k": 3, "similarity_cutoff": 0.5},
        "ingestion": {"chunk_size": 256, "chunk_overlap": 32},
    },
    "variants": {"ollama_dummy": {"agent_type": "dummy"}},
}


@pytest.fixture()
def ollama_config(tmp_path: Path) -> ExperimentConfig:
    cfg_file = tmp_path / "ollama_config.yaml"
    cfg_file.write_text(yaml.dump(_OLLAMA_CONFIG))
    return load_config("ollama_dummy", config_path=cfg_file)


@pytest.fixture()
def tracked(ollama_config: ExperimentConfig) -> tuple[DummyAgent, TelemetryTracker, threading.Event]:
    """
    Return a DummyAgent wired to a fresh TelemetryTracker.

    The tracker's _record_success is wrapped to set an Event when the first
    callback fires. LiteLLM 1.82+ fires callbacks in a background thread, so
    tests must call callback_done.wait() before asserting call_count.
    """
    tracker = register_tracker(log_level=ollama_config.telemetry.log_level)
    callback_done = threading.Event()
    original = tracker._record_success

    def _record_and_signal(*args, **kwargs):  # type: ignore[no-untyped-def]
        original(*args, **kwargs)
        callback_done.set()

    tracker._record_success = _record_and_signal  # type: ignore[method-assign]

    agent = DummyAgent(config=ollama_config)
    return agent, tracker, callback_done


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_tracker_captures_agent_reason_node_call(
    tracked: tuple[DummyAgent, TelemetryTracker, threading.Event],
) -> None:
    """
    A single agent.run() produces exactly one LiteLLM call.
    The TelemetryTracker must intercept it and increment call_count to 1.
    """
    agent, tracker, callback_done = tracked
    assert tracker.call_count == 0

    agent._memory.ingest_documents(["The Eiffel Tower is located in Paris, France."])
    response = agent.run("Where is the Eiffel Tower?")

    callback_done.wait(timeout=30)

    assert tracker.call_count == 1
    assert response["steps_taken"] == ["retrieve", "reason"]
    assert response["backend_name"] == "dummy"
    assert response["answer"]


@pytest.mark.integration
def test_tracker_records_agent_reasoning_phase_tag(
    tracked: tuple[DummyAgent, TelemetryTracker, threading.Event],
) -> None:
    """
    The telemetry record emitted for the reason node must carry
    phase=agent_reasoning and actor=langgraph_node.
    """
    agent, tracker, callback_done = tracked

    # Capture the record that _record_success receives
    captured: dict = {}
    original = tracker._record_success

    def _capture(*args, **kwargs):  # type: ignore[no-untyped-def]
        # args: (kwargs_dict, response_obj, start_time, end_time)
        captured["litellm_kwargs"] = args[0]
        original(*args, **kwargs)
        callback_done.set()

    tracker._record_success = _capture  # type: ignore[method-assign]

    agent.run("test question")
    callback_done.wait(timeout=30)

    from src.telemetry.tracker import TelemetryTracker
    metadata = tracker._extract_metadata(captured["litellm_kwargs"])
    assert metadata.get("phase") == "agent_reasoning"
    assert metadata.get("actor") == "langgraph_node"


@pytest.mark.integration
def test_tracker_captures_two_calls_for_two_runs(
    tracked: tuple[DummyAgent, TelemetryTracker, threading.Event],
) -> None:
    """Each agent.run() must produce exactly one tracked LLM call."""
    agent, tracker, callback_done = tracked

    agent.run("first question")
    callback_done.wait(timeout=30)
    assert tracker.call_count == 1

    # Reset event for the second call
    callback_done.clear()
    agent.run("second question")
    callback_done.wait(timeout=30)
    assert tracker.call_count == 2


@pytest.mark.integration
def test_agent_with_seeded_memory_still_answers(
    tracked: tuple[DummyAgent, TelemetryTracker, threading.Event],
) -> None:
    """Retrieved context reaches the LLM and the response is non-empty."""
    agent, tracker, callback_done = tracked

    agent._memory.ingest_documents([
        "Mount Everest is the highest mountain on Earth.",
        "It sits on the Nepal-Tibet border.",
    ])
    response = agent.run("What is the highest mountain?")
    callback_done.wait(timeout=30)

    assert tracker.call_count == 1
    assert len(response["context_used"]) > 0
    assert response["answer"]
