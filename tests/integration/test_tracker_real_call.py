"""
Integration tests requiring a live LLM endpoint.

Run with: uv run pytest -m integration -v
Skip with: uv run pytest -m "not integration" -v
"""
from __future__ import annotations

import threading

import pytest


@pytest.mark.integration
def test_tracker_intercepts_real_litellm_call() -> None:
    """Verify TelemetryTracker captures tokens from a real LiteLLM call."""
    import litellm
    from src.telemetry.tracker import register_tracker

    tracker = register_tracker()
    assert tracker.call_count == 0

    # Signal set by a one-shot wrapper so we can wait for the background callback.
    callback_done = threading.Event()
    original_record = tracker._record_success

    def _record_and_signal(*args, **kwargs):  # type: ignore[no-untyped-def]
        original_record(*args, **kwargs)
        callback_done.set()

    tracker._record_success = _record_and_signal  # type: ignore[method-assign]

    litellm.completion(
        model="ollama/gpt-oss:20b",
        messages=[{"role": "user", "content": "Say hello."}],
        metadata={
            "phase": "agent_reasoning",
            "actor": "langgraph_node",
            "variant_name": "integration_test",
        },
    )

    # LiteLLM 1.82+ fires callbacks in a background thread after completion()
    # returns, so we must wait before asserting.
    callback_done.wait(timeout=10)

    assert tracker.call_count == 1
