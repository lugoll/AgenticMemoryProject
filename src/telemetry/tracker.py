from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import litellm
from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger(__name__)

PhaseTag = Literal["ingest", "retrieval_overhead", "agent_reasoning", "evaluation"]
ActorTag = Literal[
    "vector_embed",
    "graph_extract",
    "graph_cypher_gen",
    "langgraph_node",
    "llm_as_judge",
]


class TelemetryTracker(CustomLogger):
    """
    LiteLLM CustomLogger that intercepts all LLM calls and appends structured
    token-usage records to a JSONL file in output_dir.

    Each tracker instance creates its own session file named
    ``telemetry_<session_timestamp>.jsonl`` inside output_dir.  Records are
    appended one JSON object per line so the file can be streamed and queried
    without loading it fully into memory.

    Correlation:
        Every record carries ``run_id`` (passed via LiteLLM metadata) so
        telemetry lines can be joined with QA results on that field.

    Registration:
        tracker = register_tracker(output_dir=Path("evaluations"))

    Tag injection — callers pass tags via metadata:
        litellm.completion(
            model=...,
            messages=...,
            metadata={
                "phase": "agent_reasoning",
                "actor": "langgraph_node",
                "variant_name": "baseline_dummy",
                "run_id": "a1b2c3d4",
            }
        )
    """

    def __init__(
        self,
        log_level: str = "INFO",
        output_dir: Path = Path("evaluations"),
    ) -> None:
        super().__init__()
        self._log_level = log_level.upper()
        self._call_count: int = 0
        self._output_dir = output_dir
        session_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._telemetry_path: Path = output_dir / f"telemetry_{session_ts}.jsonl"

    @property
    def telemetry_path(self) -> Path:
        """Path of the JSONL file this tracker writes to."""
        return self._telemetry_path

    def _write(self, record: dict[str, Any]) -> None:
        """Append one JSON record to the telemetry file."""
        self._telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        with self._telemetry_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _extract_metadata(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Extract metadata from kwargs.

        LiteLLM 1.82+ stores caller-supplied metadata inside litellm_params
        rather than at the top level of the callback kwargs dict.
        Fall back to the top-level key for direct/unit-test invocations.
        """
        litellm_params: dict[str, Any] = kwargs.get("litellm_params") or {}
        return litellm_params.get("metadata") or kwargs.get("metadata") or {}

    def _record_success(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Shared recording logic invoked by both sync and async success hooks."""
        self._call_count += 1
        metadata: dict[str, Any] = self._extract_metadata(kwargs)

        phase: str = metadata.get("phase", "UNTAGGED")
        actor: str = metadata.get("actor", "UNTAGGED")
        variant_name: str = metadata.get("variant_name", "unknown")
        run_id: str = metadata.get("run_id", "unknown")

        usage = getattr(response_obj, "usage", None)
        prompt_tokens: int = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens: int = getattr(usage, "completion_tokens", 0) or 0
        total_tokens: int = getattr(usage, "total_tokens", 0) or (
            prompt_tokens + completion_tokens
        )

        model: str = kwargs.get("model", "unknown")
        latency_ms: float = (end_time - start_time).total_seconds() * 1000

        record = {
            "event": "llm_call",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "call_index": self._call_count,
            "run_id": run_id,
            "phase": phase,
            "actor": actor,
            "variant_name": variant_name,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_ms": round(latency_ms, 2),
        }

        self._write(record)

        if phase == "UNTAGGED" or actor == "UNTAGGED":
            logger.warning(
                "LLM call missing telemetry tags! model=%s — "
                "All calls must pass metadata={'phase': ..., 'actor': ...}",
                model,
            )

    def log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Called by LiteLLM after every successful sync LLM call."""
        self._record_success(kwargs, response_obj, start_time, end_time)

    async def async_log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Called by LiteLLM after every successful async LLM call."""
        self._record_success(kwargs, response_obj, start_time, end_time)

    def log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Called by LiteLLM after a failed LLM call."""
        metadata: dict[str, Any] = self._extract_metadata(kwargs)
        model: str = kwargs.get("model", "unknown")
        error = kwargs.get("exception", "unknown error")

        record = {
            "event": "llm_error",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": metadata.get("run_id", "unknown"),
            "phase": metadata.get("phase", "UNTAGGED"),
            "actor": metadata.get("actor", "UNTAGGED"),
            "model": model,
            "error": str(error),
        }
        self._write(record)

    @property
    def call_count(self) -> int:
        """Total successful LLM calls intercepted since this tracker was created."""
        return self._call_count


def register_tracker(
    log_level: str = "INFO",
    output_dir: Path = Path("evaluations"),
) -> TelemetryTracker:
    """
    Create and register a TelemetryTracker with LiteLLM globally.

    Call once at application startup (e.g., top of a pipeline script).
    LiteLLM's callback system is process-global. Each call creates a new
    session JSONL file in output_dir.

    Args:
        log_level:  Log level string (e.g. "INFO", "DEBUG").
        output_dir: Directory where the telemetry JSONL file is written.

    Returns:
        The registered TelemetryTracker instance (useful for inspecting
        call_count and telemetry_path in tests).
    """
    tracker = TelemetryTracker(log_level=log_level, output_dir=output_dir)
    litellm.callbacks = [tracker]
    return tracker
