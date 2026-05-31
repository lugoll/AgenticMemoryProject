"""
In-process LiteLLM proxy lifecycle + telemetry-tag plumbing.

Why this exists
---------------
Microsoft GraphRAG ships only a stable CLI. When the CLI shells out to an LLM
it has no hook for forwarding our `phase` / `actor` / `variant_name` / `run_id`
telemetry tags. To capture those calls in the same JSONL stream the other
variants use, we stand up a LiteLLM proxy as a daemon thread inside the same
Python process; the GraphRAG subprocess hits it as its OpenAI-compatible
endpoint, and the existing TelemetryTracker callback fires for every call.

The phase/actor tags are stitched in by reading PROXY_CONTEXT from inside the
TelemetryTracker fallback (see src/telemetry/tracker.py). Because the proxy
runs in the same process as the wrapper, a module-level dict is enough — there
is no cross-process state to synchronise.

Concurrency note: GraphRAG subprocess calls are sequential per-variant
operation (ingest, then query), so the single-slot PROXY_CONTEXT cannot leak
across concurrent operations of different phases.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


# Module-global "current tag context" read by TelemetryTracker when a callback
# fires for a proxy-routed call (which won't carry metadata).
#   None         → no proxy session active (treat as UNTAGGED)
#   dict         → phase/actor/variant_name/run_id for the wrapping session
PROXY_CONTEXT: dict[str, Any] = {"current": None}

_started = False
_started_port: int | None = None
_lock = threading.Lock()


def _wait_for_ready(port: int, timeout_s: float = 60.0, interval_s: float = 0.5) -> None:
    """Poll the proxy's /health/liveliness endpoint until it returns 200.

    Raises RuntimeError if the deadline passes. We deliberately do not check
    /v1/models or issue a completion: those depend on backend reachability,
    while liveliness just confirms the proxy process itself is up.
    """
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/health/liveliness"
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urlopen(Request(url), timeout=2.0) as r:
                if r.status == 200:
                    return
        except URLError as e:
            last_err = e
        except Exception as e:  # noqa: BLE001 — any failure is "not ready yet"
            last_err = e
        time.sleep(interval_s)
    raise RuntimeError(
        f"LiteLLM proxy at {url} did not become ready within {timeout_s:.0f}s. "
        f"Last error: {last_err!r}"
    )


def _ensure_started(config_path: Path, port: int) -> None:
    """Start the LiteLLM proxy as a daemon thread once per process.

    If already started on a different port, that's a programmer error — the
    proxy is a singleton in this design (one variant per script run).
    """
    global _started, _started_port
    with _lock:
        if _started:
            if _started_port != port:
                raise RuntimeError(
                    f"LiteLLM proxy already running on port {_started_port}; "
                    f"refusing to start a second instance on {port}."
                )
            return

        # The custom embedding provider (local-bge) is registered via the
        # proxy YAML's litellm_settings.custom_provider_map; the proxy
        # imports it at startup. Registering it here in Python would be
        # overwritten by the YAML loader (see proxy_server.py:3328).

        from litellm.proxy.proxy_cli import run_server

        def _serve() -> None:
            # standalone_mode=False so click doesn't sys.exit on completion.
            run_server.main(
                args=[
                    "--config", str(config_path),
                    "--port", str(port),
                    "--host", "127.0.0.1",
                    "--num_workers", "1",
                ],
                standalone_mode=False,
            )

        t = threading.Thread(target=_serve, name="litellm-proxy", daemon=True)
        t.start()

        _wait_for_ready(port, timeout_s=60.0)
        _started = True
        _started_port = port


@contextmanager
def proxy_session(
    phase: str,
    actor: str,
    variant_name: str,
    run_id: str,
    config_path: Path,
    port: int,
):
    """Bracket a subprocess call so its proxy-routed LLM calls get tagged.

    Idempotently starts the proxy on first entry. Swaps PROXY_CONTEXT for the
    duration of the with-block, restoring the previous value on exit (so
    nested sessions, though not currently used, would compose).

    Yields the OpenAI-compatible base URL, e.g. ``http://127.0.0.1:4555/v1``.
    """
    _ensure_started(config_path, port)
    prev = PROXY_CONTEXT["current"]
    PROXY_CONTEXT["current"] = {
        "phase": phase,
        "actor": actor,
        "variant_name": variant_name,
        "run_id": run_id,
    }
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        PROXY_CONTEXT["current"] = prev
