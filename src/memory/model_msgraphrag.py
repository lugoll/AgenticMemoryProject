"""
MSGraphRAGMemory — wraps the Microsoft ``graphrag`` CLI behind the project's
BaseMemory interface.

Why subprocess + in-process proxy?
----------------------------------
GraphRAG's stable surface is its CLI; its Python API is undocumented and has
shifted across minor versions. To still capture every LLM call it makes in
the project's JSONL telemetry, we:

  1. Stand up a LiteLLM proxy as a daemon thread in this process
     (see src/utils/litellm_proxy.py). It routes ``ollama/llama3.1:8b`` to
     the local Ollama daemon and serves a ``local-bge`` embedding model via
     sentence-transformers, mirroring the 'vector' variant.
  2. Generate ``settings.yaml`` so GraphRAG points every model call at the
     proxy's OpenAI-compatible endpoint.
  3. Bracket each ``subprocess.run(['graphrag', ...])`` call with a
     ``proxy_session(phase, actor, ...)`` so the TelemetryTracker (already
     registered as the global litellm callback) tags incoming proxy calls
     with the right phase/actor.

End-to-end semantics
--------------------
``is_end_to_end = True`` because GraphRAG synthesises a final answer inside
``search()``. 03_run.py uses that answer directly and skips its own
answer_question() LLM call. The bypass is documented in
[BaseMemory.is_end_to_end](src/memory/base.py).
"""
from __future__ import annotations

import contextvars
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from jinja2 import Template

from src.config.cfg import Config
from src.utils.litellm_proxy import proxy_session

from .base import BaseMemory

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "_graphrag_templates"
_SETTINGS_TEMPLATE = _TEMPLATE_DIR / "settings.yaml.j2"
_PROXY_CONFIG = _TEMPLATE_DIR / "proxy_config.yaml"

# Set by 03_run.py once per question so search() can tag its proxy session
# with the correct run_id. Default value lets unit tests and 02_setup.py path
# work without explicit setup.
CURRENT_RUN_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "msgraphrag_run_id", default="unknown"
)


class MSGraphRAGMemory(BaseMemory):
    """Microsoft GraphRAG (community-based hierarchical) memory backend."""

    is_end_to_end = True

    def __init__(self, config: Config) -> None:
        self._config = config
        self._cfg = config.stores.msgraphrag
        self.root = Path(self._cfg.root_dir)
        self._input_dir = self.root / "input"
        self._output_dir = self.root / "output"
        self._settings_path = self.root / "settings.yaml"

    # ---- BaseMemory interface ----

    def ingest_documents(self, documents: list[str]) -> None:
        """Persist documents to disk and run ``graphrag index``.

        Steps:
          1. Reset → fresh root_dir.
          2. ``graphrag init`` → generates prompts/ and default settings.yaml.
          3. Overwrite settings.yaml with our template (points at proxy).
          4. Write each document as input/doc_NNNN.txt.
          5. ``graphrag index`` inside a proxy_session tagged ingest/graph_extract.
        """
        self._prepare_root()
        self._render_settings()
        self._write_inputs(documents)

        with proxy_session(
            phase="ingest",
            actor="graph_extract",
            variant_name="msgraphrag",
            run_id="ingest",
            config_path=_PROXY_CONFIG,
            port=self._cfg.proxy_port,
        ):
            self._run_cli(["index", "--root", str(self.root)])

        logger.info(
            "MSGraphRAGMemory: ingested %d documents into %s", len(documents), self.root
        )

    def search(self, query: str) -> list[str]:
        """Run ``graphrag query`` and return the synthesised answer as a one-element list.

        Because is_end_to_end=True, the runner treats the returned element as
        the final answer rather than as retrieval context.
        """
        if not self._settings_path.exists():
            raise RuntimeError(
                f"MS GraphRAG store not initialised at {self.root}. "
                "Run scripts/02_setup.py --variant msgraphrag first."
            )

        run_id = CURRENT_RUN_ID.get()
        with proxy_session(
            phase="retrieval_overhead",
            actor="graph_cypher_gen",
            variant_name="msgraphrag",
            run_id=run_id,
            config_path=_PROXY_CONFIG,
            port=self._cfg.proxy_port,
        ):
            result = self._run_cli(
                [
                    "query",
                    "--root", str(self.root),
                    "--method", self._cfg.query_method,
                    "--community-level", str(self._cfg.community_level),
                    "--response-type", self._cfg.response_type,
                    "--query", query,
                ],
                capture_output=True,
            )

        answer = _parse_answer(result.stdout if result else "")
        return [answer] if answer else []

    def update_fact(self, fact: str) -> None:
        raise NotImplementedError(
            "MS GraphRAG builds its index once and cannot ingest single facts at "
            "test-time. Re-run scripts/02_setup.py to rebuild the index."
        )

    def reset(self) -> None:
        """Delete the entire root_dir; next ingest will recreate it."""
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def get_backend_name(self) -> str:
        return "msgraphrag"

    # ---- internals ----

    def _prepare_root(self) -> None:
        """Wipe the root dir and run ``graphrag init`` so prompts/ is populated."""
        self.reset()
        # graphrag init does not need the proxy; just sets up files locally.
        subprocess.run(
            [sys.executable, "-m", "graphrag", "init", "--root", str(self.root)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def _render_settings(self) -> None:
        template = Template(_SETTINGS_TEMPLATE.read_text(encoding="utf-8"))
        rendered = template.render(
            api_base=f"http://127.0.0.1:{self._cfg.proxy_port}/v1",
            chat_model=self._cfg.chat_model,
            embed_model=self._cfg.embed_model,
            concurrent_requests=self._cfg.concurrent_requests,
            chunk_size=self._config.ingestion.chunk_size,
            chunk_overlap=self._config.ingestion.chunk_overlap,
        )
        self._settings_path.write_text(rendered, encoding="utf-8")

    def _write_inputs(self, documents: list[str]) -> None:
        self._input_dir.mkdir(parents=True, exist_ok=True)
        # Reset input/ in case a previous partial run left files behind.
        for f in self._input_dir.glob("doc_*.txt"):
            f.unlink()
        for i, doc in enumerate(documents):
            stripped = doc.strip()
            if not stripped:
                continue
            (self._input_dir / f"doc_{i:05d}.txt").write_text(stripped, encoding="utf-8")

    def _run_cli(self, args: list[str], capture_output: bool = False) -> subprocess.CompletedProcess[str] | None:
        """Run ``python -m graphrag <args>``.

        Env vars:
          - GRAPHRAG_API_KEY: any string; the proxy's master_key. graphrag
            includes it in Authorization headers; LiteLLM accepts it.

        Using ``-m graphrag`` over ``graphrag`` to guarantee we hit the
        interpreter from this venv even if the entry-point script is shadowed.
        """
        env = os.environ.copy()
        env.setdefault("GRAPHRAG_API_KEY", "sk-graphrag-local")
        cmd = [sys.executable, "-m", "graphrag", *args]
        logger.debug("MSGraphRAGMemory: running %s", cmd)
        if capture_output:
            return subprocess.run(
                cmd, check=True, capture_output=True, text=True, env=env
            )
        subprocess.run(cmd, check=True, env=env)
        return None


def _parse_answer(stdout: str) -> str:
    """Extract the answer line(s) from ``graphrag query`` stdout.

    The CLI does ``print(response)`` once with the synthesised answer. It may
    also emit progress markers from rich/typer formatting. Strategy: strip
    ANSI, drop empty/leading-banner lines, and concatenate what remains.
    Robust enough for HotpotQA-style short answers; revisit if responses get
    multi-paragraph.
    """
    if not stdout:
        return ""
    # Strip common ANSI color escape sequences without pulling in another dep.
    import re

    no_ansi = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", stdout)
    lines = [ln.rstrip() for ln in no_ansi.splitlines() if ln.strip()]
    # graphrag prints progress bars + the final response. Heuristic: the
    # response is the last contiguous block of non-empty lines.
    return "\n".join(lines).strip()
