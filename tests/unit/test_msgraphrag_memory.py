"""Unit tests for MSGraphRAGMemory.

Real ingest/query would shell out to ``graphrag`` and require a running
LiteLLM proxy + Ollama, so we mock subprocess.run for tests that exercise
the wrapper logic. Real end-to-end behaviour is covered by the integration
verification steps in the implementation plan.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.config.cfg import load_config
from src.memory.base import BaseMemory
from src.memory.model_msgraphrag import MSGraphRAGMemory, _parse_answer


@pytest.fixture
def msgraphrag_cfg(tmp_path, monkeypatch):
    """Load real config but point the msgraphrag store at a tmp dir."""
    cfg = load_config()
    cfg.stores.msgraphrag.root_dir = str(tmp_path / "store")
    return cfg


class TestBaseMemoryContract:
    def test_msgraphrag_is_subclass_of_base(self) -> None:
        assert issubclass(MSGraphRAGMemory, BaseMemory)

    def test_msgraphrag_is_end_to_end(self) -> None:
        # Critical: 03_run.py uses this flag to decide whether to skip
        # answer_question(). If False, the agent will re-answer GraphRAG's
        # synthesised answer and double-count tokens.
        assert MSGraphRAGMemory.is_end_to_end is True


class TestMSGraphRAGMemoryLifecycle:
    def test_init_does_not_touch_disk(self, msgraphrag_cfg) -> None:
        mem = MSGraphRAGMemory(msgraphrag_cfg)
        assert not mem.root.exists()
        assert mem.get_backend_name() == "msgraphrag"

    def test_reset_creates_empty_root(self, msgraphrag_cfg) -> None:
        mem = MSGraphRAGMemory(msgraphrag_cfg)
        mem.reset()
        assert mem.root.exists()
        assert list(mem.root.iterdir()) == []

    def test_reset_wipes_existing_content(self, msgraphrag_cfg) -> None:
        mem = MSGraphRAGMemory(msgraphrag_cfg)
        mem.root.mkdir(parents=True, exist_ok=True)
        (mem.root / "stale.txt").write_text("leftover", encoding="utf-8")
        mem.reset()
        assert not (mem.root / "stale.txt").exists()
        assert mem.root.exists()

    def test_update_fact_raises(self, msgraphrag_cfg) -> None:
        mem = MSGraphRAGMemory(msgraphrag_cfg)
        with pytest.raises(NotImplementedError):
            mem.update_fact("any fact")

    def test_search_without_ingest_raises(self, msgraphrag_cfg) -> None:
        mem = MSGraphRAGMemory(msgraphrag_cfg)
        with pytest.raises(RuntimeError, match="not initialised"):
            mem.search("anything")


class TestMSGraphRAGMemoryIngest:
    def test_ingest_writes_input_files_and_settings(self, msgraphrag_cfg) -> None:
        mem = MSGraphRAGMemory(msgraphrag_cfg)

        # Mock both `graphrag init` (the prepare step) and `graphrag index`.
        with patch("src.memory.model_msgraphrag.subprocess.run") as mock_run, \
             patch("src.utils.litellm_proxy._ensure_started") as mock_start:
            mock_run.return_value.stdout = ""
            mock_run.return_value.stderr = ""
            mock_start.return_value = None  # don't actually start a proxy

            mem.ingest_documents(["First doc", "Second doc", "   "])

        # Settings.yaml exists and references the proxy URL.
        settings_text = mem._settings_path.read_text(encoding="utf-8")
        assert "127.0.0.1" in settings_text
        assert str(msgraphrag_cfg.stores.msgraphrag.proxy_port) in settings_text
        assert msgraphrag_cfg.stores.msgraphrag.chat_model in settings_text
        assert msgraphrag_cfg.stores.msgraphrag.embed_model in settings_text

        # Two input docs written; whitespace-only doc dropped.
        input_files = sorted(p.name for p in (mem.root / "input").glob("doc_*.txt"))
        assert input_files == ["doc_00000.txt", "doc_00001.txt"]


class TestParseAnswer:
    def test_strips_ansi_codes(self) -> None:
        # graphrag's typer/rich output sometimes injects ANSI colour codes.
        raw = "\x1b[32mGreen text\x1b[0m The answer is X."
        assert _parse_answer(raw) == "Green text The answer is X."

    def test_drops_blank_lines(self) -> None:
        raw = "\n\nFinal answer line\n\n"
        assert _parse_answer(raw) == "Final answer line"

    def test_empty_input_returns_empty(self) -> None:
        assert _parse_answer("") == ""

    def test_joins_multi_line_response(self) -> None:
        raw = "Line one of answer.\nLine two of answer."
        assert _parse_answer(raw) == "Line one of answer.\nLine two of answer."
