from __future__ import annotations

from pathlib import Path

from src.config.settings import ExperimentConfig
from src.memory.model_bm25 import BM25Memory

from .base import BaseAgent


class BM25Agent(BaseAgent):
    """
    Agent backed by BM25Memory for keyword-based retrieval via SQLite FTS5.

    Inherits the full LangGraph graph and LiteLLM reasoning from BaseAgent.
    The memory backend uses SQLite FTS5 with BM25 ranking — no embeddings or
    LLM calls are required for ingestion or retrieval.

    When config.memory_path is set the SQLite database is persisted to that
    file, enabling the ingestion step and the QA step to share state.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        storage_path = Path(config.memory_path) if config.memory_path else None
        super().__init__(
            memory=BM25Memory(top_k=config.retrieval.top_k, storage_path=storage_path),
            config=config,
        )

    def get_agent_name(self) -> str:
        return "bm25"
