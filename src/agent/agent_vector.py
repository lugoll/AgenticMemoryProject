from __future__ import annotations

from src.config.settings import ExperimentConfig
from src.memory.model_vector import VectorMemory

from .base import BaseAgent


class VectorAgent(BaseAgent):
    """
    Agent backed by VectorMemory for semantic retrieval via ChromaDB.

    Inherits the full LangGraph graph and LiteLLM reasoning from BaseAgent.
    The memory backend uses ChromaDB (HTTP server) with BAAI/bge-base-en-v1.5
    embeddings computed locally — no LLM calls at retrieval time.

    config.memory_path is used as the ChromaDB collection name so that
    different experiment variants can coexist on the same server.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(
            memory=VectorMemory(config=config),
            config=config,
        )

    def get_agent_name(self) -> str:
        return "vector"
