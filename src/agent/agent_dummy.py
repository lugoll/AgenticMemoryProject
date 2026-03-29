from __future__ import annotations

from pathlib import Path

from src.config.settings import ExperimentConfig
from src.memory.model_dummy import DummyMemory

from .base import BaseAgent


class DummyAgent(BaseAgent):
    """
    Agent backed by DummyMemory for testing pipeline and graph logic.

    Inherits the full LangGraph graph and LiteLLM reasoning from BaseAgent.
    The only difference from a production agent is the memory backend:
    DummyMemory uses in-memory substring matching instead of real embeddings
    or a knowledge graph, so no vector/graph infrastructure is required.

    When config.memory_path is set the memory store is persisted to that file,
    enabling the ingestion step and the QA step to share state across runs.

    Use this agent to:
    - Test agent graph wiring and telemetry tagging (mock litellm.completion)
    - Run end-to-end integration tests against a real LLM without a vector store
    - Drive the full pipeline with file-backed DummyMemory
    """

    def __init__(self, config: ExperimentConfig) -> None:
        storage_path = Path(config.memory_path) if config.memory_path else None
        super().__init__(
            memory=DummyMemory(top_k=config.retrieval.top_k, storage_path=storage_path),
            config=config,
        )

    def get_agent_name(self) -> str:
        return "dummy"
