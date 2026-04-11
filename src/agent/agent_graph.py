from __future__ import annotations

from src.config.settings import ExperimentConfig
from src.memory.model_graph import GraphMemory

from .base import BaseAgent


class GraphAgent(BaseAgent):
    """
    Agent backed by GraphMemory for knowledge-graph-based retrieval.

    Inherits the full LangGraph graph and LiteLLM reasoning node from BaseAgent
    unchanged. The only difference from DummyAgent is the memory backend:
    GraphMemory builds a (subject, predicate, object) knowledge graph at ingest
    time using LLM-based extraction, then answers queries via BFS traversal
    with zero LLM cost at retrieval time.

    This asymmetry — expensive ingestion, cheap retrieval — is the core tradeoff
    the benchmark is designed to measure against Vector RAG's cheap ingestion
    but token-heavy retrieval.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(memory=GraphMemory(config=config), config=config)

    def get_agent_name(self) -> str:
        return "graph"
