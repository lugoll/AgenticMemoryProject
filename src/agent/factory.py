from __future__ import annotations

from src.config.settings import ExperimentConfig

from .base import BaseAgent


def build_agent(config: ExperimentConfig) -> BaseAgent:
    """
    Instantiate the correct agent for the given experiment variant.

    Dispatches on config.agent_type so pipeline code never imports concrete
    agent classes directly — only this factory needs updating when a new
    backend is added.

    Args:
        config: Fully resolved experiment config for a named variant.

    Returns:
        A ready-to-use BaseAgent with the appropriate memory backend wired in.

    Raises:
        ValueError: If config.agent_type is not recognised.
    """
    if config.agent_type == "dummy":
        from .agent_dummy import DummyAgent
        return DummyAgent(config=config)

    if config.agent_type == "graph":
        from .agent_graph import GraphAgent
        return GraphAgent(config=config)

    if config.agent_type == "bm25":
        from .agent_bm25 import BM25Agent
        return BM25Agent(config=config)

    if config.agent_type == "vector":
        from .agent_vector import VectorAgent
        return VectorAgent(config=config)
    
    if config.agent_type == "lightrag":
        from .agent_lightrag import LightRAGAgent
        return LightRAGAgent(config=config)

    raise ValueError(
        f"Unknown agent_type '{config.agent_type}'. "
        f"Add a branch to src/agent/factory.py to support it."
    )
