from __future__ import annotations

from src.config.settings import ExperimentConfig
from src.memory.model_vecgraph import VecGraphMemory

from .base import BaseAgent


class VecGraphAgent(BaseAgent):
    """
    Agent backed by VecGraphMemory (append-only entity/relation ledger + shared FAISS index).

    At ingest time the LLM extracts atomic entity facts and semantically-rich
    relation descriptions, both stored as flat-file ledgers and embedded into a
    shared FAISS index. At retrieval time FAISS hits on facts or relations
    surface the parent entity's full fact list and its 1-hop graph neighbourhood,
    keeping LLM token cost at test time to a single embedding call per query.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(memory=VecGraphMemory(config=config), config=config)

    def get_agent_name(self) -> str:
        return "vecgraph"
