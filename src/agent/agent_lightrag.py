from __future__ import annotations

from src.config.settings import ExperimentConfig
from src.memory.model_lightrag import LightRAGMemory

from .base import BaseAgent


class LightRAGAgent(BaseAgent):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(memory=LightRAGMemory(config=config), config=config)
