"""
Pipeline step 01 — Offline ingestion.

Reads raw documents from a dataset file and ingests them into the agent's
memory backend. Memory is reset before ingestion to ensure a clean baseline.

Usage (via main.py):
    python main.py --variant baseline_dummy --ingest
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from src.agent.factory import build_agent
from src.config.settings import ExperimentConfig

logger = logging.getLogger(__name__)


def run_ingest(config: ExperimentConfig, data_dir: Path = Path("data/raw")) -> None:
    """
    Ingest documents from config.data_file into the variant's memory backend.

    Steps:
      1. Load the dataset JSON from data_dir / config.data_file.
      2. Build the agent (and its memory backend) for this variant.
      3. Reset memory to guarantee a clean slate.
      4. Ingest all documents from the dataset.

    Args:
        config:   Fully resolved experiment config for the target variant.
        data_dir: Base directory containing raw dataset files.
    """
    dataset_path = data_dir / config.data_file
    logger.info("[ingest] variant=%s  dataset=%s", config.variant_name, dataset_path)

    with dataset_path.open("r", encoding="utf-8") as f:
        dataset: dict = json.load(f)

    documents: list[str] = dataset.get("documents", [])
    if not documents:
        logger.warning("[ingest] No documents found in %s — nothing to ingest", dataset_path)
        return

    agent = build_agent(config)
    agent._memory.reset()
    logger.info("[ingest] Memory reset. Ingesting %d documents…", len(documents))

    agent._memory.ingest_documents(documents)
    logger.info("[ingest] Done. Memory store size = %d", agent._memory.store_size)  # type: ignore[attr-defined]
