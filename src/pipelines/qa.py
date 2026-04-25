"""
Pipeline step 02 — Agentic Question Answering.

Loads questions from a dataset file, runs each through the agent, and saves
the responses to the evaluations directory as a JSONL file.

Usage (via main.py):
    python main.py --variant baseline_dummy --qa
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.agent.factory import build_agent
from src.config.settings import ExperimentConfig

logger = logging.getLogger(__name__)


def run_qa(
    config: ExperimentConfig,
    data_dir: Path = Path("data/raw"),
    output_dir: Path = Path("evaluations"),
) -> Path:
    """
    Answer all questions in config.data_file and save results to output_dir.

    Each question produces one JSONL line containing the full AgentResponse
    plus the expected answer (for later evaluation) and metadata.

    Args:
        config:     Fully resolved experiment config for the target variant.
        data_dir:   Base directory containing raw dataset files.
        output_dir: Directory where the JSONL results file is written.

    Returns:
        Path to the written JSONL results file.
    """
    dataset_path = data_dir / config.data_file
    logger.info("[qa] variant=%s  dataset=%s", config.variant_name, dataset_path)

    with dataset_path.open("r", encoding="utf-8") as f:
        dataset: dict = json.load(f)

    questions: list[dict] = dataset.get("questions", [])
    if not questions:
        logger.warning("[qa] No questions found in %s — nothing to evaluate", dataset_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{config.variant_name}_results.jsonl"

    agent = build_agent(config)
    logger.info("[qa] Running %d questions with agent_type=%s…", len(questions), config.agent_type)

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"{config.variant_name}_{timestamp}_results.jsonl"

    with output_path.open("w", encoding="utf-8") as out:
        for entry in questions:
            question: str = entry.get("question", "")
            expected: str = entry.get("answer", entry.get("expected", ""))

            logger.debug("[qa] Q: %s", question)
            response = agent.run(question)

            record = {
                "run_id": response["run_id"],
                "variant_name": config.variant_name,
                "agent_type": config.agent_type,
                "question": response["question"],
                "expected": expected,
                "answer": response["answer"],
                "context_used": response["context_used"],
                "steps_taken": response["steps_taken"],
                "backend_name": response["backend_name"],
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info("[qa] answered: %s → %s", question[:60], response["answer"][:80])

    logger.info("[qa] Results written to %s", output_path)
    return output_path
