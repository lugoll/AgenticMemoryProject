"""
Prepare HotpotQA (distractor split) for the benchmarking pipeline.

Downloads the validation split from HuggingFace, samples a stratified subset,
and writes data/raw/hotpotqa_distractor.json in the project's dataset format:

    {
        "documents": ["paragraph text", ...],          # deduplicated context passages
        "questions": [
            {
                "question": "...",
                "answer": "...",
                "type": "bridge" | "comparison",
                "level": "easy" | "medium" | "hard"
            },
            ...
        ]
    }

Usage:
    uv run python scripts/prepare_hotpotqa.py
    uv run python scripts/prepare_hotpotqa.py --samples 200 --output data/raw/hotpotqa_200.json

Why the validation split?
    The test split has no gold answers. Validation contains 7405 labelled samples
    and is the standard evaluation set for HotpotQA leaderboard submissions.

Why stratified sampling?
    The benchmark hypothesis (H1) tests whether GraphRAG outperforms Vector RAG on
    multi-hop (bridge) questions. Stratifying on type × level ensures both bridge
    and comparison questions appear in proportion regardless of sample size.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def _load_hotpotqa(split: str = "validation") -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "The 'datasets' package is required. Install it with:\n"
            "    uv sync --extra dev\n"
            "or: pip install datasets"
        )

    logger.info("Downloading hotpotqa/hotpot_qa (distractor, %s)…", split)
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split, trust_remote_code=True)
    return list(ds)


def _stratified_sample(samples: list[dict], n: int) -> list[dict]:
    """Sample n items proportionally across type × level strata."""
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in samples:
        key = (s.get("type", "unknown"), s.get("level", "unknown"))
        strata[key].append(s)

    total = len(samples)
    selected: list[dict] = []

    for key, group in strata.items():
        quota = max(1, round(n * len(group) / total))
        selected.extend(group[:quota])
        logger.info("  stratum %s: %d / %d → taking %d", key, len(group), total, min(quota, len(group)))

    # Trim or pad to exactly n (rounding may over/under-shoot by a few).
    if len(selected) > n:
        selected = selected[:n]
    elif len(selected) < n:
        remaining = [s for s in samples if s not in selected]
        selected.extend(remaining[: n - len(selected)])

    return selected


def _extract_documents(samples: list[dict]) -> list[str]:
    """Flatten and deduplicate all context paragraphs across samples."""
    seen: set[str] = set()
    docs: list[str] = []
    for sample in samples:
        # context is {"title": [...], "sentences": [[sentence, ...], ...]}
        context = sample.get("context", {})
        titles = context.get("title", [])
        sentences_list = context.get("sentences", [])
        for title, sentences in zip(titles, sentences_list):
            paragraph = f"{title}: " + " ".join(sentences)
            if paragraph not in seen:
                seen.add(paragraph)
                docs.append(paragraph)
    return docs


def _extract_questions(samples: list[dict]) -> list[dict]:
    return [
        {
            "question": s["question"],
            "answer": s["answer"],
            "type": s.get("type", "unknown"),
            "level": s.get("level", "unknown"),
        }
        for s in samples
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HotpotQA dataset subset")
    parser.add_argument("--samples", type=int, default=500, help="Number of questions to sample (default: 500)")
    parser.add_argument("--output", type=Path, default=Path("data/raw/hotpotqa_distractor.json"))
    parser.add_argument("--split", default="validation", choices=["validation", "train"])
    args = parser.parse_args()

    raw = _load_hotpotqa(split=args.split)
    logger.info("Loaded %d samples from %s split", len(raw), args.split)

    sampled = _stratified_sample(raw, args.samples)
    logger.info("Sampled %d questions (stratified by type × level)", len(sampled))

    documents = _extract_documents(sampled)
    questions = _extract_questions(sampled)
    logger.info("Extracted %d unique context paragraphs", len(documents))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset = {"documents": documents, "questions": questions}
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    logger.info("Written to %s", args.output)
    logger.info("  documents : %d", len(documents))
    logger.info("  questions : %d", len(questions))

    type_counts: dict[str, int] = defaultdict(int)
    level_counts: dict[str, int] = defaultdict(int)
    for q in questions:
        type_counts[q["type"]] += 1
        level_counts[q["level"]] += 1
    logger.info("  by type   : %s", dict(type_counts))
    logger.info("  by level  : %s", dict(level_counts))


if __name__ == "__main__":
    main()
