"""
Phase 1 — HotpotQA laden und in das Benchmark-Format konvertieren.

Aufruf:
    uv run python scripts/01_load_hotpotqa.py --n 300 --out data/hotpotqa.json
    uv run python scripts/01_load_hotpotqa.py --n 50  --out data/hotpotqa.json  # Entwicklung

Ausgabe: data/hotpotqa.json mit meta-Block, documents und questions.
Sampling: 50% bridge + 50% comparison (stratifiziert, reproduzierbar via --seed).
Split:    train (90k Fragen mit Gold-Antworten, distractor-Setting).
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path


def load_and_sample(n: int, seed: int) -> tuple[list[dict], list[str], list[dict]]:
    import os
    from pathlib import Path

    # Load HF_TOKEN from .env if present (avoids rate-limit warnings)
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                os.environ.setdefault("HF_TOKEN", line.split("=", 1)[1].strip())

    # dill 0.3.8 + Python 3.14: _batch_setitems() signature changed → patch the legacy
    # cache check that triggers it (pure migration helper, safe to skip on fresh installs).
    from datasets.builder import DatasetBuilder
    DatasetBuilder._use_legacy_cache_dir_if_possible = lambda self, _dm: None

    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")

    bridge     = [x for x in ds if x["type"] == "bridge"]
    comparison = [x for x in ds if x["type"] == "comparison"]

    rng  = random.Random(seed)
    half = n // 2
    selected = rng.sample(bridge, half) + rng.sample(comparison, n - half)

    docs: list[str] = []
    seen: set[str]  = set()
    for entry in selected:
        context = entry["context"]
        # HotpotQA context: {"title": [...], "sentences": [[s, ...], ...]}
        for title, sentences in zip(context["title"], context["sentences"]):
            paragraph = title + ". " + " ".join(sentences)
            if paragraph not in seen:
                seen.add(paragraph)
                docs.append(paragraph)

    questions = [
        {
            "id":       entry["id"],
            "question": entry["question"],
            "answer":   entry["answer"],
            "type":     entry["type"],
            "level":    entry["level"],
        }
        for entry in selected
    ]

    return selected, docs, questions


def main() -> None:
    parser = argparse.ArgumentParser(description="HotpotQA laden & konvertieren")
    parser.add_argument("--n",    type=int,  default=300, help="Anzahl Fragen (default: 300)")
    parser.add_argument("--out",  type=Path, default=Path("data/hotpotqa.json"))
    parser.add_argument("--seed", type=int,  default=42)
    args = parser.parse_args()

    print(f"Lade HotpotQA (train, distractor) und sample {args.n} Fragen …")
    selected, docs, questions = load_and_sample(args.n, args.seed)

    bridge_count     = sum(1 for q in questions if q["type"] == "bridge")
    comparison_count = sum(1 for q in questions if q["type"] == "comparison")

    output = {
        "meta": {
            "source":         "hotpotqa/hotpot_qa distractor/train",
            "n_questions":    len(questions),
            "question_types": {"bridge": bridge_count, "comparison": comparison_count},
            "n_documents":    len(docs),
            "seed":           args.seed,
            "created_at":     datetime.now(timezone.utc).isoformat(),
        },
        "documents": docs,
        "questions":  questions,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Geschrieben: {args.out}")
    print(f"  Fragen     : {len(questions)}  (bridge: {bridge_count}, comparison: {comparison_count})")
    print(f"  Paragraphen: {len(docs)}")
    print(f"  Beispiel   : {docs[0][:120]!r}")


if __name__ == "__main__":
    main()
