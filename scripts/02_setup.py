"""
Phase 2 — Store aufbauen für eine Variante.

Aufruf:
    uv run python scripts/02_setup.py --variant bm25   --data data/hotpotqa.json
    uv run python scripts/02_setup.py --variant vector  --data data/hotpotqa.json
    uv run python scripts/02_setup.py --variant graph   --data data/hotpotqa.json

Ausgabe: evaluations/<variant>_<ts>_setup.json
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src.config.cfg import load_config
from src.telemetry.tracker import register_tracker


def _read_telemetry(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG-Store aufbauen")
    parser.add_argument("--variant", required=True, choices=["bm25", "vector", "graph"])
    parser.add_argument("--data", type=Path, default=Path("data/hotpotqa.json"))
    args = parser.parse_args()

    cfg = load_config()
    output_dir = Path(cfg.telemetry.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker = register_tracker(output_dir=output_dir, variant_name=args.variant)

    with args.data.open(encoding="utf-8") as f:
        data = json.load(f)
    documents: list[str] = data["documents"]
    print(f"Variante: {args.variant}  |  Dokumente: {len(documents)}")

    if args.variant == "bm25":
        from src.memory.model_bm25 import BM25Memory
        memory = BM25Memory(
            top_k=cfg.retrieval.top_k,
            storage_path=Path(cfg.stores.bm25),
        )
    elif args.variant == "vector":
        from src.memory.model_vector import VectorMemory
        memory = VectorMemory(config=cfg)
    elif args.variant == "graph":
        from src.memory.model_graph import GraphMemory
        memory = GraphMemory(config=cfg)
    else:
        raise ValueError(f"Unbekannte Variante: {args.variant}")

    t0 = time.perf_counter()
    memory.reset()
    memory.ingest_documents(documents)
    elapsed = time.perf_counter() - t0

    print(f"Ingest abgeschlossen in {elapsed:.1f}s")

    tel = _read_telemetry(tracker.telemetry_path)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    setup_stats = {
        "variant":                  args.variant,
        "n_documents":              len(documents),
        "ingest_time_s":            round(elapsed, 2),
        "ingest_tokens_prompt":     sum(r.get("prompt_tokens", 0)     for r in tel),
        "ingest_tokens_completion": sum(r.get("completion_tokens", 0) for r in tel),
        "ingest_tokens_total":      sum(r.get("total_tokens", 0)      for r in tel),
        "llm_calls":                len(tel),
        "created_at":               datetime.now(timezone.utc).isoformat(),
    }

    out_path = output_dir / f"{args.variant}_{ts}_setup.json"
    out_path.write_text(json.dumps(setup_stats, indent=2), encoding="utf-8")

    print(f"Setup-Stats -> {out_path}")
    print(f"  Zeit      : {elapsed:.1f}s")
    print(f"  LLM-Calls : {setup_stats['llm_calls']}")
    print(f"  Tokens    : {setup_stats['ingest_tokens_total']}")


if __name__ == "__main__":
    main()
