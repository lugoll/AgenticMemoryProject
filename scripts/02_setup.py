"""
Phase 2 — Unified Store aufbauen (ein Lauf für alle Retrieval-Varianten).

Aufruf:
    uv run python scripts/02_setup.py --data data/hotpotqa.json
    uv run python scripts/02_setup.py --data data/hotpotqa.json --n 50
    uv run python scripts/02_setup.py --data data/hotpotqa.json --resume

--n begrenzt die Anzahl der ingestierten Dokumente (nützlich für schnelle Tests).
Ohne --n werden alle Dokumente in der Datei verarbeitet.
--resume macht ab dem letzten Checkpoint (in Neo4j persistiert) weiter.

Der Ingest schreibt Chunks (mit Embedding, Volltext- und Vektor-Index) sowie
den extrahierten Knowledge Graph in dieselbe Neo4j-Datenbank. Danach können
alle Varianten (bm25, vector, graph, vectorgraph) ohne weiteren Ingest laufen.

Ausgabe: evaluations/unified_<ts>_setup.json
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src.config.cfg import load_config
from src.telemetry.tracker import register_tracker
from src.utils.docker_utils import ensure_containers_running, stop_containers, get_required_containers


def _read_telemetry(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified RAG-Store aufbauen")
    parser.add_argument("--data", type=Path, default=Path("data/hotpotqa.json"))
    parser.add_argument("--n", type=int, default=None,
                        help="Maximale Anzahl Dokumente (default: alle). "
                             "Nützlich für schnelle Tests, z.B. --n 50.")
    parser.add_argument("--resume", action="store_true",
                        help="Weitermachen ab dem letzten Checkpoint")
    args = parser.parse_args()

    cfg = load_config()
    output_dir = Path(cfg.telemetry.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    containers = get_required_containers("unified")
    if containers:
        print("Starting containers for unified ingest...")
        ensure_containers_running(containers)

    try:
        tracker = register_tracker(output_dir=output_dir, variant_name="unified")

        with args.data.open(encoding="utf-8") as f:
            data = json.load(f)

        documents: list[str] = data["documents"]
        if args.n is not None:
            documents = documents[: args.n]

        print(f"Unified Ingest  |  Dokumente: {len(documents)}"
              + (f"  (von {len(data['documents'])} gesamt, --n {args.n})" if args.n else ""))

        from src.memory import UnifiedMemoryStore
        memory = UnifiedMemoryStore(config=cfg)

        t0 = time.perf_counter()
        if args.resume:
            start_from = memory.read_checkpoint()
            if start_from > 0:
                print(f"Resume ab Chunk {start_from + 1}")
            else:
                print("Kein Checkpoint gefunden — starte von vorne")
                memory.reset()
            memory.ingest_documents(documents, start_from=start_from)
        else:
            memory.reset()
            memory.ingest_documents(documents)
        elapsed = time.perf_counter() - t0

        print(f"Ingest abgeschlossen in {elapsed:.1f}s")

        tel = _read_telemetry(tracker.telemetry_path)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        setup_stats = {
            "variant":                  "unified",
            "n_documents":              len(documents),
            "n_chunks":                 memory.chunk_count,
            "n_entities":               memory.entity_count,
            "n_edges":                  memory.edge_count,
            "ingest_time_s":            round(elapsed, 2),
            # Kosten-Attribution pro Variante: Chunk-Embedding (CPU, keine
            # LLM-Tokens) = Anteil von bm25/vector; Graph-Extraktion (LLM) =
            # Anteil der Graph-Varianten.
            "chunk_embed_time_s":       memory.ingest_timings.get("chunk_embed_s", 0.0),
            "graph_extract_time_s":     memory.ingest_timings.get("graph_extract_s", 0.0),
            "ingest_tokens_prompt":     sum(r.get("prompt_tokens", 0)     for r in tel),
            "ingest_tokens_completion": sum(r.get("completion_tokens", 0) for r in tel),
            "ingest_tokens_total":      sum(r.get("total_tokens", 0)      for r in tel),
            "llm_calls":                len(tel),
            "created_at":               datetime.now(timezone.utc).isoformat(),
        }

        out_path = output_dir / f"unified_{ts}_setup.json"
        out_path.write_text(json.dumps(setup_stats, indent=2), encoding="utf-8")

        print(f"Setup-Stats -> {out_path}")
        print(f"  Zeit      : {elapsed:.1f}s")
        print(f"  Chunks    : {setup_stats['n_chunks']}")
        print(f"  Entities  : {setup_stats['n_entities']}  |  Kanten: {setup_stats['n_edges']}")
        print(f"  LLM-Calls : {setup_stats['llm_calls']}")
        print(f"  Tokens    : {setup_stats['ingest_tokens_total']}")

    finally:
        if containers:
            print("\nStopping containers...")
            stop_containers(containers)


if __name__ == "__main__":
    main()
