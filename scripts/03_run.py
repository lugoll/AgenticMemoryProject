"""
Phase 3 — Experiment ausführen: N Fragen durch eine Variante jagen.

Aufruf:
    uv run python scripts/03_run.py --variant bm25            --n 100
    uv run python scripts/03_run.py --variant vector          --n 100
    uv run python scripts/03_run.py --variant vectorrerank     --n 100
    uv run python scripts/03_run.py --variant graph           --n 100
    uv run python scripts/03_run.py --variant graphtext       --n 100
    uv run python scripts/03_run.py --variant vectorgraph     --n 100
    uv run python scripts/03_run.py --variant vectorgraphtext --n 100

Ohne --variant werden alle Varianten nacheinander ausgeführt (bm25, vector,
vectorrerank, graph, graphtext, vectorgraph, vectorgraphtext) — praktisch um die
gesamte Pipeline mit && zu verketten:
    uv run python scripts/03_run.py --n 100

Voraussetzung: der Unified Ingest (02_setup.py) wurde einmal ausgeführt —
alle Varianten lesen aus demselben Neo4j-Store.

Ausgabe:
    evaluations/<variant>_<ts>_results.jsonl   ← eine Zeile pro Frage
    evaluations/<variant>_<ts>_telemetry.jsonl ← automatisch vom TelemetryTracker
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import litellm

from src.config.cfg import load_config
from src.telemetry.tracker import register_tracker, set_run_context
from src.utils.docker_utils import ensure_containers_running, stop_containers, get_required_containers

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using only the "
    "provided context. Give a direct answer — keep it brief but complete. "
    "If the context does not contain the answer, say 'I don't know'."
)


@dataclass
class AnswerResult:
    answer: str
    tokens_prompt: int
    tokens_completion: int
    llm_latency_ms: float


def answer_question(
    question: str,
    context: list[str],
    cfg_agent,
    variant: str,
    run_id: str,
) -> AnswerResult:
    context_str = "\n".join(context) if context else "No context available."
    t0 = time.perf_counter()
    response = litellm.completion(
        model=cfg_agent.model,
        api_base=cfg_agent.base_url,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {question}"},
        ],
        temperature=cfg_agent.temperature,
        max_tokens=cfg_agent.max_tokens,
        metadata={
            "phase": "agent_reasoning",
            "actor": "langgraph_node",
            "variant_name": variant,
            "run_id": run_id,
        },
    )
    llm_latency_ms = (time.perf_counter() - t0) * 1000
    usage = response.usage
    return AnswerResult(
        answer=(response.choices[0].message.content or "").strip(),
        tokens_prompt=usage.prompt_tokens if usage else 0,
        tokens_completion=usage.completion_tokens if usage else 0,
        llm_latency_ms=round(llm_latency_ms, 2),
    )


def build_memory(variant: str, cfg):
    if variant == "bm25":
        from src.memory.model_bm25 import BM25Memory
        return BM25Memory(config=cfg)
    elif variant == "vector":
        from src.memory.model_vector import VectorMemory
        return VectorMemory(config=cfg)
    elif variant == "vectorrerank":
        from src.memory.model_vectorrerank import VectorRerankMemory
        return VectorRerankMemory(config=cfg)
    elif variant == "graph":
        from src.memory.model_graph import GraphMemory
        return GraphMemory(config=cfg)
    elif variant == "graphtext":
        from src.memory.model_graphtext import GraphTextMemory
        return GraphTextMemory(config=cfg)
    elif variant == "vectorgraph":
        from src.memory.model_vectorgraph import VectorGraphMemory
        return VectorGraphMemory(config=cfg)
    elif variant == "vectorgraphtext":
        from src.memory.model_vectorgraphtext import VectorGraphTextMemory
        return VectorGraphTextMemory(config=cfg)
    raise ValueError(f"Unbekannte Variante: {variant}")


def check_store_ready(variant: str, memory, cfg) -> None:
    """Prüft ob der Unified Store Daten für diese Variante enthält.
    Bricht mit verständlicher Fehlermeldung ab wenn nicht — damit niemand
    einen leeren Run startet ohne zu merken dass der Ingest fehlt.
    """
    hint = (
        "         Bitte zuerst den Unified Ingest ausführen:\n"
        "         uv run python scripts/02_setup.py --data data/hotpotqa.json\n"
    )
    try:
        chunks = memory.chunk_count
    except Exception as e:
        raise SystemExit(
            f"\n[FEHLER] Neo4j nicht erreichbar: {e}\n"
            f"         Ist Neo4j gestartet?  docker compose up -d neo4j\n" + hint
        )
    if chunks == 0:
        raise SystemExit(
            f"\n[FEHLER] Unified Store ist leer (0 Chunks in Neo4j)\n" + hint
        )

    if variant in ("graph", "graphtext", "vectorgraph", "vectorgraphtext") and memory.edge_count == 0:
        raise SystemExit(
            f"\n[FEHLER] Knowledge Graph ist leer (0 Kanten in Neo4j)\n"
            f"         Chunks vorhanden ({chunks}), aber keine extrahierten Triples.\n" + hint
        )

    print(
        f"  Unified Store: {chunks} Chunks, {memory.entity_count} Entities, "
        f"{memory.edge_count} Kanten ✓"
    )


ALL_VARIANTS = ["bm25", "vector", "vectorrerank", "graph", "graphtext", "vectorgraph", "vectorgraphtext"]


def run_questions(
    variant: str,
    questions: list[dict],
    cfg,
    memory,
    results_path: Path,
) -> None:
    """Läuft die N Fragen durch eine bereits gebaute Memory und schreibt results.jsonl.

    Reiner Frage-Loop ohne Container- oder Tracker-Verwaltung — geteilt von der
    03-CLI (run_variant) und dem Sweep-Orchestrator (05_sweep.py), damit beide
    identisches Record-Format und identische Telemetrie-Bindung nutzen. Der
    Aufrufer ist dafür verantwortlich, Container zu starten, den Tracker zu
    registrieren und die Memory via build_memory zu erstellen.
    """
    with results_path.open("w", encoding="utf-8") as out:
        for i, q in enumerate(questions):
            run_id = uuid.uuid4().hex[:8]
            # Bind retrieval_overhead telemetry events to this question.
            set_run_context(run_id)
            t_total = time.perf_counter()

            context = memory.search(q["question"])
            result = answer_question(
                question=q["question"],
                context=context,
                cfg_agent=cfg.llm.agent,
                variant=variant,
                run_id=run_id,
            )

            total_latency_ms = round((time.perf_counter() - t_total) * 1000, 2)

            record = {
                "run_id":            run_id,
                "variant":           variant,
                "question":          q["question"],
                "expected":          q["answer"],
                "answer":            result.answer,
                "type":              q.get("type", "unknown"),
                "context":           context,
                "latency_ms":        total_latency_ms,
                "tokens_prompt":     result.tokens_prompt,
                "tokens_completion": result.tokens_completion,
                "ts":                datetime.now(timezone.utc).isoformat(),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            if (i + 1) % 10 == 0 or i == 0:
                print(
                    f"  [{i+1:3d}/{len(questions)}]  {total_latency_ms:6.0f}ms  "
                    f"ctx={len(context)}  answer={result.answer[:60]!r}"
                )


def run_variant(variant: str, questions: list[dict], cfg, output_dir: Path) -> None:
    """Führt eine einzelne Variante aus (Container-Start/-Stop inklusive)."""
    containers = get_required_containers(variant)
    if containers:
        print(f"Starting containers for {variant}...")
        ensure_containers_running(containers)

    try:
        tracker = register_tracker(output_dir=output_dir, variant_name=variant)

        print(f"Variante: {variant}  |  Fragen: {len(questions)}")

        memory = build_memory(variant, cfg)
        check_store_ready(variant, memory, cfg)

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        results_path = output_dir / f"{variant}_{ts}_results.jsonl"

        run_questions(variant, questions, cfg, memory, results_path)

        print(f"\nErgebnisse -> {results_path}")
        print(f"Telemetry  -> {tracker.telemetry_path}")
        print(f"LLM-Calls  : {tracker.call_count}")
    finally:
        # Stop containers when done
        if containers:
            print("\nStopping containers...")
            stop_containers(containers)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG-Experiment ausführen")
    parser.add_argument("--variant", choices=ALL_VARIANTS, default=None,
                        help="Einzelne Variante. Ohne Angabe werden alle "
                             "Varianten nacheinander ausgeführt.")
    parser.add_argument("--n",    type=int,  default=100, help="Anzahl Fragen (default: 100)")
    parser.add_argument("--data", type=Path, default=Path("data/hotpotqa.json"))
    args = parser.parse_args()

    cfg = load_config()
    output_dir = Path(cfg.telemetry.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with args.data.open(encoding="utf-8") as f:
        data = json.load(f)
    questions = data["questions"][: args.n]

    variants = [args.variant] if args.variant else ALL_VARIANTS
    for variant in variants:
        if len(variants) > 1:
            print(f"\n{'='*62}\n  {variant}\n{'='*62}")
        run_variant(variant, questions, cfg, output_dir)


if __name__ == "__main__":
    main()
