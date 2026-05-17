"""
Phase 3 — Experiment ausführen: N Fragen durch eine Variante jagen.

Aufruf:
    uv run python scripts/03_run.py --variant bm25   --n 100
    uv run python scripts/03_run.py --variant vector --n 100
    uv run python scripts/03_run.py --variant graph  --n 100

Voraussetzung: 02_setup.py wurde für die Variante bereits ausgeführt.

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
from src.telemetry.tracker import register_tracker

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using only the "
    "provided context. Be concise — one sentence or less. "
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
        return BM25Memory(
            top_k=cfg.retrieval.top_k,
            storage_path=Path(cfg.stores.bm25),
        )
    elif variant == "vector":
        from src.memory.model_vector import VectorMemory
        return VectorMemory(config=cfg)
    elif variant == "graph":
        from src.memory.model_graph import GraphMemory
        return GraphMemory(config=cfg)
    raise ValueError(f"Unbekannte Variante: {variant}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG-Experiment ausführen")
    parser.add_argument("--variant", required=True, choices=["bm25", "vector", "graph"])
    parser.add_argument("--n",    type=int,  default=100, help="Anzahl Fragen (default: 100)")
    parser.add_argument("--data", type=Path, default=Path("data/hotpotqa.json"))
    args = parser.parse_args()

    cfg = load_config()
    output_dir = Path(cfg.telemetry.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker = register_tracker(output_dir=output_dir, variant_name=args.variant)

    with args.data.open(encoding="utf-8") as f:
        data = json.load(f)

    questions = data["questions"][: args.n]
    print(f"Variante: {args.variant}  |  Fragen: {len(questions)}")

    memory = build_memory(args.variant, cfg)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_path = output_dir / f"{args.variant}_{ts}_results.jsonl"

    with results_path.open("w", encoding="utf-8") as out:
        for i, q in enumerate(questions):
            run_id = uuid.uuid4().hex[:8]
            t_total = time.perf_counter()

            context = memory.search(q["question"])
            result = answer_question(
                question=q["question"],
                context=context,
                cfg_agent=cfg.llm.agent,
                variant=args.variant,
                run_id=run_id,
            )

            total_latency_ms = round((time.perf_counter() - t_total) * 1000, 2)

            record = {
                "run_id":            run_id,
                "variant":           args.variant,
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

    print(f"\nErgebnisse -> {results_path}")
    print(f"Telemetry  -> {tracker.telemetry_path}")
    print(f"LLM-Calls  : {tracker.call_count}")


if __name__ == "__main__":
    main()
