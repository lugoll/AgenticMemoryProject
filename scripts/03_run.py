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
    elif variant == "msgraphrag":
        from src.memory.model_msgraphrag import MSGraphRAGMemory
        return MSGraphRAGMemory(config=cfg)
    raise ValueError(f"Unbekannte Variante: {variant}")


def _sum_tokens_for_run(telemetry_path: Path, run_id: str) -> tuple[int, int]:
    """Sum prompt/completion tokens across all llm_call records for run_id.

    Used by end-to-end memories (e.g. msgraphrag) where multiple proxy-routed
    calls happen inside one search(); we have no per-call response object to
    pull usage from, so we re-read the JSONL the TelemetryTracker just wrote.
    """
    if not telemetry_path.exists():
        return 0, 0
    prompt = completion = 0
    for line in telemetry_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("event") == "llm_call" and rec.get("run_id") == run_id:
            prompt += rec.get("prompt_tokens", 0) or 0
            completion += rec.get("completion_tokens", 0) or 0
    return prompt, completion


def check_store_ready(variant: str, memory, cfg) -> None:
    """Prüft ob der Store für diese Variante Daten enthält.
    Bricht mit verständlicher Fehlermeldung ab wenn nicht — damit niemand
    einen leeren Run startet ohne zu merken dass der Ingest fehlt.
    """
    if variant == "bm25":
        store_path = Path(cfg.stores.bm25)
        if not store_path.exists() or store_path.stat().st_size < 1024:
            raise SystemExit(
                f"\n[FEHLER] BM25-Store nicht gefunden oder leer: {store_path}\n"
                f"         Bitte zuerst ausführen:\n"
                f"         uv run python scripts/02_setup.py --variant bm25 --data data/hotpotqa.json\n"
            )

    elif variant == "graph":
        store_path = Path(cfg.stores.graph)
        if not store_path.exists():
            raise SystemExit(
                f"\n[FEHLER] Graph-Store nicht gefunden: {store_path}\n"
                f"         Bitte zuerst ausführen:\n"
                f"         uv run python scripts/02_setup.py --variant graph --data data/hotpotqa.json\n"
            )
        if memory.edge_count == 0:
            raise SystemExit(
                f"\n[FEHLER] Graph-Store ist leer (0 Kanten): {store_path}\n"
                f"         Datei existiert, aber enthält keinen Graphen.\n"
                f"         Bitte Ingest erneut ausführen:\n"
                f"         uv run python scripts/02_setup.py --variant graph --data data/hotpotqa.json\n"
            )
        print(f"  Graph-Store: {memory.node_count} Nodes, {memory.edge_count} Kanten ✓")

    elif variant == "vector":
        try:
            count = memory._collection.count()
        except Exception as e:
            raise SystemExit(
                f"\n[FEHLER] Vector-Store nicht erreichbar: {e}\n"
                f"         Ist ChromaDB gestartet?  docker compose up -d chromadb\n"
                f"         Falls ja, Ingest ausführen:\n"
                f"         uv run python scripts/02_setup.py --variant vector --data data/hotpotqa.json\n"
            )
        if count == 0:
            raise SystemExit(
                f"\n[FEHLER] Vector-Store ist leer (0 Dokumente in ChromaDB)\n"
                f"         Bitte zuerst ausführen:\n"
                f"         uv run python scripts/02_setup.py --variant vector --data data/hotpotqa.json\n"
            )
        print(f"  Vector-Store: {count} Dokumente in ChromaDB ✓")

    elif variant == "msgraphrag":
        settings_path = Path(cfg.stores.msgraphrag.root_dir) / "settings.yaml"
        output_dir = Path(cfg.stores.msgraphrag.root_dir) / "output"
        if not settings_path.exists() or not output_dir.exists() or not any(output_dir.iterdir()):
            raise SystemExit(
                f"\n[FEHLER] MS GraphRAG store nicht gefunden oder leer: {cfg.stores.msgraphrag.root_dir}\n"
                f"         Bitte zuerst ausführen:\n"
                f"         uv run python scripts/02_setup.py --variant msgraphrag --data data/hotpotqa.json\n"
            )
        print(f"  MS GraphRAG store: {output_dir} ✓")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG-Experiment ausführen")
    parser.add_argument("--variant", required=True, choices=["bm25", "vector", "graph", "msgraphrag"])
    parser.add_argument("--n",    type=int,  default=100, help="Anzahl Fragen (default: 100)")
    parser.add_argument("--data", type=Path, default=Path("data/hotpotqa.json"))
    args = parser.parse_args()

    cfg = load_config()
    output_dir = Path(cfg.telemetry.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Start required containers — always include ollama-agent since answer_question
    # calls the agent LLM for every non-end-to-end variant (including bm25).
    containers = get_required_containers(args.variant, include_agent_llm=True)
    if containers:
        print(f"Starting containers for {args.variant}...")
        ensure_containers_running(containers)

    try:
        tracker = register_tracker(output_dir=output_dir, variant_name=args.variant)

        with args.data.open(encoding="utf-8") as f:
            data = json.load(f)

        questions = data["questions"][: args.n]
        print(f"Variante: {args.variant}  |  Fragen: {len(questions)}")

        memory = build_memory(args.variant, cfg)
        check_store_ready(args.variant, memory, cfg)

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        results_path = output_dir / f"{args.variant}_{ts}_results.jsonl"

        with results_path.open("w", encoding="utf-8") as out:
            for i, q in enumerate(questions):
                run_id = uuid.uuid4().hex[:8]
                t_total = time.perf_counter()

                if memory.is_end_to_end:
                    # End-to-end variants (e.g. msgraphrag) synthesise the
                    # answer inside search(). Stamp run_id into the contextvar
                    # so proxy-routed telemetry rows pick it up, then read
                    # tokens back from the JSONL since there's no single
                    # response object to inspect.
                    from src.memory.model_msgraphrag import CURRENT_RUN_ID
                    token = CURRENT_RUN_ID.set(run_id)
                    try:
                        context = memory.search(q["question"])
                    finally:
                        CURRENT_RUN_ID.reset(token)
                    answer = context[0] if context else ""
                    tp, tc = _sum_tokens_for_run(tracker.telemetry_path, run_id)
                    result = AnswerResult(
                        answer=answer,
                        tokens_prompt=tp,
                        tokens_completion=tc,
                        llm_latency_ms=0.0,
                    )
                else:
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
    finally:
        # Stop containers when done
        if containers:
            print("\nStopping containers...")
            stop_containers(containers)


if __name__ == "__main__":
    main()
