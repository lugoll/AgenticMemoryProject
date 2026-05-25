"""
Phase 4 — Evaluation & Aggregation.

Liest results.jsonl, berechnet EM + F1 + LLM-as-Judge,
schreibt scores.jsonl und summary_table.json.

Aufruf:
    uv run python scripts/04_evaluate.py --results evaluations/bm25_*_results.jsonl
    uv run python scripts/04_evaluate.py --all               # alle results.jsonl
    uv run python scripts/04_evaluate.py --all --no-judge    # nur EM + F1, kein LLM
"""
from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import litellm

from src.config.cfg import load_config
from src.telemetry.tracker import register_tracker
from src.utils.docker_utils import ensure_containers_running, stop_containers

# ── Normalisierung (Standard HotpotQA-Metrik) ────────────────────────────────

def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> bool:
    return _normalize(pred) == _normalize(gold)


def f1_score(pred: str, gold: str) -> float:
    pred_toks = _normalize(pred).split()
    gold_toks = _normalize(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_toks)
    recall    = n_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


# ── LLM-as-Judge ─────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = (
    "You are a strict evaluation judge for a question-answering benchmark. "
    "Reply with exactly one word: CORRECT, PARTIAL, or INCORRECT."
)

_JUDGE_USER = """\
Question:       {question}
Correct answer: {expected}
Model answer:   {answer}

Grading rules — apply in order:
1. INCORRECT — if the model says "I don't know", gives a wrong answer, or contradicts the correct answer.
2. CORRECT   — if the model answer contains the correct answer, even with extra words or explanation.
   Extra context does NOT downgrade to PARTIAL.
   Synonyms and minor reformulations count as CORRECT.
3. PARTIAL   — only if the model gives a relevant but genuinely incomplete answer
   (e.g. names one person when two are required, or gives a vague hint without the actual answer).

Verdict:"""

_VERDICT_RE = re.compile(r"\b(CORRECT|PARTIAL|INCORRECT)\b")


def call_judge(question: str, expected: str, answer: str, cfg, run_id: str) -> str:
    response = litellm.completion(
        model=cfg.llm.judge.model,
        api_base=cfg.llm.judge.base_url,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": _JUDGE_USER.format(
                question=question, expected=expected, answer=answer
            )},
        ],
        temperature=0.0,
        max_tokens=10,
        metadata={
            "phase": "evaluation",
            "actor": "llm_as_judge",
            "variant_name": "judge",
            "run_id": run_id,
        },
    )
    raw = (response.choices[0].message.content or "").strip().upper()
    m = _VERDICT_RE.search(raw)
    return m.group(1) if m else "INCORRECT"


# ── Datei-Helfer ──────────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _latest_setup(output_dir: Path, variant: str) -> dict | None:
    files = sorted(output_dir.glob(f"{variant}_*_setup.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


# ── Haupt-Evaluations-Logik ───────────────────────────────────────────────────

def evaluate_file(
    results_path: Path,
    output_dir: Path,
    cfg,
    use_judge: bool,
) -> tuple[str, list[dict]]:
    records = _read_jsonl(results_path)
    if not records:
        return "unknown", []

    variant = records[0].get("variant", results_path.stem.split("_")[0])
    scores: list[dict] = []

    for rec in records:
        em = exact_match(rec["answer"], rec["expected"])
        f1 = f1_score(rec["answer"], rec["expected"])

        # LLM-Judge: nur für bridge-Fragen wo EM = False
        verdict: str | None = None
        if use_judge and rec.get("type") == "bridge" and not em:
            verdict = call_judge(
                question=rec["question"],
                expected=rec["expected"],
                answer=rec["answer"],
                cfg=cfg,
                run_id=rec.get("run_id", "unknown"),
            )

        scores.append({
            "run_id":        rec.get("run_id"),
            "variant":       variant,
            "question":      rec["question"],
            "expected":      rec["expected"],
            "answer":        rec["answer"],
            "type":          rec.get("type", "unknown"),
            "exact_match":   em,
            "f1":            round(f1, 4),
            "judge_verdict": verdict,
            # pass-through für summary aggregation
            "_latency_ms":        rec.get("latency_ms", 0),
            "_tokens_prompt":     rec.get("tokens_prompt", 0),
            "_tokens_completion": rec.get("tokens_completion", 0),
        })

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    scores_path = output_dir / f"{variant}_{ts}_scores.jsonl"
    with scores_path.open("w", encoding="utf-8") as f:
        for s in scores:
            # Private _-Felder nicht in die Ausgabe schreiben
            public = {k: v for k, v in s.items() if not k.startswith("_")}
            f.write(json.dumps(public, ensure_ascii=False) + "\n")

    print(f"  {results_path.name} -> {scores_path.name}  ({len(scores)} Fragen)")
    return variant, scores


# ── summary_table.json ────────────────────────────────────────────────────────

def build_summary(all_scores: dict[str, list[dict]], output_dir: Path, cfg) -> None:
    summary: dict[str, dict] = {}

    for variant, scores in all_scores.items():
        n = len(scores)
        if n == 0:
            continue

        em_count    = sum(1 for s in scores if s["exact_match"])
        f1_vals     = [s["f1"] for s in scores]

        # Judge-Accuracy: EM=True zählt als korrekt; bridge+EM=False nach Verdict
        correct_count = em_count
        for s in scores:
            if not s["exact_match"]:
                if s.get("judge_verdict") == "CORRECT":
                    correct_count += 1
                elif s.get("judge_verdict") == "PARTIAL":
                    correct_count += 0.5

        setup = _latest_setup(output_dir, variant) or {}

        summary[variant] = {
            "n":                           n,
            "accuracy_em":                 round(em_count / n, 4),
            "accuracy_judge":              round(correct_count / n, 4),
            "f1_mean":                     round(sum(f1_vals) / n, 4),
            "ingest_time_s":               setup.get("ingest_time_s", None),
            "ingest_tokens_prompt":        setup.get("ingest_tokens_prompt", None),
            "ingest_tokens_completion":    setup.get("ingest_tokens_completion", None),
            "ingest_tokens_total":         setup.get("ingest_tokens_total", None),
            "query_latency_ms_mean":       round(sum(s["_latency_ms"] for s in scores) / n, 2),
            "query_tokens_prompt_mean":    round(sum(s["_tokens_prompt"] for s in scores) / n, 2),
            "query_tokens_completion_mean":round(sum(s["_tokens_completion"] for s in scores) / n, 2),
            "query_tokens_total_mean":     round(sum(s["_tokens_prompt"] + s["_tokens_completion"] for s in scores) / n, 2),
        }

    out = output_dir / "summary_table.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsummary_table.json -> {out}")

    # Kurzausgabe in die Konsole
    print(f"\n{'Variante':<10} {'N':>5}  {'EM':>6}  {'Judge':>6}  {'F1':>6}  {'ms/q':>7}  {'tok/q':>7}")
    print("-" * 62)
    for variant, row in summary.items():
        print(
            f"{variant:<10} {row['n']:>5}  "
            f"{row['accuracy_em']:>6.2%}  {row['accuracy_judge']:>6.2%}  "
            f"{row['f1_mean']:>6.3f}  {row['query_latency_ms_mean']:>7.0f}  "
            f"{row['query_tokens_total_mean']:>7.0f}"
        )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation: EM + F1 + LLM-Judge")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--results", nargs="+", type=Path,
                       help="Pfad(e) zu results.jsonl Dateien")
    group.add_argument("--all", action="store_true",
                       help="Alle *_results.jsonl in evaluations/ auswerten")
    parser.add_argument("--no-judge", action="store_true",
                        help="LLM-Judge überspringen (nur EM + F1)")
    args = parser.parse_args()

    cfg = load_config()
    output_dir = Path(cfg.telemetry.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_judge = not args.no_judge

    # Start ollama-judge only if using LLM-as-Judge
    if use_judge:
        print("Starting ollama-judge for evaluation...")
        ensure_containers_running(["ollama-judge"])

    try:
        if use_judge:
            register_tracker(output_dir=output_dir, variant_name="judge")

        if args.all:
            result_files = sorted(output_dir.glob("*_results.jsonl"))
            # Alte Dateien mit anderer Namenskonvention ausschließen
            result_files = [f for f in result_files if "_baseline_" not in f.name and "_rag_" not in f.name]
        else:
            result_files = [Path(p) for p in args.results]

        if not result_files:
            print("Keine results.jsonl gefunden.")
            return

        print(f"Evaluiere {len(result_files)} Datei(en)  |  Judge: {'ja' if use_judge else 'nein'}\n")

        all_scores: dict[str, list[dict]] = {}
        for path in result_files:
            variant, scores = evaluate_file(path, output_dir, cfg, use_judge)
            # Mehrere Dateien pro Variante zusammenfassen
            all_scores.setdefault(variant, []).extend(scores)

        build_summary(all_scores, output_dir, cfg)
    finally:
        # Stop containers when done
        if use_judge:
            print("\nStopping ollama-judge...")
            stop_containers(["ollama-judge"])


if __name__ == "__main__":
    main()
