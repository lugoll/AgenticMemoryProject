"""
Phase 5 — Hyperparameter-Sweep (ohne retrieval.top_k).

Läuft je Variante das kartesische Produkt eines Parameter-Grids durch N Fragen
und schreibt eine tidy Long-Form-Tabelle (eine Zeile pro (Variante, Grid-Punkt))
mit den gefegten Parametern + Accuracy/Token/Latenz — das Eingabeformat für die
spätere Visualisierung.

Aufruf:
    uv run python scripts/05_sweep.py --grid scripts/sweeps/default.yaml
    uv run python scripts/05_sweep.py --grid scripts/sweeps/default.yaml --variant graph --n 20
    uv run python scripts/05_sweep.py --grid scripts/sweeps/default.yaml --no-judge

Voraussetzung: der Unified Ingest (02_setup.py) wurde einmal ausgeführt. ALLE
gefegten Knöpfe sind Retrieval-Zeit-Parameter und lesen aus demselben Neo4j-Store
— ein Sweep re-ingestet NIE. retrieval.top_k wird auf dem Base-Config-Wert
festgehalten (separate Visualisierungs-Achse).

Design — spiegelt den 03→04 Phasen-Split (Agent- und Judge-Modell sind nicht
gleichzeitig GPU-resident):
    Phase A (Antworten):  neo4j + ollama-agent, Judge aus. Pro Grid-Punkt wird
                          cfg in-memory überschrieben, die Memory neu gebaut und
                          N Fragen beantwortet → <variant>_c<NN>_results.jsonl.
    Phase B (Scoring):    ollama-judge an. Jede Ergebnisdatei wird mit EM + F1 +
                          LLM-Judge bewertet und zu sweep_summary.csv/.jsonl
                          aggregiert.

Ausgabe (evaluations/sweeps/<sweep_ts>/):
    grid.json                      ← combo_id → variant + params (Reproduzierbarkeit)
    <variant>_c<NN>_results.jsonl  ← eine Zeile pro Frage (03-Record-Format)
    sweep_summary.jsonl / .csv     ← eine Zeile pro (Variante, Grid-Punkt)
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import copy
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import yaml

from src.config.cfg import load_config
from src.telemetry.tracker import register_tracker
from src.utils.docker_utils import ensure_containers_running, stop_containers

SCRIPTS_DIR = Path(__file__).parent


def _load_script(module_name: str, filename: str) -> ModuleType:
    """Importiert ein ziffern-präfigiertes Pipeline-Skript als Modul.

    03_run.py / 04_evaluate.py sind keine gültigen Modulnamen (führende Ziffer),
    deshalb per importlib laden. Beide sind durch `if __name__ == "__main__"`
    geschützt — der Import führt nur die Definitionen aus, nicht main().
    """
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # In sys.modules registrieren BEVOR exec — sonst kann @dataclass im Skript
    # sein eigenes Modul nicht auflösen (dataclasses schlägt cls.__module__ nach).
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_run = _load_script("pipeline_03_run", "03_run.py")
_eval = _load_script("pipeline_04_evaluate", "04_evaluate.py")

build_memory = _run.build_memory
check_store_ready = _run.check_store_ready
run_questions = _run.run_questions
exact_match = _eval.exact_match
f1_score = _eval.f1_score
call_judge = _eval.call_judge

# Grid-Feld → Ziel-Sub-Config. Jeder Knopf sitzt entweder auf cfg.retrieval
# oder cfg.graph; alles andere ist ein Tippfehler im Grid.
_RETRIEVAL_FIELDS = {"top_k", "similarity_cutoff", "rerank_fetch_k", "hop0_fetch_k"}
_GRAPH_FIELDS = {"max_hops", "seed_top_k", "max_frontier", "max_candidates",
                 "beam_width", "max_per_tail"}


def apply_overrides(cfg, params: dict) -> None:
    """Überschreibt cfg-Felder in-place (Dataclasses sind mutable)."""
    for key, value in params.items():
        if key in _RETRIEVAL_FIELDS:
            setattr(cfg.retrieval, key, value)
        elif key in _GRAPH_FIELDS:
            setattr(cfg.graph, key, value)
        else:
            raise ValueError(
                f"Unbekannter Sweep-Parameter '{key}'. Erlaubt: "
                f"{sorted(_RETRIEVAL_FIELDS | _GRAPH_FIELDS)}"
            )


def expand_grid(grid: dict) -> list[dict]:
    """Grid-YAML → Liste von Combos {combo_id, variant, params}.

    Kartesisches Produkt PRO Variante über deren Parameter-Listen.
    """
    combos: list[dict] = []
    for variant, param_lists in grid["variants"].items():
        if not param_lists:
            combos.append({"combo_id": f"{variant}_c00", "variant": variant, "params": {}})
            continue
        keys = list(param_lists.keys())
        value_lists = [param_lists[k] for k in keys]
        for i, values in enumerate(itertools.product(*value_lists)):
            params = dict(zip(keys, values))
            combos.append({
                "combo_id": f"{variant}_c{i:02d}",
                "variant": variant,
                "params": params,
            })
    return combos


# ── Phase A — Antworten ───────────────────────────────────────────────────────

def run_answers(combos: list[dict], base_cfg, questions: list[dict], sweep_dir: Path) -> None:
    """Läuft alle Grid-Punkte durch die Fragen (Container werden vom Aufrufer verwaltet)."""
    # Store-Check einmal pro Variante (nicht pro Combo).
    checked: set[str] = set()

    for combo in combos:
        variant, params = combo["variant"], combo["params"]
        cfg = copy.deepcopy(base_cfg)
        apply_overrides(cfg, params)

        print(f"\n── {combo['combo_id']}  params={params or '(none)'} "
              f"| {len(questions)} Fragen ──")

        # Tracker pro Combo → eigene Telemetrie-Datei mit passendem Namen.
        register_tracker(output_dir=sweep_dir, variant_name=combo["combo_id"])

        memory = build_memory(variant, cfg)
        if variant not in checked:
            check_store_ready(variant, memory, cfg)
            checked.add(variant)

        results_path = sweep_dir / f"{combo['combo_id']}_results.jsonl"
        run_questions(variant, questions, cfg, memory, results_path)


# ── Phase B — Scoring & Aggregation ───────────────────────────────────────────

def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def score_combo(combo: dict, base_cfg, sweep_dir: Path, use_judge: bool) -> dict:
    """Bewertet eine Combo-Ergebnisdatei → eine aggregierte Summary-Zeile."""
    results_path = sweep_dir / f"{combo['combo_id']}_results.jsonl"
    records = _read_jsonl(results_path)
    n = len(records)

    em_count = 0
    correct = 0.0            # judge-gewichtet: EM=1, PARTIAL=0.5
    f1_sum = 0.0
    latency_sum = 0.0
    tokens_sum = 0

    for rec in records:
        em = exact_match(rec["answer"], rec["expected"])
        f1_sum += f1_score(rec["answer"], rec["expected"])
        latency_sum += rec.get("latency_ms", 0)
        tokens_sum += rec.get("tokens_prompt", 0) + rec.get("tokens_completion", 0)

        if em:
            em_count += 1
            correct += 1
        elif use_judge and rec.get("type") == "bridge":
            verdict = call_judge(
                question=rec["question"], expected=rec["expected"],
                answer=rec["answer"], cfg=base_cfg,
                run_id=rec.get("run_id", "unknown"),
            )
            if verdict == "CORRECT":
                correct += 1
            elif verdict == "PARTIAL":
                correct += 0.5

    row = {
        "combo_id": combo["combo_id"],
        "variant": combo["variant"],
        # gefegte Parameter als eigene Spalten (leer bei Varianten ohne den Knopf)
        **combo["params"],
        # top_k ist normalerweise die fixe Achse (Base-Config-Wert), KANN aber
        # selbst gefegt werden (topk.yaml) — dann gewinnt der Grid-Wert.
        "top_k": combo["params"].get("top_k", base_cfg.retrieval.top_k),
        "n": n,
        "accuracy_em": round(em_count / n, 4) if n else 0.0,
        "accuracy_judge": round(correct / n, 4) if n else 0.0,
        "f1_mean": round(f1_sum / n, 4) if n else 0.0,
        "query_latency_ms_mean": round(latency_sum / n, 2) if n else 0.0,
        "query_tokens_total_mean": round(tokens_sum / n, 2) if n else 0.0,
    }
    return row


def write_summary(rows: list[dict], sweep_dir: Path) -> None:
    """Schreibt sweep_summary.jsonl + .csv (Long-Form, eine Zeile pro Combo)."""
    jsonl_path = sweep_dir / "sweep_summary.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # CSV-Header = Union aller Keys (Param-Spalten variieren je Variante).
    fixed = ["combo_id", "variant"]
    metrics = ["top_k", "n", "accuracy_em", "accuracy_judge", "f1_mean",
               "query_latency_ms_mean", "query_tokens_total_mean"]
    param_keys = sorted(
        {k for row in rows for k in row if k not in fixed and k not in metrics}
    )
    header = fixed + param_keys + metrics

    csv_path = sweep_dir / "sweep_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in header})

    print(f"\nsweep_summary.jsonl -> {jsonl_path}")
    print(f"sweep_summary.csv   -> {csv_path}")

    # Kurzausgabe.
    print(f"\n{'combo':<24} {'judge':>7}  {'EM':>6}  {'F1':>6}  {'ms/q':>7}  {'tok/q':>7}")
    print("-" * 66)
    for row in rows:
        print(
            f"{row['combo_id']:<24} {row['accuracy_judge']:>7.2%}  "
            f"{row['accuracy_em']:>6.2%}  {row['f1_mean']:>6.3f}  "
            f"{row['query_latency_ms_mean']:>7.0f}  {row['query_tokens_total_mean']:>7.0f}"
        )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter-Sweep (ohne top_k)")
    parser.add_argument("--grid", type=Path, default=Path("scripts/sweeps/default.yaml"))
    parser.add_argument("--n", type=int, default=None,
                        help="Fragen pro Grid-Punkt (überschreibt grid.n)")
    parser.add_argument("--variant", default=None,
                        help="Nur diese Variante fegen (sonst alle im Grid)")
    parser.add_argument("--data", type=Path, default=Path("data/hotpotqa.json"))
    parser.add_argument("--no-judge", action="store_true",
                        help="LLM-Judge überspringen (nur EM + F1)")
    args = parser.parse_args()

    base_cfg = load_config()
    grid = yaml.safe_load(args.grid.read_text(encoding="utf-8"))

    if args.variant:
        if args.variant not in grid["variants"]:
            raise SystemExit(f"Variante '{args.variant}' nicht im Grid {args.grid}")
        grid = {**grid, "variants": {args.variant: grid["variants"][args.variant]}}

    n = args.n if args.n is not None else grid.get("n", 100)
    combos = expand_grid(grid)

    with args.data.open(encoding="utf-8") as f:
        questions = json.load(f)["questions"][:n]

    sweep_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sweep_dir = Path(base_cfg.telemetry.output_dir) / "sweeps" / sweep_ts
    sweep_dir.mkdir(parents=True, exist_ok=True)

    (sweep_dir / "grid.json").write_text(
        json.dumps(combos, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Sweep: {len(combos)} Grid-Punkte über {len(grid['variants'])} Variante(n)"
          f" | {n} Fragen | Judge: {'nein' if args.no_judge else 'ja'}")
    print(f"Ausgabe -> {sweep_dir}")

    use_judge = not args.no_judge

    # ── Phase A — Antworten ──────────────────────────────────────────────
    print(f"\n{'='*66}\n  PHASE A — Antworten (neo4j + ollama-agent)\n{'='*66}")
    ensure_containers_running(["neo4j", "ollama-agent"])
    try:
        run_answers(combos, base_cfg, questions, sweep_dir)
    finally:
        if use_judge:
            # ollama-agent stoppen, damit die GPU für den Judge frei wird.
            stop_containers(["ollama-agent"])

    # ── Phase B — Scoring ────────────────────────────────────────────────
    print(f"\n{'='*66}\n  PHASE B — Scoring (EM + F1{' + Judge' if use_judge else ''})\n{'='*66}")
    if use_judge:
        ensure_containers_running(["ollama-judge"])
        register_tracker(output_dir=sweep_dir, variant_name="judge")
    try:
        rows = [score_combo(c, base_cfg, sweep_dir, use_judge) for c in combos]
    finally:
        if use_judge:
            stop_containers(["ollama-judge"])
        stop_containers(["neo4j"])

    write_summary(rows, sweep_dir)


if __name__ == "__main__":
    main()
