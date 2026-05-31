# AgenticMemoryProject

Wissenschaftliches Benchmark-Framework zum Vergleich von drei RAG-Retrieval-Architekturen
auf Token-Effizienz und Antwortqualität bei Multi-Hop-Fragen (HotpotQA).

**Forschungsfrage:** Welcher RAG-Ansatz erzielt die höchste Antwortqualität auf Multi-Hop-Fragen —
und wie verhält sich das im Verhältnis zum Token-Verbrauch?

| Variante | Architektur | Ingest-Aufwand |
|---|---|---|
| `bm25` | BM25 Keyword-Suche | keiner (Store im Repo) |
| `vector` | Semantic Embeddings via ChromaDB | ~72s lokal (kein LLM) |
| `graph` | Knowledge Graph + BFS via NetworkX | ~100 Min (LLM, Store im Repo) |
| `msgraphrag` | Microsoft GraphRAG (Community-Hierarchie, lokale Suche) | ~mehrere Stunden (LLM, Index lokal) |

> **`msgraphrag`-Hinweise:** Diese Variante ruft das `graphrag`-CLI per
> Subprocess auf und schickt alle LLM-Calls durch einen in-Process
> LiteLLM-Proxy, damit Token-Verbrauch identisch erfasst wird. Embeddings
> laufen über denselben lokalen `BAAI/bge-base-en-v1.5`-Encoder wie die
> `vector`-Variante. Da GraphRAG seinen Index nur einmalig baut, wirft
> `UpdateMemory` für diese Variante `NotImplementedError`. Außerdem läuft
> GraphRAG end-to-end: `search()` liefert direkt die finale Antwort und
> `03_run.py` überspringt den zusätzlichen Agent-LLM-Call.

---

## Einmaliges Setup (Erstes Mal)

### Schritt 1 — Voraussetzungen prüfen

- Docker Desktop läuft (mit GPU-Unterstützung für Ollama)
- `uv` ist installiert: `uv --version`
- `git` ist installiert: `git --version`
- VRAM-Empfehlung: ≥ 16 GB (Agent 8B ~5 GB + Judge 14B ~9 GB, sequenziell)

### Schritt 2 — Repository klonen & Abhängigkeiten installieren

```bash
git clone <repo-url>
cd AgenticMemoryProject
uv sync
```

### Schritt 3 — Docker-Container starten

```bash
docker compose up -d
```

Prüfen ob alles läuft:

```bash
docker compose ps
# Erwartet: chromadb, ollama-agent, ollama-judge alle "running"
```

### Schritt 4 — LLM-Modelle laden (einmalig, ~14 GB gesamt)

```bash
docker exec ollama-agent ollama pull llama3.1:8b
docker exec ollama-judge ollama pull qwen2.5:14b
```

> **Hinweis:** Die Downloads können 5–15 Minuten dauern je nach Verbindung.
> Fortschritt wird direkt in der Konsole angezeigt.

### Schritt 5 — Datensatz laden

```bash
uv run python scripts/01_load_hotpotqa.py --n 500 --out data/hotpotqa.json
```

Ausgabe: `data/hotpotqa.json` mit 500 Fragen (250 bridge, 250 comparison) und ~3.500 Dokumenten.

> Dauer: ~10–30 Sekunden (HuggingFace-Download beim ersten Mal etwas länger).

### Schritt 6 — Vector-Store aufbauen

BM25 und Graph sind bereits als fertige Stores im Repository enthalten (`data/stores/`).
Nur der Vector-Store muss lokal aufgebaut werden (ChromaDB-Volume ist nicht committbar):

```bash
uv run python scripts/02_setup.py --variant vector --data data/hotpotqa.json
```

> Dauer: ~72 Sekunden. Kein LLM-Call — nur lokale Embeddings (BAAI/bge-base-en-v1.5).

---

## Experiment ausführen

### Schritt 7 — Alle drei Varianten laufen lassen

```bash
uv run python scripts/03_run.py --variant bm25       --n 500
uv run python scripts/03_run.py --variant vector     --n 500
uv run python scripts/03_run.py --variant graph      --n 500
uv run python scripts/03_run.py --variant msgraphrag --n 500
```

> Dauer pro Variante: BM25 ~5 Min | Vector ~12 Min | Graph ~15 Min | MS GraphRAG variabel  
> Ausgabe je: `evaluations/<variante>_<timestamp>_results.jsonl`

### Schritt 8 — Ergebnisse auswerten

```bash
uv run python scripts/04_evaluate.py --all
```

> Dauer: ~10–20 Min (LLM-Judge läuft für bridge-Fragen mit EM=false).  
> Ausgabe: `evaluations/<variante>_<timestamp>_scores.jsonl` + `evaluations/summary_table.json`

### Schritt 9 — Ergebnisse ansehen

```bash
# Schnelle Übersicht
cat evaluations/summary_table.json

# Detaillierte Scores einer Variante
cat evaluations/graph_*_scores.jsonl | head -20
```

---

## Wiederholung (nach erstem Setup)

Wenn Docker bereits läuft und Modelle geladen sind:

```bash
# Nur wenn Datensatz noch nicht existiert:
uv run python scripts/01_load_hotpotqa.py --n 500 --out data/hotpotqa.json

# Runs starten (BM25 + Graph: kein Ingest nötig, Stores aus Git)
uv run python scripts/03_run.py --variant bm25   --n 500
uv run python scripts/03_run.py --variant vector --n 500
uv run python scripts/03_run.py --variant graph  --n 500

# Evaluation
uv run python scripts/04_evaluate.py --all
```

---

## Stores: Was ist im Repository enthalten?

| Store | Datei | Im Git? | Neu aufbauen |
|---|---|---|---|
| BM25 | `data/stores/bm25.db` | ✅ ja | `02_setup.py --variant bm25` (< 1 Min) |
| Graph | `data/stores/graph.json` | ✅ ja | `02_setup.py --variant graph` (~6–7 Std, LLM) |
| Vector | ChromaDB Docker Volume | ❌ nein | `02_setup.py --variant vector` (~5 Min) |

**Warum ChromaDB nicht im Repo?** ChromaDB speichert den Index als binäres Docker Volume —
kein git-taugliches Format. BM25 (~1 MB) und Graph (~7 MB bei N=500) liegen als
JSON/SQLite-Dateien vor und sind gut git-committbar.

---

## VRAM-Hinweis (16 GB)

Agent (`llama3.1:8b`, ~5 GB) und Judge (`qwen2.5:14b`, ~9 GB) laufen **nie gleichzeitig**.
`OLLAMA_KEEP_ALIVE=0` am Judge-Container sorgt dafür, dass nach jedem Script-Ende
das Modell sofort aus dem VRAM entladen wird.

> Nicht `03_run.py` und `04_evaluate.py` gleichzeitig in zwei Terminals starten —
> das würde beide Modelle gleichzeitig laden (~14 GB) und ist bei 16 GB VRAM knapp.

---

## Konfiguration

Alle Parameter in einem einzigen File: `src/config/unified_config.yaml`

```yaml
llm:
  agent:  ollama/llama3.1:8b  @ localhost:11434  # QA-Agent + Graph-Ingest
  judge:  ollama/qwen2.5:14b  @ localhost:11435  # LLM-as-Judge (Evaluation)

graph:
  max_hops: 3    # HotpotQA braucht bis zu 3 Hops
  top_k:    10   # Triples sind kurz (~8 Wörter), mehr Kontext nötig

retrieval:
  top_k:             5     # Textpassagen für BM25 und Vector
  similarity_cutoff: 0.5   # Cutoff für Vector-Similarity
```

---

## Projektstruktur

```
AgenticMemoryProject/
├── src/
│   ├── config/        unified_config.yaml + Dataclass-Loader (cfg.py)
│   ├── memory/        BaseMemory + Implementierungen (bm25, vector, graph)
│   └── telemetry/     LiteLLM Callback → JSONL Token-Tracking
├── scripts/
│   ├── 01_load_hotpotqa.py    HuggingFace → data/hotpotqa.json
│   ├── 02_setup.py            Store aufbauen (bm25 | vector | graph)
│   ├── 03_run.py              Experiment ausführen
│   └── 04_evaluate.py         EM + F1 + LLM-Judge + summary_table.json
├── data/
│   ├── hotpotqa.json          Datensatz (nicht im Git, per 01_load erzeugen)
│   └── stores/
│       ├── bm25.db            ✅ im Git
│       ├── graph.json         ✅ im Git
│       └── graph.checkpoint   ✅ im Git
├── evaluations/               Run-Outputs (nicht im Git)
└── docs/
    └── graphrag_analysis.md   Analyse: Triple-Extraktion vs. HotpotQA, Paper-Vergleich
```

---

## Weiterführende Dokumentation

| Dokument | Inhalt |
|---|---|
| [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) | Vollständiger Experiment-Plan, alle Architektur-Entscheidungen |
| [`docs/graphrag_analysis.md`](docs/graphrag_analysis.md) | Warum GraphRAG auf HotpotQA strukturell benachteiligt ist, Paper-Vergleich |
