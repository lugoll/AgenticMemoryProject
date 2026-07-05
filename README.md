# AgenticMemoryProject

Wissenschaftliches Benchmark-Framework zum Vergleich von vier RAG-Retrieval-Architekturen
auf Token-Effizienz und Antwortqualität bei Multi-Hop-Fragen (HotpotQA).

**Forschungsfrage:** Welcher RAG-Ansatz erzielt die höchste Antwortqualität auf Multi-Hop-Fragen —
und wie verhält sich das im Verhältnis zum Token-Verbrauch?

Alle Varianten lesen aus **einem gemeinsamen Neo4j-Store** (Unified Ingest):
Chunks werden einmal gespeichert, embedded und volltext-indiziert, der Knowledge
Graph einmal per LLM extrahiert. Die Varianten unterscheiden sich nur im Retrieval —
dadurch operieren alle auf identischen Daten (höhere interne Validität) und der
Ingest läuft genau einmal.

| Variante | Retrieval-Architektur | Retrieval-Kosten |
|---|---|---|
| `bm25` | Lucene-Volltextindex (BM25) über Chunks | keine (kein LLM, kein Embedding) |
| `vector` | Neo4j-Vektorindex (Cosine) über Chunk-Embeddings | 1 Query-Embedding (CPU) |
| `graph` | BM25 Entity-Linking + Hop-für-Hop-BFS über den Knowledge Graph + Cross-Encoder-Rerank | keine (CPU-only) |
| `vectorgraph` | Vektor-verankerte Entity-Suche + Graph-Traversierung (LlamaIndex) + Cross-Encoder-Rerank | 1 Query-Embedding (CPU) |

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
# Erwartet: neo4j, ollama-agent, ollama-judge alle "running"
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

### Schritt 6 — Unified Store aufbauen (ein Lauf für alle Varianten)

```bash
uv run python scripts/02_setup.py --data data/hotpotqa.json
```

Der Ingest schreibt in dieselbe Neo4j-Datenbank:
1. **Chunks** (300 Wörter, 50 Overlap) mit lokalem Embedding (BAAI/bge-base-en-v1.5, CPU)
2. **Volltext-Index** (Lucene/BM25) und **Vektor-Index** (Cosine) über die Chunks
3. **Knowledge Graph**: LLM-Triple-Extraktion (LlamaIndex + JSON-Schema, Prädikat-Whitelist)

> Dauer: mehrere Stunden bei N=500 (LLM-Extraktion dominiert; Chunk-Embedding ~1–2 Min).
> Abbruch ist unkritisch — `--resume` macht ab dem letzten Checkpoint (in Neo4j) weiter.
> Die Kosten-Attribution pro Variante steht im Setup-Report
> (`chunk_embed_time_s` = bm25/vector-Anteil, `graph_extract_time_s` + Tokens = Graph-Anteil).

---

## Experiment ausführen

### Schritt 7 — Alle vier Varianten laufen lassen

```bash
uv run python scripts/03_run.py --variant bm25        --n 500
uv run python scripts/03_run.py --variant vector      --n 500
uv run python scripts/03_run.py --variant graph       --n 500
uv run python scripts/03_run.py --variant vectorgraph --n 500
```

> Kein weiterer Ingest nötig — alle Varianten lesen aus dem Unified Store.  
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

Wenn Docker bereits läuft, Modelle geladen sind und der Unified Store existiert:

```bash
# Nur wenn Datensatz noch nicht existiert:
uv run python scripts/01_load_hotpotqa.py --n 500 --out data/hotpotqa.json

# Runs starten (kein Ingest nötig — Unified Store liegt im Neo4j-Volume)
uv run python scripts/03_run.py --variant bm25        --n 500
uv run python scripts/03_run.py --variant vector      --n 500
uv run python scripts/03_run.py --variant graph       --n 500
uv run python scripts/03_run.py --variant vectorgraph --n 500

# Evaluation
uv run python scripts/04_evaluate.py --all
```

---

## Der Unified Store

Alles liegt im Neo4j-Docker-Volume (`neo4j_data`) — es gibt keine Store-Dateien mehr im Repo:

| Inhalt | Schema | Genutzt von |
|---|---|---|
| Chunks | `(:Chunk {text, embedding})` | bm25, vector |
| Volltext-Index | `chunk_fulltext` (Lucene/BM25 über `Chunk.text`) | bm25 |
| Vektor-Index | `chunk_vector` (Cosine über `Chunk.embedding`) | vector |
| Knowledge Graph | `(:__Entity__ {name})-[PRÄDIKAT]->(:__Entity__)` + `MENTIONS` von Chunks | graph, vectorgraph |
| Checkpoint | `(:Meta {key: 'ingest_checkpoint'})` | 02_setup `--resume` |

Browser-UI zum Inspizieren: <http://localhost:7474> (neo4j / password).

**Methodik-Hinweis:** Durch die Migration auf den Unified Store haben sich zwei
Scoring-Implementierungen geändert (BM25: SQLite FTS5 → Lucene; Vector: ChromaDB →
Neo4j-HNSW, Score-Semantik identisch `(1+cos)/2`). Ergebnisse von vor der Migration
sind daher nicht direkt vergleichbar — alle Varianten müssen neu gelaufen werden.

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

stores:
  neo4j: bolt://localhost:7687  # Der eine Store für alle Varianten
```

---

## Projektstruktur

```
AgenticMemoryProject/
├── src/
│   ├── config/        unified_config.yaml + Dataclass-Loader (cfg.py)
│   ├── memory/        BaseMemory (Neo4j + Unified Ingest) + Retrieval-Views
│   │                  (bm25, vector, graph, vectorgraph) + extraction.py
│   └── telemetry/     LiteLLM Callback → JSONL Token-Tracking
├── scripts/
│   ├── 01_load_hotpotqa.py    HuggingFace → data/hotpotqa.json
│   ├── 02_setup.py            Unified Ingest (einmal für alle Varianten)
│   ├── 03_run.py              Experiment ausführen (--variant ...)
│   └── 04_evaluate.py         EM + F1 + LLM-Judge + summary_table.json
├── data/
│   └── hotpotqa.json          Datensatz (nicht im Git, per 01_load erzeugen)
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
