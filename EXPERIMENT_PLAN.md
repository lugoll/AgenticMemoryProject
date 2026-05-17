# Experiment Plan — RAG Benchmark auf HotpotQA

**Modul:** Anwendungsfelder Business Analytics (MKBA), SS26  
**Forschungsfrage:** Welcher RAG-Ansatz (BM25 / Vector / Graph) erzielt die höchste Antwortqualität auf Multi-Hop-Fragen — und wie verhält sich das im Verhältnis zum Token-Verbrauch?  
**Datensatz:** HotpotQA (distractor split, 100% multi-hop)  
**Methodik:** CRISP-DM

---

## Architekturentscheidung

Das Repo war wie eine Produktions-App gebaut (LangGraph, Factory-Pattern, Pydantic-Config, abstrakte Pipeline-Klassen). Für ein wissenschaftliches Experiment ist das Overhead ohne Mehrwert: schwerer zu verstehen, schwerer zu erklären, schwerer zu debuggen.

**Neuer Ansatz: 4 flache Scripts, modular, experimentell.**

| Was bleibt | Was fliegt raus |
|---|---|
| `src/memory/model_bm25.py` | `src/agent/` komplett |
| `src/memory/model_vector.py` | `src/pipelines/` komplett |
| `src/memory/model_graph.py` | `src/config/settings.py` (Pydantic) |
| `src/memory/base.py` | `main.py` |
| `src/telemetry/tracker.py` | `model_lightrag.py`, `model_vecgraph.py` |

Die Memory-Implementierungen sind die wissenschaftliche Arbeit — die bleiben.  
LangGraph für `retrieve → reason` ist ein Hammer für eine Schraube — der geht.

---

## Zielstruktur

```
scripts/
  01_load_hotpotqa.py    # Daten laden & konvertieren
  02_setup.py            # Store aufbauen    (--variant bm25|vector|graph)
  03_run.py              # Experiment laufen (--variant X --n 100 --model Y)
  04_evaluate.py         # Ergebnisse bewerten & aggregieren

src/
  memory/
    base.py              # ABC, unverändert
    model_bm25.py        # unverändert
    model_vector.py      # unverändert
    model_graph.py       # Refactor: NetworkX statt manueller BFS
  telemetry/
    tracker.py           # unverändert (LiteLLM Token-Tracking)

data/
  hotpotqa.json          # Ausgabe von 01 (nicht ins Git)
  stores/
    bm25.db              # Ausgabe von 02 --variant bm25
    graph.json           # Ausgabe von 02 --variant graph
    # vector läuft in ChromaDB (Docker), kein lokales File

evaluations/
  bm25_20260601_results.jsonl    # Ausgabe von 03
  bm25_20260601_telemetry.jsonl  # Ausgabe von tracker.py
  bm25_20260601_scores.jsonl     # Ausgabe von 04
  summary_table.json             # Aggregat aller Varianten

notebooks/
  01_data_exploration.ipynb      # HotpotQA EDA
  02_results_analysis.ipynb      # Vergleichstabelle + Plots
  03_graph_visualization.ipynb   # NetworkX Graph visualisieren
```

---

## Datenfluss

```
HuggingFace
    │
    ▼
01_load_hotpotqa.py
    │  data/hotpotqa.json
    ▼
02_setup.py ──────────────────────────────────────────────┐
    │  --variant bm25   → data/stores/bm25.db              │
    │  --variant vector → ChromaDB (Docker)                │
    │  --variant graph  → data/stores/graph.json           │
    ▼                                                       │
03_run.py                           (liest Store von 02) ──┘
    │  evaluations/<variant>_<ts>_results.jsonl
    │  evaluations/<variant>_<ts>_telemetry.jsonl
    ▼
04_evaluate.py
    │  evaluations/<variant>_<ts>_scores.jsonl
    │  evaluations/summary_table.json
    ▼
notebooks/02_results_analysis.ipynb
```

---

## Phase 0 — Setup von Null

> Ziel: Sauberer Startpunkt. Repo aufgeräumt, alle Services laufen, Python-Env bereit.

---

### 0.A — Systemvoraussetzungen

- [x ] **0.1** Sicherstellen dass folgendes installiert ist:
  - Docker Desktop (Windows) mit aktiviertem WSL2-Backend
  - NVIDIA Container Toolkit (`nvidia-ctk`) — ermöglicht GPU-Zugriff in Containern
  - `uv` (Python Package Manager)
  - Git

- [x ] **0.2** NVIDIA Container Toolkit für Docker Desktop konfigurieren:
  ```bash
  # In WSL2-Terminal:
  nvidia-ctk runtime configure --runtime=docker
  # Docker Desktop danach neu starten
  ```

- [x ] **0.3** GPU-Zugriff in Docker testen:
  ```bash
  docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
  # Ausgabe muss die GPU zeigen — wenn nicht, Toolkit-Setup wiederholen
  ```

- [ x] **0.4** VRAM prüfen — zwei 8B-Modelle gleichzeitig brauchen ~16 GB:
  ```bash
  nvidia-smi --query-gpu=name,memory.total --format=csv
  # Beispiel: NVIDIA RTX 4090, 24576 MiB → ausreichend
  # Bei < 16 GB: entweder kleinere Modelle wählen (z.B. 3B statt 8B)
  #              oder vllm-agent und vllm-judge sequenziell starten
  ```

- [ ] **0.5** HuggingFace-Account anlegen und Zugriff auf Llama 3.1 beantragen:
  - Account: https://huggingface.co
  - Modell-Seite aufrufen und "Request access" klicken: `meta-llama/Llama-3.1-8B-Instruct`
  - API-Token generieren: HuggingFace → Settings → Access Tokens → New token (read)

---

### 0.B — Repo aufräumen

- [ ] **0.6** Löschen: `src/agent/` (komplett — LangGraph, BaseAgent, factory)
- [ ] **0.7** Löschen: `src/pipelines/` (komplett — ingest.py, qa.py)
- [ ] **0.8** Löschen: `src/config/settings.py` (Pydantic-Loader nicht mehr nötig)
- [ ] **0.9** Löschen: `main.py`
- [ ] **0.10** Löschen: `src/memory/model_lightrag.py`, `src/memory/model_vecgraph.py`
- [ ] **0.11** Löschen: `tests/integration/` (waren gegen LangGraph-Logik)
- [ ] **0.12** Löschen: `Dockerfile` (der alte `app`-Service im compose wird ersetzt, kein Build mehr nötig)
- [ ] **0.13** Unit Tests in `tests/unit/` prüfen — die für memory-Klassen behalten, rest löschen
- [ ] **0.14** `scripts/`, `data/stores/` und `evaluations/` Ordner anlegen:
  ```bash
  mkdir scripts
  mkdir -p data/stores
  mkdir evaluations
  ```
- [ ] **0.15** `.gitignore` aktualisieren:
  ```
  # Experiment-Outputs — nicht ins Git
  data/hotpotqa.json
  data/stores/
  evaluations/
  .env
  ```

---

### 0.C — Python-Abhängigkeiten

- [ ] **0.16** Abhängigkeiten installieren:
  ```bash
  uv add pyyaml datasets networkx litellm chromadb-client sentence-transformers
  ```

- [ ] **0.17** `src/config/unified_config.yaml` neu schreiben (komplettes File):
  ```yaml
  llm:
    agent:
      model: "openai/meta-llama/Llama-3.1-8B-Instruct"
      base_url: "http://localhost:8001/v1"
      temperature: 0.0
      max_tokens: 256
    ingest:
      # Gleicher Container wie agent — läuft sequenziell, nie gleichzeitig
      # guided_json erzwingt valide Triple-Struktur bei der Graphextraktion
      model: "openai/meta-llama/Llama-3.1-8B-Instruct"
      base_url: "http://localhost:8001/v1"
      temperature: 0.0
      max_tokens: 128
    judge:
      # Andere Model-Family → kein Self-Enhancement-Bias
      # guided_choice erzwingt exakt CORRECT / PARTIAL / INCORRECT
      model: "openai/Qwen/Qwen2.5-7B-Instruct"
      base_url: "http://localhost:8002/v1"
      temperature: 0.0
      max_tokens: 5

  embedding:
    # BAAI/bge-base-en-v1.5: MIT-Lizenz, ~400MB, Top-Performer BEIR-Benchmark (2023)
    # läuft vollständig auf CPU, keine API-Kosten
    model: "BAAI/bge-base-en-v1.5"
    batch_size: 32
    chroma_host: "http://localhost:8000"

  retrieval:
    top_k: 5
    similarity_cutoff: 0.5    # 0.5 statt 0.7: Multi-Hop braucht breiteres Netz

  ingestion:
    chunk_size: 300            # HotpotQA-Paragraphen sind kurz (~100 Wörter)
    chunk_overlap: 50

  graph:
    max_hops: 3                # HotpotQA braucht bis zu 3 Hops

  stores:
    bm25:   "data/stores/bm25.db"
    graph:  "data/stores/graph.json"
    vector: "vector_hotpotqa"  # ChromaDB collection name

  telemetry:
    output_dir: "evaluations/"
  ```

---

### 0.D — Infrastruktur (Docker)

> Drei Services: ChromaDB (Vektordatenbank), vllm-agent (Llama, Ingest + QA), vllm-judge (Qwen2.5, Evaluation).
> Zwei 8B-Modelle gleichzeitig: ~16 GB VRAM benötigt.

| Service | Image | Port | Zweck |
|---|---|---|---|
| `chromadb` | `chromadb/chroma` | 8000 | Vektordatenbank für Vector RAG |
| `vllm-agent` | `vllm/vllm-openai` | 8001 | Graph-Ingest + Agent (Llama-3.1-8B) |
| `vllm-judge` | `vllm/vllm-openai` | 8002 | LLM-as-Judge (Qwen2.5-7B) |

- [ ] **0.18** `.env` anlegen (`.gitignore` wurde bereits in 0.15 aktualisiert):
  ```bash
  HF_TOKEN=hf_...   # HuggingFace API Token
  ```

- [ ] **0.19** `docker-compose.yml` schreiben (komplettes File von Null):
  ```yaml
  services:
    chromadb:
      image: chromadb/chroma:0.6.3      # Version pinnen für Reproduzierbarkeit
      container_name: chromadb
      ports:
        - "8000:8000"
      volumes:
        - chroma_data:/chroma/chroma
      environment:
        - IS_PERSISTENT=TRUE            # Index überlebt Container-Restart
        - ANONYMIZED_TELEMETRY=FALSE

    vllm-agent:
      image: vllm/vllm-openai:latest
      container_name: vllm-agent
      runtime: nvidia
      environment:
        - NVIDIA_VISIBLE_DEVICES=all
        - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
      volumes:
        - vllm_agent_cache:/root/.cache/huggingface
      ports:
        - "8001:8000"
      command: >
        --model meta-llama/Llama-3.1-8B-Instruct
        --guided-decoding-backend outlines
        --max-model-len 4096
      deploy:
        resources:
          reservations:
            devices:
              - driver: nvidia
                count: 1
                capabilities: [gpu]

    vllm-judge:
      image: vllm/vllm-openai:latest
      container_name: vllm-judge
      runtime: nvidia
      environment:
        - NVIDIA_VISIBLE_DEVICES=all
        - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
      volumes:
        - vllm_judge_cache:/root/.cache/huggingface
      ports:
        - "8002:8000"
      command: >
        --model Qwen/Qwen2.5-7B-Instruct
        --guided-decoding-backend outlines
        --max-model-len 4096
      deploy:
        resources:
          reservations:
            devices:
              - driver: nvidia
                count: 1
                capabilities: [gpu]

  volumes:
    chroma_data:
    vllm_agent_cache:
    vllm_judge_cache:
  ```

---

### 0.E — Services starten & verifizieren

- [ ] **0.20** Alle Services starten:
  ```bash
  docker compose up -d chromadb    # zuerst ChromaDB (startet sofort)
  docker compose up -d vllm-agent  # Modell-Download ~8GB, dauert einige Minuten
  docker compose up -d vllm-judge  # Modell-Download ~8GB
  ```

- [ ] **0.21** Warten bis vLLM-Container bereit sind:
  ```bash
  docker logs vllm-agent --follow  # warten auf: "Application startup complete"
  docker logs vllm-judge --follow
  ```

- [ ] **0.22** ChromaDB prüfen:
  ```bash
  curl http://localhost:8000/api/v1/heartbeat
  # Erwartet: {"nanosecond heartbeat": ...}
  ```

- [ ] **0.23** vLLM-Container prüfen:
  ```bash
  curl http://localhost:8001/v1/models
  curl http://localhost:8002/v1/models
  # Erwartet: JSON mit dem jeweiligen Modellnamen
  ```

- [ ] **0.24** End-to-End Smoke-Test (Python):
  ```python
  import litellm

  # Agent-Container
  r = litellm.completion(
      model="openai/meta-llama/Llama-3.1-8B-Instruct",
      api_base="http://localhost:8001/v1",
      messages=[{"role": "user", "content": "Reply with one word: ready"}],
      max_tokens=5,
  )
  print("Agent:", r.choices[0].message.content)

  # Judge-Container mit Guided Decoding
  r = litellm.completion(
      model="openai/Qwen/Qwen2.5-7B-Instruct",
      api_base="http://localhost:8002/v1",
      messages=[{"role": "user", "content": "Is 2+2=4 correct?"}],
      max_tokens=5,
      extra_body={"guided_choice": ["CORRECT", "PARTIAL", "INCORRECT"]},
  )
  print("Judge:", r.choices[0].message.content)  # garantiert einer der drei Strings
  ```

---

## Phase 1 — Daten laden

> Ziel: Echte HotpotQA Multi-Hop-Daten im einheitlichen Format.

**Script:** `scripts/01_load_hotpotqa.py`

```
Aufruf:   python scripts/01_load_hotpotqa.py --n 300 --out data/hotpotqa.json
Ausgabe:  data/hotpotqa.json

Format:
{
  "meta": {
    "source": "hotpotqa/hotpot_qa distractor/train",
    "n_questions": 300,
    "question_types": {"bridge": 150, "comparison": 150},
    "n_documents": 2841,
    "seed": 42,
    "created_at": "2026-..."
  },
  "documents": [
    "Alexander Fleming. Fleming was born in Darvel, Scotland. He discovered penicillin.",
    "Penicillin. Penicillin is an antibiotic discovered in 1928.",
    ...     ← alle unique Paragraphen (Titel + Sätze), dedupliziert
  ],
  "questions": [
    {
      "id": "5a8b57f2...",
      "question": "In which country was the person born who discovered penicillin?",
      "answer": "Scotland",
      "type": "bridge",       ← bridge | comparison
      "level": "hard"         ← easy | medium | hard
    }
  ]
}
```

**Paragraph-Format:**  
`"Titel. Satz1 Satz2 Satz3"` — Titel vorne damit BM25 und Graph-Extraktion den Kontext haben.

**TODOs:**

- [ ] **1.1** `scripts/01_load_hotpotqa.py` schreiben:
  ```python
  # Kernlogik:
  ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")

  bridge     = [x for x in ds if x["type"] == "bridge"]
  comparison = [x for x in ds if x["type"] == "comparison"]

  rng  = random.Random(args.seed)
  half = args.n // 2
  selected = rng.sample(bridge, half) + rng.sample(comparison, args.n - half)

  docs, seen = [], set()
  for entry in selected:
      for title, sentences in entry["context"]:
          paragraph = title + ". " + " ".join(sentences)
          if paragraph not in seen:
              seen.add(paragraph)
              docs.append(paragraph)
  ```
- [ ] **1.3** Skript ausführen mit `--n 50` (Entwicklung), Output prüfen
- [ ] **1.4** Sicherstellen: `meta.n_documents` und Paragraph-Format sehen sinnvoll aus

---

## Phase 2 — Stores aufbauen

> Ziel: Ein Script baut den Store für eine Variante auf. Jede Variante ist unabhängig.

**Script:** `scripts/02_setup.py`

```
Aufruf:   python scripts/02_setup.py --variant bm25   --data data/hotpotqa.json
          python scripts/02_setup.py --variant vector  --data data/hotpotqa.json
          python scripts/02_setup.py --variant graph   --data data/hotpotqa.json
```

Intern ruft das Script die passende Memory-Klasse auf, resettet, und ingestiert alle Documents.  
Kein LangGraph, kein Factory-Pattern — direkte Instanziierung.

**Ingest-Metriken** (Ausgabe: `evaluations/<variant>_<ts>_setup.json`):

```json
{
  "variant":              "graph",
  "n_documents":          2841,
  "ingest_time_s":        3847.2,
  "ingest_tokens_prompt": 1204800,
  "ingest_tokens_completion": 89600,
  "ingest_tokens_total":  1294400
}
```

| Variante | Zeit | Tokens |
|---|---|---|
| bm25   | Wanduhr (SQLite-Index) | keine — kein LLM |
| vector | Wanduhr (SentenceTransformer lokal) | keine — kein LiteLLM-Call |
| graph  | Wanduhr + TelemetryTracker (phase=ingest, actor=graph_extract) | prompt + completion pro Dokument |

**TODOs:**

- [ ] **2.1** `scripts/02_setup.py` schreiben:
  - Config per `yaml.safe_load("src/config/unified_config.yaml")` laden
  - Minimal-Config-Objekte als Dataclasses (kein Pydantic):
    ```python
    from dataclasses import dataclass

    @dataclass
    class EmbeddingCfg:
        model: str
        batch_size: int
        chroma_host: str

    @dataclass
    class RetrievalCfg:
        top_k: int
        similarity_cutoff: float
    # usw.
    ```
  - Switch auf `--variant`:
    ```python
    if args.variant == "bm25":
        from src.memory.model_bm25 import BM25Memory
        memory = BM25Memory(top_k=cfg.retrieval.top_k,
                            storage_path=Path(cfg.stores.bm25))
    elif args.variant == "vector":
        from src.memory.model_vector import VectorMemory
        memory = VectorMemory(config=cfg)
    elif args.variant == "graph":
        from src.memory.model_graph import GraphMemory
        memory = GraphMemory(config=cfg)
    ```
  - `memory.reset()` → `memory.ingest_documents(data["documents"])`
  - Ingest-Zeit mit `time.perf_counter()` messen (Start vor reset, Stop nach ingest)
  - Nach Ingest `evaluations/<variant>_<ts>_setup.json` schreiben:
    ```python
    import time, json
    from src.telemetry.tracker import register_tracker

    tracker = register_tracker(output_dir=Path("evaluations"), variant_name=args.variant)
    # vector/bm25: tracker zählt 0 Calls — das ist korrekt und gewollt

    t0 = time.perf_counter()
    memory.reset()
    memory.ingest_documents(data["documents"])
    elapsed = time.perf_counter() - t0

    def _read_telemetry(path: Path) -> list[dict]:
        """Liest alle JSONL-Zeilen aus dem Telemetry-File des Trackers."""
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    setup_stats = {
        "variant":                   args.variant,
        "n_documents":               len(data["documents"]),
        "ingest_time_s":             round(elapsed, 2),
        # Tokens nur für graph != 0; bei bm25/vector = 0 (kein LLM-Call)
        "ingest_tokens_prompt":      sum(r["prompt_tokens"]     for r in _read_telemetry(tracker.telemetry_path)),
        "ingest_tokens_completion":  sum(r["completion_tokens"] for r in _read_telemetry(tracker.telemetry_path)),
    }
    (Path("evaluations") / f"{args.variant}_{ts}_setup.json").write_text(
        json.dumps(setup_stats, indent=2)
    )
    ```

- [ ] **2.2** `src/memory/model_graph.py` auf NetworkX umstellen:
  - `uv add networkx`
  - `self._nodes + self._edges` → `self._graph: nx.DiGraph`
  - `_add_triples()` → `G.add_edge(s, o, predicate=p)`
  - BFS → `nx.ego_graph(G, seed, radius=MAX_HOPS, undirected=True)`
  - Persistenz → `nx.node_link_data(G)` / `nx.node_link_graph(data)`
  - `reset()` → `self._graph = nx.DiGraph()`
  - `_MAX_HOPS = 3`
  - Triple-Extraktion via vLLM Guided Decoding — `guided_json` Schema:
    ```python
    TRIPLE_SCHEMA = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "subject":   {"type": "string"},
                "predicate": {"type": "string"},
                "object":    {"type": "string"},
            },
            "required": ["subject", "predicate", "object"],
            "additionalProperties": False,
        },
    }

    def _extract_triples(self, text: str, run_id: str) -> list[dict]:
        response = litellm.completion(
            model=self._cfg_ingest.model,
            api_base=self._cfg_ingest.base_url,
            messages=[
                {"role": "system", "content":
                    "Extract all entities and relationships from the text as triples. "
                    "Return a JSON array of {subject, predicate, object}."},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=self._cfg_ingest.max_tokens,
            extra_body={"guided_json": TRIPLE_SCHEMA},
            metadata={
                "phase": "ingest",
                "actor": "graph_extract",
                "variant_name": "graph",
                "run_id": run_id,
            },
        )
        return json.loads(response.choices[0].message.content)
    ```

- [ ] **2.3** Graph-Ingest parallelisieren (optional, wenn N > 200):
  ```python
  from concurrent.futures import ThreadPoolExecutor
  with ThreadPoolExecutor(max_workers=4) as pool:
      results = list(pool.map(self._extract_triples_tagged, documents))
  ```

- [ ] **2.4** Jede Variante einmal durchlaufen lassen (`--n 20` zum Testen)

- [ ] **2.5** Graph-Qualität stichprobenartig prüfen:
  ```python
  # In Python-Shell oder Notebook:
  import json, networkx as nx
  data = json.loads(open("data/stores/graph.json").read())
  G = nx.node_link_graph(data, directed=True)
  print(G.number_of_nodes(), G.number_of_edges())
  for s, o, d in list(G.edges(data=True))[:10]:
      print(f"  {s} --{d['predicate']}--> {o}")
  ```

---

## Phase 3 — Experiment ausführen

> Ziel: N Fragen durch eine Variante jagen, Antworten + Tokens speichern.

**Script:** `scripts/03_run.py`

```
Aufruf:   python scripts/03_run.py --variant bm25   --n 100
          python scripts/03_run.py --variant vector --n 100
          python scripts/03_run.py --variant graph  --n 100
          # Modell kommt aus unified_config.yaml — kein --model Flag nötig

Ausgabe:  evaluations/<variant>_<timestamp>_results.jsonl
          evaluations/<variant>_<timestamp>_telemetry.jsonl  ← automatisch vom tracker

Pro Zeile in results.jsonl:
{
  "run_id":          "a1b2c3d4",
  "variant":         "bm25",
  "question":        "In which country was the person born who discovered penicillin?",
  "expected":        "Scotland",
  "answer":          "Scotland, United Kingdom",
  "type":            "bridge",
  "context":         ["Fleming was born in Darvel...", "..."],
  "latency_ms":      1243.7,   ← Wanduhr Gesamtzeit (Retrieval + LLM-Call)
  "tokens_prompt":   1847,     ← aus LiteLLM response.usage
  "tokens_completion": 12,
  "ts":              "2026-06-01T14:23:01Z"
}

Hinweis: `latency_ms` misst die komplette Frage-Antwort-Zeit (Retrieval + LLM).
Der TelemetryTracker erfasst zusätzlich die reine LLM-Latenz pro Call — damit
lässt sich Retrieval-Overhead = latency_ms − llm_latency_ms isolieren.
```

**LLM-Aufruf — kein LangGraph, eine Funktion:**

```python
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using only the "
    "provided context. Be concise — one sentence or less. "
    "If the context does not contain the answer, say 'I don't know'."
)

import time
from dataclasses import dataclass

@dataclass
class AnswerResult:
    answer: str
    tokens_prompt: int
    tokens_completion: int
    latency_ms: float          # reine LLM-Latenz (ohne Retrieval)

def answer_question(question: str, context: list[str], model: str,
                    variant: str, run_id: str) -> AnswerResult:
    context_str = "\n".join(context) or "No context available."
    t0 = time.perf_counter()
    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"Context:\n{context_str}\n\nQuestion: {question}"},
        ],
        temperature=0.0,
        max_tokens=256,
        metadata={
            "phase": "agent_reasoning",
            "actor": "reasoning",
            "variant_name": variant,
            "run_id": run_id,
        },
    )
    llm_latency_ms = (time.perf_counter() - t0) * 1000
    usage = response.usage
    return AnswerResult(
        answer=response.choices[0].message.content or "",
        tokens_prompt=usage.prompt_tokens,
        tokens_completion=usage.completion_tokens,
        latency_ms=round(llm_latency_ms, 2),
    )

# In der Hauptschleife von 03_run.py:
# t_total_start = time.perf_counter()
# context = memory.search(question)          # Retrieval
# result  = answer_question(question, ...)   # LLM
# total_latency_ms = (time.perf_counter() - t_total_start) * 1000
# → total_latency_ms in results.jsonl schreiben
```

**TODOs:**

- [ ] **3.1** `scripts/03_run.py` schreiben (Struktur wie oben)
- [ ] **3.2** Telemetry-Tracker registrieren am Script-Start:
  ```python
  from src.telemetry.tracker import register_tracker
  register_tracker(output_dir=Path("evaluations"), variant_name=args.variant)
  ```
- [ ] **3.3** Testlauf: `--variant bm25 --n 10` — prüfen ob results.jsonl und telemetry.jsonl entstehen
- [ ] **3.4** Token-Zahlen in telemetry.jsonl manuell prüfen (sind sie > 0?)

---

## Phase 4 — Evaluation & Aggregation

> Ziel: Aus rohen Antworten Scores berechnen, alle Varianten vergleichen.

**Script:** `scripts/04_evaluate.py`

```
Aufruf:   python scripts/04_evaluate.py --results evaluations/bm25_*_results.jsonl
          python scripts/04_evaluate.py --all          ← alle results.jsonl in evaluations/

Ausgabe:  evaluations/<variant>_<ts>_scores.jsonl
          evaluations/summary_table.json

Pro Zeile in scores.jsonl:
{
  "run_id":        "a1b2c3d4",
  "variant":       "bm25",
  "question":      "...",
  "expected":      "Scotland",
  "answer":        "Scotland, United Kingdom",
  "type":          "bridge",
  "exact_match":   false,
  "f1":            0.67,
  "judge_verdict": "CORRECT"    ← nur für bridge wenn EM=false; null sonst
}
```

**Metrik-Logik:**

```
Für ALLE Fragen:
  → Exact Match (kostenlos)
  → F1 Token Overlap (kostenlos)

Für bridge-Fragen wo EM = false:
  → LLM-as-Judge (1 LLM-Call pro Frage, getaggt als phase="evaluation")

Für comparison-Fragen:
  → EM reicht (Antwort ist "yes" oder "no")
```

**Wissenschaftliche Begründung für LLM-as-Judge:**  
Zheng et al. (2023) "Judging LLM-as-a-Judge" (NeurIPS). HotpotQA-Antworten sind keine exakten Strings — "Scotland" und "Scotland, UK" sind beide korrekt. EM würde das als falsch werten. Limitation: gleicher Judge wie Agent → muss im Vortrag als Bias-Quelle benannt werden.

**Judge-Prompt:**
```
You are an evaluation judge for a question-answering benchmark.

Question:       {question}
Correct answer: {expected}
Model answer:   {answer}

Is the model's answer correct? Consider synonyms and partial phrasings.
Reply with exactly one word: CORRECT, PARTIAL, or INCORRECT.
```

**`summary_table.json` — Kernoutput für die Präsentation:**
```json
{
  "bm25": {
    "n": 300,
    "accuracy_em":               0.24,
    "accuracy_judge":            0.31,
    "f1_mean":                   0.38,
    "ingest_time_s":             4.1,
    "ingest_tokens_prompt":      0,
    "ingest_tokens_completion":  0,
    "query_latency_ms_mean":     2341,   ← Retrieval + LLM
    "query_tokens_prompt_mean":  1847,
    "query_tokens_completion_mean": 12,
    "query_tokens_total_mean":   1859
  },
  "vector": {
    "ingest_time_s":             312.4,  ← SentenceTransformer lokal, kein LLM
    "ingest_tokens_prompt":      0,
    "ingest_tokens_completion":  0,
    ...
  },
  "graph":  {
    "ingest_time_s":             3847.2,
    "ingest_tokens_prompt":      1204800,
    "ingest_tokens_completion":  89600,
    ...
  }
}
```

**TODOs:**

- [ ] **4.1** `scripts/04_evaluate.py` schreiben:
  - Exact Match + F1 für alle Einträge
  - LLM-Judge nur für `type=="bridge"` und `exact_match==false`
  - Telemetry-JSONL und Results-JSONL über `run_id` joinen für Token-Zahlen
  - `summary_table.json` schreiben

- [ ] **4.2** Mit `--no-judge` Flag testen (nur EM + F1, kein LLM-Call)

- [ ] **4.3** Judge-Qualität manuell auf 10 Beispielen prüfen:
  Stimmen die Verdicts mit eigenem Urteil überein?

---

## Phase 5 — Analyse & Visualisierung

> Ziel: Plots und Tabellen für die Präsentation.

### `notebooks/01_data_exploration.ipynb`
- [ ] **5.1** Verteilung Fragetypen (bridge/comparison) — Pie Chart
- [ ] **5.2** Verteilung Schwierigkeit (easy/medium/hard) — Bar Chart
- [ ] **5.3** Paragraph-Länge in Wörtern — Histogram (begründet chunk_size=300)
- [ ] **5.4** Anzahl Paragraphen pro Frage — zeigt Distractor-Dichte

### `notebooks/02_results_analysis.ipynb`
- [ ] **5.5** `summary_table.json` laden
- [ ] **5.6** **Tabelle**: Variante × Accuracy (EM / Judge) / F1 / Tokens / Latenz
- [ ] **5.7** **Plot 1 — Hauptergebnis**: Accuracy (Judge) vs. Tokens pro Query  
  Scatter-Plot, jeder Punkt = eine Variante, beschriftet  
  → zeigt den Trade-off visuell
- [ ] **5.8** **Plot 2 — Token-Breakdown**: gestapeltes Balkendiagramm  
  X-Achse: Variante, Y-Achse: Tokens  
  Segmente: Ingest-Tokens / Retrieval-Tokens / Reasoning-Tokens  
  → zeigt wo jede Architektur Tokens "ausgibt"
- [ ] **5.9** **Plot 3 — F1-Verteilung**: Boxplot pro Variante  
  → zeigt Varianz, nicht nur Mittelwert
- [ ] **5.10** Kernsatz formulieren:  
  *"[Variante X] erreicht [Y%] Accuracy bei [Z] Tokens pro Anfrage — [A]× effizienter als [Variante B] mit vergleichbarer Qualität."*

### `notebooks/03_graph_visualization.ipynb`
- [ ] **5.11** Graph laden: `nx.node_link_graph(json.loads(...))`
- [ ] **5.12** Subgraph für eine konkrete Beispielfrage plotten:
  ```python
  seed = "Alexander Fleming"
  sub  = nx.ego_graph(G, seed, radius=2, undirected=True)
  pos  = nx.spring_layout(sub, seed=42)
  edge_labels = {(s, o): d["predicate"] for s, o, d in sub.edges(data=True)}
  nx.draw_networkx(sub, pos, node_size=2000, font_size=8, arrows=True)
  nx.draw_networkx_edge_labels(sub, pos, edge_labels=edge_labels, font_size=7)
  plt.title("GraphRAG: Traversal für 'Where was Fleming born?'")
  plt.savefig("graph_example.png", dpi=150, bbox_inches="tight")
  ```
- [ ] **5.13** Graph-Statistiken ausgeben: Nodes, Edges, Avg. Degree

---

## Phase 6 — Präsentation

> Struktur orientiert sich am erwarteten Aufbau laut Prüfungsleistung (CRISP-DM).

```
Einleitung  (5 min)
  - Gruppenvorstellung
  - Forschungsfrage: Token-Effizienz vs. Qualität bei Multi-Hop-QA
  - Überblick: 3 Ansätze, 1 Datensatz, 1 Metrik

Theorie  (8 min)
  - Was ist RAG? Warum Multi-Hop schwierig?
  - BM25: keyword-basiert, kein LLM bei Retrieval
  - Vector RAG: semantische Suche via Embeddings
  - Graph RAG: Wissensgraph + BFS-Traversal
  - Einordnung in KI-Taxonomie (NLP / IR)
  - HotpotQA: Datensatz-Erklärung, bridge vs. comparison

Praxis — CRISP-DM  (12 min)
  - Business Understanding: Forschungsfrage operationalisiert
  - Data Understanding: EDA (Plots aus Notebook 01)
  - Data Preparation: Ingestion-Pipeline (02_setup.py), Graph-Extraktion
  - Modeling: 3 Retrieval-Strategien im Vergleich
  - Evaluation: LLM-as-Judge Begründung + Hybrid-Ansatz
  - Live-Demo: eine Multi-Hop-Frage live durch alle 3 Varianten

Ergebnisse  (3 min)
  - Tabelle + Plots (aus Notebook 02)
  - Kernsatz: wer gewinnt auf dem Accuracy/Token-Trade-off?

Abschluss  (2 min)
  - Limitations (Entity Normalization, Judge-Bias, lokales Modell)
  - Lessons Learned
  - Literatur
```

**TODOs:**

- [ ] **6.1** Slides bauen (Struktur oben als Gerüst)
- [ ] **6.2** Live-Demo vorbereiten:
  - Eine Beispielfrage wählen die alle 3 Ansätze unterschiedlich beantworten
  - `scripts/03_run.py --variant bm25 --n 1` live auf Folien zeigen
- [ ] **6.3** Alle Plots als PNG exportieren, in Slides einfügen
- [ ] **6.4** Limitations-Folie: Entity Normalization, lokales Modell via vLLM (Llama-3.1-8B / Qwen2.5-7B, kein GPT-4 o.ä.) → Ergebnisse nicht direkt mit kommerziellen Benchmarks vergleichbar
- [ ] **6.5** Literaturliste:
  - Yang et al. (2018) — HotpotQA Original-Paper
  - Lewis et al. (2020) — RAG (Original-Paper, Facebook/Meta)
  - Zheng et al. (2023) — LLM-as-Judge (NeurIPS)
  - Edge et al. (2024) — GraphRAG (Microsoft Research)
  - Robertson & Zaragoza (2009) — BM25 (Original)

---

## Abhängigkeiten

```
Phase 0 (Aufräumen)
    │
    ├── Phase 1 (Daten laden)          ──────────────────────┐
    │       │                                                 │
    │       └── Phase 2 (Stores aufbauen)                    │
    │                   │                                     │
    │                   └── Phase 3 (Experiment ausführen) ──┘
    │                               │
    │                               └── Phase 4 (Evaluation)
    │                                           │
    └───────────────────────────────────────────┴── Phase 5 (Analyse)
                                                        │
                                                    Phase 6 (Präsentation)
```

Phase 1 und Phase 2 (NetworkX-Refactor in model_graph.py) können parallel laufen.

---

## Parameter-Entscheidungen

| Parameter | Wert | Begründung |
|---|---|---|
| `--n` (Fragen) | 50 (Test) / 300 (Final) | 300 statistisch solide, mit vLLM realistisch |
| Fragetypen | 50/50 bridge/comparison | Ausgeglichener Benchmark |
| `max_hops` (Graph) | 3 | HotpotQA hat bis zu 3-Hop-Ketten |
| `top_k` (Retrieval) | 5 | Standard in RAG-Literatur |
| `similarity_cutoff` (Vector) | 0.5 | 0.7 zu restriktiv für Multi-Hop |
| LLM Agent | `Llama-3.1-8B-Instruct` · vLLM Port 8001 | vLLM Guided Decoding, hoher Durchsatz |
| LLM Ingest (Graph-Extraktion) | `Llama-3.1-8B-Instruct` · vLLM Port 8001 | gleicher Container wie Agent, sequenziell |
| LLM Judge | `Qwen2.5-7B-Instruct` · vLLM Port 8002 | andere Model-Family → kein Self-Enhancement-Bias |
| Graph-Ingest parallel | `max_workers=4` bei N>100 | Laufzeit N=300: ~4h statt ~15h |

---

## Aufwand-Schätzung

| Phase | Aufwand |
|---|---|
| 0 — Aufräumen | 1–2 Stunden |
| 1 — HotpotQA Loader | 2–3 Stunden |
| 2 — Setup Script + NetworkX | 4–6 Stunden |
| 3 — Run Script | 2–3 Stunden |
| 4 — Evaluate Script | 3–4 Stunden |
| 5 — Notebooks | 4–6 Stunden |
| 6 — Präsentation | offen |
| **Gesamt Code** | **~2 Tage** |
