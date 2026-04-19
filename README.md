# AgenticMemoryProject

Scientific benchmarking framework comparing three RAG retrieval architectures on token efficiency and answer quality for multi-hop questions.

**Research question:** Which retrieval architecture delivers the best quality-to-token-cost ratio for multi-hop questions?

| Model | Architecture | Role |
|---|---|---|
| A | BM25 (keyword search) | Baseline |
| B | Vector RAG (semantic embeddings) | Industry standard |
| C | GraphRAG (knowledge graph + vector) | Innovative approach |

Evaluated on HotpotQA (distractor split) measuring Exact Match, F1, and input tokens per query.

---

## Quick Start (Docker)

**Prerequisites:** Docker Desktop running.

```bash
# 1. Build the application image
docker compose build

# 2. Start Ollama and pull the LLM (one-time, ~2 GB)
docker compose up -d ollama
docker compose exec ollama ollama pull llama3.2:3b

# 3. Run a variant (ingest documents, then answer questions)
docker compose run --rm app --variant bm25_baseline --ingest --qa
```

Results are written to `evaluations/` on your host machine.

---

## Project Structure

```
AgenticMemoryProject/
├── main.py                          # CLI entry point
├── Dockerfile                       # Python 3.14-slim + uv
├── docker-compose.yml               # ollama service + app service
│
├── src/
│   ├── config/
│   │   ├── unified_config.yaml      # Single source of truth for all experiment params
│   │   └── settings.py              # Pydantic models, load_config()
│   │
│   ├── memory/                      # Memory backends — all implement BaseMemory
│   │   ├── base.py                  # Abstract interface (search, ingest, update, reset)
│   │   ├── model_dummy.py           # DummyMemory — substring scoring, for smoke tests
│   │   ├── model_bm25.py            # BM25Memory  — SQLite FTS5, no LLM (Model A)
│   │   ├── model_graph.py           # GraphMemory — LLM entity extraction + BFS (Model C)
│   │   └── model_vector.py          # VectorMemory — ChromaDB + embeddings (Model B) [TODO]
│   │
│   ├── agent/                       # LangGraph agents — all extend BaseAgent
│   │   ├── base.py                  # Concrete agent: retrieve → reason graph
│   │   ├── agent_dummy.py           # Wires DummyMemory
│   │   ├── agent_bm25.py            # Wires BM25Memory
│   │   ├── agent_graph.py           # Wires GraphMemory
│   │   └── factory.py               # build_agent(config) — single dispatch point
│   │
│   ├── pipelines/
│   │   ├── ingest.py                # Phase A: load dataset → reset → ingest_documents
│   │   ├── qa.py                    # Phase B: load questions → agent.run() → write JSONL
│   │   └── evaluate.py              # Phase C: score results JSONL → EM, F1, token cost [TODO]
│   │
│   └── telemetry/
│       └── tracker.py               # LiteLLM callback → telemetry JSONL
│
├── data/
│   ├── raw/                         # Input datasets (documents + questions, JSON)
│   └── processed/                   # Persisted memory stores (SQLite, ChromaDB, JSON graph)
│
├── evaluations/                     # Pipeline outputs — never modified by src/ code
│   ├── <variant>_<ts>_results.jsonl # One record per question, keyed by run_id
│   └── <variant>_<ts>_telemetry.jsonl  # One record per LLM call, keyed by run_id
│
├── notebooks/                       # EDA and visualization — never imported by pipelines
└── tests/
    ├── unit/                        # No LLM, no network (default pytest suite)
    └── integration/                 # Requires live Ollama; run with -m integration
```

---

## Docker Setup

### Services

**`ollama`** — shared LLM backend. All experiment variants (BM25, Vector, Graph) route their reasoning calls here. Runs persistently in the background; models are stored in the `ollama_models` Docker volume.

**`app`** — the Python pipeline. Runs `main.py` for a single command and exits. Depends on `ollama` being healthy before starting.

### Volumes

| Volume | Contents | Persists across |
|---|---|---|
| `ollama_models` | Downloaded LLM weights | Container restarts, rebuilds |
| `huggingface_cache` | Embedding model weights (~400 MB) | Container restarts, rebuilds |
| `./data` (bind) | Raw datasets + processed memory stores | Always on host |
| `./evaluations` (bind) | QA results + telemetry | Always on host |

### Common commands

```bash
# Start Ollama in the background
docker compose up -d ollama

# Pull a different LLM (default is llama3.2:3b, set in unified_config.yaml)
docker compose exec ollama ollama pull llama3.1:8b

# List models available in Ollama
docker compose exec ollama ollama list

# Run only the ingestion step
docker compose run --rm app --variant bm25_baseline --ingest

# Run only the QA step (requires prior ingestion)
docker compose run --rm app --variant bm25_baseline --qa

# Run all variants sequentially
docker compose run --rm app --variant all --ingest --qa

# Stop Ollama (volumes are preserved)
docker compose down

# Stop Ollama and delete all volumes (LLM weights will need re-downloading)
docker compose down -v
```

### GPU acceleration (optional)

Uncomment the `deploy` block in `docker-compose.yml` under the `ollama` service:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

### Local development (without Docker)

```bash
uv sync
uv run pytest                          # unit tests
uv run python main.py --variant bm25_baseline --ingest --qa
```

Requires a local Ollama instance running on `localhost:11434`. The `OLLAMA_API_BASE` environment variable overrides the URL if needed.

---

## Running the Pipeline

### Parameters

All pipeline steps go through `main.py`. The full signature:

```
docker compose run --rm app --variant NAME [--ingest] [--qa]
                             [--data-dir PATH] [--output-dir PATH] [--config PATH]
```

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--variant NAME` | yes | — | Variant name from `unified_config.yaml`, or `all` to run every variant |
| `--ingest` | at least one | — | Run Phase A: load dataset → reset memory → ingest documents |
| `--qa` | at least one | — | Run Phase B: load questions → agent.run() → write results JSONL |
| `--data-dir PATH` | no | `data/raw` | Directory containing raw dataset JSON files |
| `--output-dir PATH` | no | `evaluations` | Directory where results and telemetry are written |
| `--config PATH` | no | `src/config/unified_config.yaml` | Path to a custom config YAML |

`--ingest` and `--qa` can be used individually or together. At least one must be passed.

### What happens step by step

Using `docker compose run --rm app --variant bm25_baseline --ingest --qa` as example:

```
docker compose run --rm app --variant bm25_baseline --ingest --qa
│
├── Docker starts the app container (waits for ollama to be healthy)
│   └── sets OLLAMA_API_BASE=http://ollama:11434
│
├── main.py
│   ├── load_config("bm25_baseline")
│   │   └── reads unified_config.yaml, deep-merges variant onto defaults
│   │       → ExperimentConfig(agent_type="bm25", data_file="baseline_dummy.json", ...)
│   │
│   ├── register_tracker()
│   │   └── registers TelemetryTracker as LiteLLM callback
│   │       → opens evaluations/bm25_baseline_<ts>_telemetry.jsonl
│   │
│   ├── --ingest → run_ingest(config)
│   │   ├── reads data/raw/baseline_dummy.json  → list of document strings
│   │   ├── build_agent(config)  → BM25Agent  → BM25Memory(storage_path=data/processed/bm25_baseline.db)
│   │   ├── memory.reset()       → drops + recreates SQLite FTS5 table
│   │   └── memory.ingest_documents([...])  → INSERT into SQLite (no LLM calls)
│   │
│   └── --qa → run_qa(config)
│       ├── reads data/raw/baseline_dummy.json  → list of questions
│       ├── build_agent(config)  → BM25Agent  → opens same SQLite file
│       ├── opens evaluations/bm25_baseline_<ts>_results.jsonl
│       │
│       └── for each question:
│           └── agent.run(question)
│               ├── retrieve_node
│               │   └── memory.search(question)
│               │       └── SQLite FTS5 MATCH query, BM25 ranked, returns top 5 passages
│               │           (no LLM call)
│               │
│               └── reason_node
│                   └── litellm.completion(model="ollama/llama3.2:3b", ...)
│                       ├── sends: system prompt + context passages + question
│                       ├── TelemetryTracker logs: tokens, latency → telemetry.jsonl
│                       └── returns: answer string → results.jsonl
```

### Commands

```bash
# Ingest + QA in one step
docker compose run --rm app --variant bm25_baseline --ingest --qa

# Ingest only (writes to data/processed/)
docker compose run --rm app --variant bm25_baseline --ingest

# QA only (requires prior ingestion, writes to evaluations/)
docker compose run --rm app --variant bm25_baseline --qa

# All variants sequentially
docker compose run --rm app --variant all --ingest --qa
```

### Output files

After `--qa`, two files appear in `evaluations/`:

**`<variant>_<timestamp>_results.jsonl`** — one record per question:
```json
{
  "run_id": "1fe02ca4",
  "variant_name": "bm25_baseline",
  "agent_type": "bm25",
  "question": "Who built the Eiffel Tower?",
  "expected": "Gustave Eiffel",
  "answer": "Gustave Eiffel.",
  "context_used": ["The Eiffel Tower is located in Paris..."],
  "steps_taken": ["retrieve", "reason"],
  "backend_name": "bm25"
}
```

**`<variant>_<timestamp>_telemetry.jsonl`** — one record per LLM call:
```json
{
  "event": "llm_call",
  "run_id": "1fe02ca4",
  "phase": "agent_reasoning",
  "actor": "langgraph_node",
  "variant_name": "bm25_baseline",
  "model": "ollama/llama3.2:3b",
  "prompt_tokens": 312,
  "completion_tokens": 48,
  "total_tokens": 360,
  "latency_ms": 843.2
}
```

Join on `run_id` to attribute token costs to individual questions.

---

## Configuration

All experiment parameters live in `src/config/unified_config.yaml`. Variants carry only overrides; the loader deep-merges them with `default`.

```yaml
default:
  llm:
    model: "ollama/llama3.2:3b"   # always "provider/model" format
    temperature: 0.0
    max_tokens: 2048
  retrieval:
    top_k: 5
  ingestion:
    chunk_size: 300
    chunk_overlap: 50

variants:
  bm25_baseline:
    agent_type: "bm25"
    data_file: "baseline_dummy.json"
    memory_path: "data/processed/bm25_baseline.db"
```

**Validation rules:**
- Model names must use `provider/model` format — `ollama/llama3.2:3b`, `openai/gpt-4o`. A bare name raises `ValidationError`.
- `chunk_overlap` must be strictly less than `chunk_size`.
- `agent_type` must be one of `dummy`, `bm25`, `vector`, `graph`.

### Adding a new variant

Add a block under `variants:` with only the fields that differ from `default`:

```yaml
variants:
  vector_hotpotqa:
    agent_type: "vector"
    data_file: "hotpotqa_distractor.json"
    memory_path: "data/processed/vector_hotpotqa"
    ingestion:
      chunk_size: 300
      chunk_overlap: 50
```

Then run:
```bash
docker compose run --rm app --variant vector_hotpotqa --ingest --qa
```

---

## Architecture

### Two-phase separation

The system has two strictly separated phases. Ingestion is offline and stateless. The agent only runs at test time and never triggers re-ingestion.

```
Phase A — Offline Ingestion
  Dataset (JSON) → memory.reset() → memory.ingest_documents() → persisted store

Phase B — Agentic QA
  Question → agent.run() → [retrieve_node → reason_node] → answer + telemetry
```

This separation is intentional: GraphRAG ingestion makes many LLM calls for entity extraction. If ingestion ran at test time, those token costs would contaminate the per-question measurements.

### Memory abstraction

The agent never knows which backend it is querying. It only calls two methods:

```python
memory.search(query)        # retrieve relevant context
memory.update_fact(fact)    # inject a newly learned fact
```

`build_agent(config)` in `factory.py` is the only place that maps `agent_type` → concrete implementation.

```
agent_type = "bm25"   →  BM25Memory   (SQLite FTS5, no LLM)
agent_type = "vector" →  VectorMemory (ChromaDB + BAAI/bge-base-en-v1.5)
agent_type = "graph"  →  GraphMemory  (LLM entity extraction + BFS traversal)
```

### LangGraph graph topology

```
START → retrieve_node → reason_node → END
```

`retrieve_node` calls `memory.search(question)` and stores the results in state.
`reason_node` calls the LLM via LiteLLM with the retrieved context and returns the answer.

Every LLM call is tagged with `phase` and `actor` so token costs can be attributed:

| phase | actor | When |
|---|---|---|
| `ingest` | `graph_extract` | GraphRAG entity extraction during ingestion |
| `agent_reasoning` | `langgraph_node` | Answer generation for all variants |
| `ingest` | `vector_embed` | Embedding calls during Vector RAG ingestion |
| `evaluation` | `llm_as_judge` | LLM-based evaluation scoring (planned) |

### Class overview

```
BaseMemory (ABC)
├── DummyMemory      — substring scoring, file-backed JSON (smoke tests)
├── BM25Memory       — SQLite FTS5, BM25 ranking, no LLM
├── VectorMemory     — ChromaDB, sentence-transformers embeddings [TODO]
└── GraphMemory      — JSON graph, LLM entity extraction, BFS retrieval

BaseAgent (LangGraph)
├── DummyAgent       → DummyMemory
├── BM25Agent        → BM25Memory
├── VectorAgent      → VectorMemory [TODO]
└── GraphAgent       → GraphMemory
```

---

## Telemetry

Every LLM call is intercepted by `TelemetryTracker` (a LiteLLM `CustomLogger`) and written to a JSONL file. The tracker warns on any call missing `phase` or `actor` tags.

**Joining results with token costs:**

```python
import json
import pandas as pd

tel = pd.DataFrame(json.loads(l) for l in open("evaluations/bm25_baseline_..._telemetry.jsonl"))
res = pd.DataFrame(json.loads(l) for l in open("evaluations/bm25_baseline_..._results.jsonl"))

merged = res.merge(tel[["run_id", "total_tokens", "latency_ms"]], on="run_id")
print(merged[["question", "answer", "total_tokens"]].to_string())
```

---

## Adding a new memory backend

1. Create `src/memory/model_<name>.py`, subclass `BaseMemory`, implement all four methods.
2. Tag every internal LLM call with the correct `phase` / `actor` via the `metadata` argument to `litellm.completion()`.
3. Export from `src/memory/__init__.py`.
4. Add to `src/agent/factory.py`:
   ```python
   if config.agent_type == "vector":
       from src.memory.model_vector import VectorMemory
       return VectorAgent(memory=VectorMemory(config), config=config)
   ```
5. Add the variant to `unified_config.yaml`.
6. Add unit tests in `tests/unit/test_<name>_memory.py`.

---

## Tests

```bash
# Unit tests (no LLM required, default suite)
uv run pytest -v

# Integration tests (requires Ollama running with llama3.2:3b)
uv run pytest -m integration -v

# With coverage
uv run pytest --cov=src --cov-report=term-missing
```
