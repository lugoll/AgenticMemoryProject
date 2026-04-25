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

## Quick Start

**Prerequisite:** Docker Desktop running.

```bash
# 1. Build the app image
docker compose build

# 2. Start infrastructure
docker compose up -d ollama chromadb
docker compose exec ollama ollama pull llama3.2:3b

# 3. Download dataset (one-time, ~560 MB)
docker compose run --rm app python scripts/prepare_hotpotqa.py

# 4. Run an experiment
docker compose run --rm app --variant bm25_baseline --ingest --qa
```

Results are written to `evaluations/` on your host machine.

---

## Running Experiments

All experiments run via `docker compose run --rm app --variant NAME [--ingest] [--qa]`.

```bash
# Model A — BM25 (no vector store needed)
docker compose run --rm app --variant bm25_baseline --ingest --qa

# Model B — Vector RAG (requires chromadb running)
docker compose run --rm app --variant vector_rag --ingest --qa

# Model C — GraphRAG (slow ingestion ~30–60 min, LLM entity extraction)
docker compose run --rm app --variant graph_v1 --ingest --qa

# All variants sequentially
docker compose run --rm app --variant all --ingest --qa
```

See [`docs/vector_rag.md`](docs/vector_rag.md) for the full Vector RAG setup guide.

---

## Project Structure

```
AgenticMemoryProject/
├── src/
│   ├── config/          # unified_config.yaml — all experiment parameters
│   ├── memory/          # BaseMemory + BM25 / Vector / Graph backends
│   ├── agent/           # LangGraph agents + factory
│   ├── pipelines/       # ingest.py, qa.py, evaluate.py
│   └── telemetry/       # LiteLLM callback → JSONL
├── scripts/
│   └── prepare_hotpotqa.py
├── data/raw/            # Input datasets
├── evaluations/         # Results + telemetry (JSONL, written by pipelines)
├── docs/                # Per-variant setup guides
└── tests/               # Unit tests (run locally, no Docker needed)
```

---

## Tests

```bash
uv sync --extra dev
uv run pytest -v
```
