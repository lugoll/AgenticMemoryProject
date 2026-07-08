# AgenticMemoryProject

Scientific benchmark framework comparing RAG retrieval architectures on token
efficiency and answer quality for multi-hop questions (HotpotQA).

**Research question:** Which RAG approach achieves the highest answer quality on
multi-hop questions — and how does that trade off against token consumption?

All variants read from **one shared Neo4j store** (unified ingest): chunks are
stored, embedded and full-text indexed once, and the knowledge graph is
extracted once via LLM. The variants differ *only* in retrieval, so they all
operate on identical data and the ingest runs exactly
once.

---

## The retrieval variants

Seven retrieval views sit on top of the same store. None of them makes an LLM
call at retrieval time — entity linking, graph traversal, embedding and reranking
all run locally on CPU/GPU, which preserves the zero-cost-retrieval property that
distinguishes these approaches from test-time-heavy Vector RAG. The only LLM cost
in the whole pipeline is (a) the one-off graph extraction during ingest and
(b) the agent's reasoning at test time.

The graph variants are best understood along three axes: **how the graph is
seeded**, **how it is traversed**, and **what it returns**.

| Variant | Seeding | Traversal | Returns |
|---|---|---|---|
| `bm25` | — | — | chunk passages |
| `vector` | — | — | chunk passages |
| `graph` | BM25 entity linking | flat hop-by-hop BFS | triples |
| `graphtext` | BM25 entity linking | flat hop-by-hop BFS | chunk passages |
| `vectorgraph` | chunk-vector cosine | beam chain-scoring | triples |
| `vectorgraphtext` | chunk-vector cosine | flat hop-by-hop BFS | chunk passages |

### `bm25`

Lucene full-text index (BM25-family scoring) over the `:Chunk` nodes. No
embeddings, no LLM. The query is sanitised into a Lucene OR-query and the top
`retrieval.top_k` chunk texts are returned. This is the classic sparse-retrieval
baseline.

### `vector`

Neo4j's native vector index (cosine) over `Chunk.embedding`. The query is
embedded locally with the shared sentence-transformer (`BAAI/bge-base-en-v1.5`,
no API call), matched against the index, and chunks above
`retrieval.similarity_cutoff` are returned. The dense-retrieval baseline.

### `graph`

Knowledge-graph retrieval that returns **triples**. It maps the query to graph
nodes with **BM25 entity linking** — an in-memory BM25 index over entity names,
so query tokens pick seed entities without any LLM call (stopwords are filtered
so e.g. "who" doesn't match the band node "The Who"). From those seeds it runs a
**flat hop-by-hop BFS** over the `:__Entity__` subgraph (`BaseMemory._expand_triples`):
the frontier starts as all seeds and each hop is one batched Cypher query, so
1-hop edges are emitted before 2-hop edges and survive truncation. The collected
triples are cross-encoder reranked **as flat single triples** to `top_k`. Output
looks like `"Alan Turing devised the Turing test"`.

### `graphtext`

Identical seeding and traversal to `graph` (BM25 entity linking → flat BFS), but
instead of the triples it returns the **source chunks** those visited entities
were extracted from (`BaseMemory._collect_chunks`, ranked by how many traversed
entities each chunk mentions, so multi-hop bridge passages rank highest), then
cross-encoder reranks the chunks to `top_k`. It exists to isolate the *output
representation*: `graph` vs `graphtext` differ only in triples-vs-passages,
everything upstream is shared.

### `vectorgraph`

Chunk-anchored graph retrieval that returns **triples** (comparable to `graph`),
but with a different seeding and traversal. Entity-name vectors turned out to be
a weak anchor (a nickname rarely matches the embedding of the bare entity name),
so seeds are found *through the chunks*: the query is embedded and matched against
the `chunk_vector` index (the same signal `vector` uses), then
`(:Chunk)-[:MENTIONS]->(:__Entity__)` hops to the seed entities. From there it
grows reasoning chains via a **beam traversal** (`src/memory/traversal.py`):
whole chains are scored, a per-node-capped top-K beam is kept, and the top
chains' triples are returned (no separate final rerank). The beam won out over
flat BFS specifically for `vectorgraph`'s noisier chunk-anchored seeds.

### `vectorgraphtext`

Chunk-vector seeded like `vectorgraph`, but returns **source chunks** like
`graphtext`.

1. Embed the query, match `chunk_vector` — keep both the hop-0 chunk texts and
   their `MENTIONS` entities as seeds.
2. Flat hop-by-hop BFS over the entity subgraph → visited entity set.
3. Collect **bridge chunks** that mention the visited entities (the passages pure
   cosine similarity cannot reach).
4. Deduplicate hop-0 + bridge chunks and cross-encoder rerank to `top_k`.

### How the four graph variants differ at a glance

|  | **returns triples** | **returns passages** |
|---|---|---|
| **BM25-seeded** | `graph` | `graphtext` |
| **vector-seeded** | `vectorgraph` | `vectorgraphtext` |

Traversal is flat BFS + single-item rerank everywhere except `vectorgraph`, which
uses beam chain-scoring.

---

## Cost attribution

The unified ingest does two kinds of work, and its costs split cleanly by which
variants consume them:

- **Graph extraction (LLM):** the expensive part — the one-off triple extraction
  is the *only* ingest cost that spends LLM tokens, and it is attributable
  entirely to the **graph variants** (`graph`, `graphtext`, `vectorgraph`,
  `vectorgraphtext`). The `bm25` and `vector` baselines never touch the graph.
- **Chunk embedding (local):** the only ingest cost of the vector side is the
  local embedding pass (sentence-transformers, on GPU when available, else CPU).
  It spends no LLM tokens and is not actively measured as a cost — it consumes
  some local compute and wall-clock time, but on a different order of magnitude
  from LLM inference.

The per-run split is written to the setup report (`graph_extract_s` = graph
share, `chunk_embed_s` = vector/bm25 share).

---

## Hardware limits

This framework was built to run on a single machine with a **16 GB VRAM** GPU,
and several design choices exist purely to fit that budget:

- **The agent and judge models never run concurrently.** Agent (`llama3.1:8b`,
  ~5 GB) and judge (`qwen2.5:14b`, ~9 GB) would need ~14 GB together; they run
  sequentially instead.
- **Do not run `03_run.py` and `04_evaluate.py` at the same time** in two
  terminals — that loads both models at once and is tight at 16 GB.
- **Local Torch models** (the `bge-base` embedder and the `bge-reranker-base`
  cross-encoder) run on `config.device` (`auto` → CUDA when available, else CPU).
  They spend no LLM tokens; the device knob only affects local-inference latency.
- **Ingest is the slow, LLM-bound step:** several hours at N=500 (extraction
  dominates; chunk embedding is 1–2 minutes). It is checkpointed — an interrupted
  run resumes from the last checkpoint in Neo4j via `--resume`.

---

## One-time setup

### Step 1 — Prerequisites

- Docker Desktop running (with GPU support for Ollama)
- `uv` installed: `uv --version`
- `git` installed: `git --version`
- Recommended VRAM: ≥ 16 GB (see [Hardware limits](#hardware-limits))

### Step 2 — Clone & install dependencies

```bash
git clone <repo-url>
cd AgenticMemoryProject
uv sync
```

### Step 3 — Start the Docker containers

```bash
docker compose up -d
docker compose ps
# Expected: neo4j, ollama-agent, ollama-judge all "running"
```

### Step 4 — Pull the LLM models (one-time, ~14 GB total)

```bash
docker exec ollama-agent ollama pull llama3.1:8b
docker exec ollama-judge ollama pull qwen2.5:14b
```

> Downloads can take 5–15 minutes depending on your connection.

### Step 5 — Load the dataset

```bash
uv run python scripts/01_load_hotpotqa.py --n 500 --out data/hotpotqa.json
```

Output: `data/hotpotqa.json` with 500 questions (250 bridge, 250 comparison) and
~3,500 documents.

### Step 6 — Build the unified store (one run for all variants)

```bash
uv run python scripts/02_setup.py --data data/hotpotqa.json
```

The ingest writes into a single Neo4j database:

1. **Chunks** (300 words, 50 overlap) with local embeddings
   (`BAAI/bge-base-en-v1.5`, CPU)
2. **Full-text index** (Lucene/BM25) and **vector index** (cosine) over the chunks
3. **Knowledge graph**: LLM triple extraction (LlamaIndex + JSON schema, predicate
   whitelist)

> Several hours at N=500 (see [Hardware limits](#hardware-limits)). Interrupting
> is safe — `--resume` continues from the last checkpoint (stored in Neo4j).

---

## Running the experiment

### Step 7 — Run the variants

```bash
uv run python scripts/03_run.py --variant bm25            --n 500
uv run python scripts/03_run.py --variant vector          --n 500
uv run python scripts/03_run.py --variant graph           --n 500
uv run python scripts/03_run.py --variant graphtext       --n 500
uv run python scripts/03_run.py --variant vectorgraph     --n 500
uv run python scripts/03_run.py --variant vectorgraphtext --n 500
```

> No further ingest needed — every variant reads from the unified store. Omitting
> `--variant` runs all variants sequentially.
> Output per variant: `evaluations/<variant>_<timestamp>_results.jsonl`

### Step 8 — Evaluate

```bash
uv run python scripts/04_evaluate.py --all
```

> ~10–20 min (the LLM judge runs for bridge questions with EM=false).
> Output: `evaluations/<variant>_<timestamp>_scores.jsonl` + `evaluations/summary_table.json`

### Step 9 — Inspect results

```bash
cat evaluations/summary_table.json          # quick overview
cat evaluations/graph_*_scores.jsonl | head -20   # detailed per-variant scores
```

---

## The unified store

Everything lives in the Neo4j Docker volume (`neo4j_data`) — no store files in the
repo:

| Content | Schema | Used by |
|---|---|---|
| Chunks | `(:Chunk {text, embedding})` | bm25, vector |
| Full-text index | `chunk_fulltext` (Lucene/BM25 over `Chunk.text`) | bm25 |
| Vector index | `chunk_vector` (cosine over `Chunk.embedding`) | vector, vectorgraph(text) |
| Knowledge graph | `(:__Entity__ {name})-[PREDICATE]->(:__Entity__)` + `MENTIONS` from chunks | graph, graphtext, vectorgraph, vectorgraphtext |
| Checkpoint | `(:Meta {key: 'ingest_checkpoint'})` | `02_setup --resume` |

Browser UI for inspection: <http://localhost:7474> (neo4j / password).

---

## Configuration

All parameters live in a single file: `src/config/unified_config.yaml`

```yaml
llm:
  agent:   ollama/llama3.1:8b  @ localhost:11434  # QA agent
  ingest:  ollama/llama3.1:8b  @ localhost:11434  # graph triple extraction
  judge:   ollama/qwen2.5:14b  @ localhost:11435  # LLM-as-judge (evaluation)

device: auto                 # local Torch models (embedder + reranker): cuda if available

retrieval:
  top_k:             5       # shared final-context knob for ALL variants
  similarity_cutoff: 0.4     # vector cosine cutoff
  rerank_fetch_k:    60      # candidate pool before the cross-encoder rerank
  hop0_fetch_k:      5       # vectorgraphtext: hop-0 chunk-vector pool

graph:
  max_hops:    2             # BFS depth (2 > 3 on HotpotQA per sweep)
  seed_top_k:  10            # BM25 entity-linking seeds (graph + graphtext)
  beam_width:  10            # vectorgraph: chains per hop
  max_per_tail: 3            # vectorgraph: chains per tail node per hop

stores:
  neo4j: bolt://localhost:7687   # the one store for all variants
```

---

## Project structure

```
AgenticMemoryProject/
├── src/
│   ├── config/        unified_config.yaml + loader (cfg.py)
│   ├── memory/        BaseMemory (Neo4j + unified ingest) + retrieval views
│   │                  (bm25, vector, graph, graphtext,
│   │                   vectorgraph, vectorgraphtext) + extraction.py + traversal.py
│   └── telemetry/     LiteLLM callback → JSONL token tracking
├── scripts/
│   ├── 01_load_hotpotqa.py    HuggingFace → data/hotpotqa.json
│   ├── 02_setup.py            unified ingest (once for all variants)
│   ├── 03_run.py              run the experiment (--variant ...)
│   └── 04_evaluate.py         EM + F1 + LLM judge + summary_table.json
├── data/
│   └── hotpotqa.json          dataset (not in git; produced by 01_load)
├── evaluations/               run outputs (not in git)
└── docs/
    └── graphrag_analysis.md   analysis: triple extraction vs. HotpotQA
```

---

## Further documentation

| Document | Content |
|---|---|
| [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) | Full experiment plan, all architecture decisions |
| [`docs/graphrag_analysis.md`](docs/graphrag_analysis.md) | Why GraphRAG is structurally disadvantaged on HotpotQA |
