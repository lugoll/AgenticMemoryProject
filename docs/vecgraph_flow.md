# VecGraph Memory — Architecture & Data Flow

`VecGraphMemory` (Model E) is a hybrid memory backend that combines a flat-file entity/relation ledger with a shared FAISS vector index.  It sits behind the `BaseMemory` interface, so the LangGraph agent never interacts with it directly.

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Phase A — Offline Ingestion                                         │
│                                                                      │
│  Raw document                                                        │
│      │                                                               │
│      ▼                                                               │
│  _chunk_text()  ──► overlapping word-boundary chunks                 │
│      │                                                               │
│      ▼  (per chunk)                                                  │
│  _extract()  ──► LiteLLM (phase=ingest, actor=graph_extract)         │
│      │           extracts {entities: [{name, fact}],                 │
│      │                     relations: [{source, target, description}]}│
│      │                                                               │
│      ▼                                                               │
│  _append_facts_and_relations()                                       │
│      │  1. normalize candidate entity name (lowercase+strip)         │
│      │  2. exact-key lookup in _entities dict                        │
│      │  3. alias lookup in _alias_to_key                             │
│      │  4. if still unknown → _find_similar_entity()  ◄─────┐       │
│      │        embeds candidate + all uncached entity names   │       │
│      │        (LiteLLM, phase=ingest, actor=vector_embed)    │       │
│      │        cosine similarity ≥ entity_name_dedup_threshold│       │
│      │            YES → merge: add alias, reuse entity        │       │
│      │            NO  → create new Entity, cache name vec    ─┘       │
│      │  5. guard: skip chunk if source_hash already in sources       │
│      │  6. canonicalize relation endpoints via _resolve_entity_key() │
│      │  7. append new relations (deduplicated by (src, tgt, desc))   │
│      │  → persist entities.json + relations.json                     │
│      │                                                               │
│      ▼  (after all chunks)                                           │
│  _save_and_rebuild_index()                                           │
│      │  1. collect all fact texts + relation descriptions            │
│      │  2. embed in batches (LiteLLM, phase=ingest, actor=vector_embed)│
│      │  3. L2-normalise vectors                                      │
│      │  4. per-entity cosine dedup of facts (≥ dedup_threshold)      │
│      │  5. prune entity.raw_facts to match surviving rows            │
│      │  6. build faiss.IndexFlatIP on kept vectors                   │
│      │  → persist faiss_vectors.npy + faiss_keys.json                │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  Phase B — Online Retrieval (agent calls SearchMemory)               │
│                                                                      │
│  query string                                                        │
│      │                                                               │
│      ▼                                                               │
│  embed + L2-normalise                                                │
│  (LiteLLM, phase=retrieval_overhead, actor=vector_embed)             │
│      │                                                               │
│      ▼                                                               │
│  FAISS IndexFlatIP.search(k=top_k)                                   │
│      │                                                               │
│      ├─ hit type "fact:entity_key:idx"                               │
│      │       → surface ALL facts of parent entity                    │
│      │       → add entity to matched_entity_keys                     │
│      │                                                               │
│      └─ hit type "rel:idx"                                           │
│              → surface relation description                          │
│              → resolve both endpoint names via _resolve_entity_key() │
│              → surface ALL facts of both endpoint entities           │
│                                                                      │
│      ▼  (graph expansion)                                            │
│  1-hop traversal: for every matched entity, walk all relations       │
│  whose source or target matches → add neighbour entities + relations │
│                                                                      │
│      ▼                                                               │
│  return deduplicated list of context strings (capped at top_k × 4)  │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  Phase B — Online Update (agent calls UpdateMemory)                  │
│                                                                      │
│  fact string  →  _extract()  →  _append_facts_and_relations()        │
│                  →  _save_and_rebuild_index()                        │
│  (same pipeline as ingest, tagged phase=agent_reasoning)             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Deduplication Layers

VecGraph applies deduplication at three distinct stages, each catching a different kind of redundancy.

### Layer 1 — Source-hash guard (chunk level)

**Where:** `_append_facts_and_relations`, before appending any facts.

**How:** Each text chunk is identified by the MD5 of its content.  An entity records every chunk hash it has seen in `entity.sources`.  If the same chunk arrives again (e.g. overlapping chunks or a re-run), its facts are silently skipped.

**Catches:** Exact repeated chunks.

---

### Layer 2 — Entity-name semantic deduplication (entity level)

**Where:** `_append_facts_and_relations`, when a new normalized key is not found in `_entities` or `_alias_to_key`.

**How:**

1. The candidate name (e.g. `"LLM"`) and all existing entity names that are not yet cached are embedded together in a single batched LiteLLM call.  Embeddings are L2-normalised and stored in `_name_vecs[entity_key]`.
2. Cosine similarities between the candidate and every cached entity name are computed.
3. If the best match exceeds `VecGraphConfig.entity_name_dedup_threshold` (default `0.88`), the candidate is treated as an alias of that entity:
   - The surface form is added to `entity.aliases`.
   - `_alias_to_key[normalized_candidate]` is set to the canonical entity key.
   - All incoming facts are appended to the existing entity.
4. Otherwise a new entity is created.

**Example:**

| Chunk A ingested first | Chunk B processed later |
|---|---|
| `"Large Language Model"` → Entity key `"large language model"` | `"LLM"` → normalized `"llm"`, not in `_entities` |
| | embed `"LLM"` vs `"Large Language Model"` → sim ≈ 0.94 > 0.88 |
| | merge: `entity.aliases = ["LLM"]`, `_alias_to_key["llm"] = "large language model"` |

Relation endpoints are canonicalized through `_resolve_entity_key()` so the graph always references the canonical entity regardless of which surface form appeared in a chunk.

**Catches:** Acronym/expansion pairs, spelling variants, paraphrases of the same concept.

---

### Layer 3 — Fact-level semantic deduplication (fact level, within entity)

**Where:** `_save_and_rebuild_index`, after all facts are embedded.

**How:** Within each entity, cosine similarities are computed between all pairs of fact embeddings.  If two facts share similarity above `VecGraphConfig.dedup_threshold` (default `0.92`), the later one is dropped.  `entity.raw_facts` is pruned in-place before the FAISS index is built.

**Catches:** Near-identical atomic statements about the same entity extracted from different (overlapping) chunks.

---

## Storage Layout

```
<memory_path>/
    entities.json       # {normalized_key: {name, raw_facts, sources, aliases}}
    relations.json      # [{source_entity, target_entity, description, sources}]
    faiss_vectors.npy   # float32 [N, D], L2-normalised, one row per surviving fact/relation
    faiss_keys.json     # ["fact:entity_key:idx", ..., "rel:idx", ...]
```

The FAISS index is never stored as a FAISS object — only the raw float32 matrix is persisted, keeping `FilesystemStorage` free of the `faiss` dependency.  On startup `_load_state` rebuilds the in-memory `IndexFlatIP` from the stored vectors.

---

## Telemetry Tags

Every LiteLLM call is tagged for the scientific benchmarking pipeline:

| Call site | `phase` | `actor` |
|---|---|---|
| `_extract` during ingest | `ingest` | `graph_extract` |
| `_embed` during ingest | `ingest` | `vector_embed` |
| `_embed` for entity-name dedup (ingest) | `ingest` | `vector_embed` |
| `_embed` during `search` | `retrieval_overhead` | `vector_embed` |
| `_extract` during `update_fact` | `agent_reasoning` | `graph_extract` |
| `_embed` during `update_fact` | `agent_reasoning` | `vector_embed` |

---

## Configuration Reference

All parameters live in `VecGraphConfig` (unified_config.yaml → `vecgraph:` block):

| Parameter | Default | Effect |
|---|---|---|
| `embedding_dim` | `768` | Must match the chosen embedding model's output size |
| `extraction_max_tokens` | `1024` | Max tokens for the entity/relation extraction LLM call |
| `dedup_threshold` | `0.92` | Cosine similarity above which two **facts** within the same entity are considered duplicates |
| `entity_name_dedup_threshold` | `0.88` | Cosine similarity above which two **entity names** refer to the same concept |

Set `entity_name_dedup_threshold: 0.0` to disable entity-name semantic dedup entirely (exact-match only).
