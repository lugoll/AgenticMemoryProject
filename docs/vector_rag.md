# Vector RAG (Modell B) — Setup und Durchführung

Vollständige Anleitung für das Vector RAG Experiment: Datenvorbereitung, Ingestion, QA-Lauf und Konfiguration.

**Voraussetzung:** Docker Desktop läuft. Infrastruktur wurde einmalig aufgesetzt (siehe README Quick Start).

---

## Architekturüberblick

```
┌─────────────────────────────────────────────────────┐
│  Docker Compose                                     │
│                                                     │
│  ┌──────────┐   HTTP :11434   ┌──────────────────┐  │
│  │  ollama  │◄────────────────│                  │  │
│  │ (LLM)    │                 │   app container  │  │
│  └──────────┘                 │   (main.py)      │  │
│                               │                  │  │
│  ┌──────────┐   HTTP :8000    │  VectorMemory    │  │
│  │ chromadb │◄────────────────│                  │  │
│  │ (vector  │                 └──────────────────┘  │
│  │  store)  │                                       │
│  └──────────┘                                       │
└─────────────────────────────────────────────────────┘
```

ChromaDB läuft als eigener HTTP-Server — der Vektorspeicher ist vollständig vom App-Prozess isoliert und überlebt Container-Neustarts. BM25 und GraphRAG nutzen ChromaDB **nicht**.

### Embedding-Strategie

| Schritt | Wo | Tool |
|---|---|---|
| Embedding-Berechnung | App-Container (CPU) | `sentence-transformers` BAAI/bge-base-en-v1.5 |
| Vektorspeicherung | ChromaDB-Container | HNSW-Index, Cosine-Distanz |
| Reasoning | Ollama-Container | `llama3.2:3b` via LiteLLM |

**Warum BAAI/bge-base-en-v1.5:** MIT-Lizenz, ~400 MB, MTEB-Score 63.6, kein API-Key, vollständig lokal reproduzierbar.

**Token-Telemetrie:** Embedding-Inference erzeugt keine LLM-Token und erscheint nicht in der Telemetrie. Nur der `reason_node` (LiteLLM → Ollama) erzeugt Telemetrie-Einträge.

---

## Schritt 1 — Infrastruktur starten

```bash
# Ollama + ChromaDB starten
docker compose up -d ollama chromadb

# Status prüfen — beide müssen "healthy" zeigen
docker compose ps
```

ChromaDB braucht ~15 Sekunden bis der Healthcheck grün wird.

---

## Schritt 2 — Datenvorbereitung (einmalig)

Das Skript lädt HotpotQA vom HuggingFace Hub (~560 MB), sampelt stratifiziert nach `type × level` und schreibt das Ergebnis nach `data/raw/hotpotqa_distractor.json`.

```bash
# Standard: 500 Fragen, Validation-Split
docker compose run --rm app python scripts/prepare_hotpotqa.py

# Kleineres Subset für schnelle Tests
docker compose run --rm app python scripts/prepare_hotpotqa.py --samples 100 --output data/raw/hotpotqa_100.json

# Größeres Subset
docker compose run --rm app python scripts/prepare_hotpotqa.py --samples 1000
```

Die Infrastruktur (ollama, chromadb) muss dafür **nicht** laufen — das Skript hat keine Datenbankverbindung.

### Stratifizierung nach Fragetyp

HotpotQA enthält zwei Fragetypen, die für die Hypothesenprüfung relevant sind:

| type | level | Anteil (ca.) | Relevant für |
|---|---|---|---|
| bridge | easy/medium/hard | ~75% | H1: Multi-Hop Reasoning |
| comparison | easy/medium/hard | ~25% | H2: Vergleichsfragen |

`type` und `level` werden pro Frage in den Ergebnissen gespeichert, damit die Evaluierung nach diesen Dimensionen aufschlüsseln kann.

---

## Schritt 3 — Ingestion

```bash
docker compose run --rm app --variant vector_rag --ingest
```

**Was passiert:**

```
docker compose run --rm app --variant vector_rag --ingest
│
├── load_config("vector_rag")
│   └── agent_type="vector", data_file="hotpotqa_distractor.json",
│       memory_path="vector_hotpotqa" (= ChromaDB Collection-Name)
│
├── build_agent(config) → VectorAgent → VectorMemory.__init__()
│   ├── SentenceTransformer("BAAI/bge-base-en-v1.5")
│   │   └── lädt Modell aus huggingface_cache Volume (~400 MB, cached)
│   └── chromadb.HttpClient("http://chromadb:8000")
│       └── get_or_create_collection("vector_hotpotqa", hnsw:space=cosine)
│
├── memory.reset()
│   └── delete_collection + create_collection (saubere Baseline)
│
└── memory.ingest_documents([...])
    ├── Chunking: 300 Wörter pro Chunk, 50 Wörter Overlap (Wortgrenzen)
    ├── Batches à 32 Chunks → SentenceTransformer.encode() → float32-Vektoren
    └── collection.upsert(documents=[...], embeddings=[...], ids=[uuid4()...])
        └── ChromaDB baut HNSW-Index auf
```

**Typische Laufzeit:** 5–15 Minuten (CPU-Embedding, ~500 Dokumente × ~3 Chunks = ~1500 Vektoren).

---

## Schritt 4 — QA-Experiment

```bash
docker compose run --rm app --variant vector_rag --qa
```

**Was passiert pro Frage:**

```
agent.run(question)
│
├── retrieve_node → memory.search(question)
│   ├── SentenceTransformer.encode([question]) → Query-Vektor
│   ├── collection.query(query_embeddings=[...], n_results=5)
│   │   └── ChromaDB: ANN-Suche im HNSW-Index
│   └── Filtert Ergebnisse: similarity = 1 - (cosine_distance / 2) ≥ 0.7
│
└── reason_node → litellm.completion(
        model="ollama/llama3.2:3b",
        api_base="http://ollama:11434",
        messages=[system_prompt + retrieved_passages + question]
    )
    └── TelemetryTracker → evaluations/vector_rag_<ts>_telemetry.jsonl
```

---

## Schritt 5 — Ingestion + QA in einem Schritt

```bash
docker compose run --rm app --variant vector_rag --ingest --qa
```

---

## Ausgabedateien

Nach `--qa` entstehen zwei Dateien in `evaluations/`:

**`vector_rag_<timestamp>_results.jsonl`** — eine Zeile pro Frage:
```json
{
  "run_id": "a3f1c9d2",
  "variant_name": "vector_rag",
  "agent_type": "vector",
  "question": "What is the nationality of the engineer who designed the Eiffel Tower?",
  "expected": "French",
  "answer": "The engineer Gustave Eiffel was French.",
  "context_used": ["Eiffel Tower: The tower was designed by Gustave Eiffel ..."],
  "steps_taken": ["retrieve", "reason"],
  "backend_name": "vector"
}
```

**`vector_rag_<timestamp>_telemetry.jsonl`** — eine Zeile pro LLM-Call:
```json
{
  "event": "llm_call",
  "run_id": "a3f1c9d2",
  "phase": "agent_reasoning",
  "actor": "langgraph_node",
  "variant_name": "vector_rag",
  "model": "ollama/llama3.2:3b",
  "prompt_tokens": 428,
  "completion_tokens": 52,
  "total_tokens": 480,
  "latency_ms": 1124.5
}
```

Join über `run_id` um Token-Kosten pro Frage zuzuordnen.

---

## Vergleich aller Varianten

```bash
# Alle drei Modelle sequenziell durchlaufen
docker compose run --rm app --variant all --ingest --qa
```

GraphRAG-Ingestion dauert deutlich länger (LLM-Calls für Entity-Extraction). Empfehlung: Varianten einzeln starten und überwachen.

---

## Konfiguration

Alle Parameter in `src/config/unified_config.yaml` unter `vector_rag`:

```yaml
vector_rag:
  agent_type: "vector"
  data_file: "hotpotqa_distractor.json"
  memory_path: "vector_hotpotqa"       # ChromaDB Collection-Name
  # Vererbte Defaults — bei Bedarf hier überschreiben:
  # llm:
  #   model: "ollama/llama3.2:3b"
  # retrieval:
  #   top_k: 5
  #   similarity_cutoff: 0.7           # 0.0–1.0, zu hoch → leere Kontexte
  # ingestion:
  #   chunk_size: 300                  # Wörter pro Chunk
  #   chunk_overlap: 50
  # embedding:
  #   model: "huggingface/BAAI/bge-base-en-v1.5"
  #   batch_size: 32
```

---

## Troubleshooting

**`dependency failed to start: chromadb is unhealthy`**
→ `docker compose up -d chromadb` und warten bis `docker compose ps` "healthy" zeigt (~15s).

**`ConnectionError: Failed to connect to ChromaDB`**
→ ChromaDB läuft nicht. `docker compose up -d chromadb` ausführen.

**Leere `context_used` in den Ergebnissen**
→ `similarity_cutoff` zu hoch. In `unified_config.yaml` unter `vector_rag` auf `0.5` reduzieren und neu testen.

**Embedding-Modell wird bei jedem Start neu heruntergeladen**
→ `huggingface_cache`-Volume fehlt oder wurde gelöscht. Prüfen: `docker volume ls | findstr huggingface`

**ChromaDB-Collection existiert nicht beim QA-Schritt**
→ Ingestion fehlt oder ChromaDB-Volume wurde gelöscht. `--ingest` erneut ausführen.
