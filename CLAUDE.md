# 🤖 Coding Agent Context: Token-Efficient Agentic RAG Project

## 1. Project Overview & Scientific Goal

This repository is a scientific evaluation framework designed to measure and optimize token efficiency, memory architectures, and latency in autonomous AI agents. The project follows the CRISP-DM methodology.

**Core Problem:** Current open-source agents suffer from "context poisoning" and massive token consumption when using standard Vector RAG for complex, multi-hop reasoning.
**Proposed Solution:** Systematically compare a baseline Vector RAG system against an optimized (Hybrid) GraphRAG to measure the tradeoff between heavy upfront ingestion costs (GraphRAG) and test-time token waste (Vector RAG).

## 2. Tech Stack & Frameworks

You must strictly adhere to the following stack when generating code:

- **Agent Orchestration:** `LangGraph` (Used for defining agent state, reasoning loops, and tool calling).
- **Memory & RAG:** Not defined yet.
- **LLM Gateway & Telemetry:** `LiteLLM` (MUST be used for ALL LLM calls to ensure unified configuration and precise token/cost tracking).
- **Configuration:** Pydantic / YAML (Unified config across all models).

## 3. Architecture & Separation of Concerns

The system is strictly divided into two phases. **Do not mix these contexts.**

### Phase A: Offline Ingestion (Pipeline 1)

- **Goal:** Process raw datasets (e.g., MemoryAgentBench) into memory stores.
- **Constraint:** The LangGraph agent is NOT active here. This is purely deterministic data processing.
- **Hidden Costs:** GraphRAG ingestion requires heavy LLM inference for entity extraction. These LLM calls MUST be routed through LiteLLM and tagged appropriately (see Telemetry rules).

### Phase B: Online Test-Time / Agentic Reasoning (Pipeline 2)

- **Goal:** The LangGraph agent answers complex multi-hop questions.
- **Constraint:** The agent is stateless except for its context window. It interacts with memory EXCLUSIVELY via two tools:
  1. `SearchMemory(query: str)`: Pulls context from the active memory model.
  2. `UpdateMemory(fact: str)`: Injects new facts learned at test-time into the memory model.
- **Abstraction:** The LangGraph agent MUST NOT know if it is querying a Vector DB or a Knowledge Graph. The `SearchMemory` tool delegates to a standard interface (`src/memory/base.py`).

## 4. Telemetry & Tagging Rules (CRITICAL)

Scientific benchmarking is the core of this project. Every single LLM call—whether for agent reasoning, graph node extraction, or Cypher query generation—must be tracked.

Whenever utilizing LiteLLM or an LLM integration within LlamaIndex/LangGraph, you must inject custom tags to track the _phase_ and _actor_.

**Required Telemetry Tags:**

- `phase`: ["ingest", "retrieval_overhead", "agent_reasoning", "evaluation"]
- `actor`: ["vector_embed", "graph_extract", "graph_cypher_gen", "langgraph_node", "llm_as_judge"]

## 5. Directory Rationale & Architectural Layout

When navigating or generating new code, adhere to this directory-level separation of concerns:

- **`data/` (Data Persistence Layer):** Isolates state from logic. It separates raw benchmarking datasets from the processed, persisted artifacts (like ChromaDB vector indices or Neo4j/LlamaIndex property graphs).
- **`evaluations/` (Output & Metrics Layer):** The destination for all telemetry logs, generated agent trajectories, and final accuracy scores. Code here should only read from the pipeline outputs, never modify system state.
- **`notebooks/` (Exploratory Data Analysis & Visualization):** Used for interactive data exploration, visualizing telemetry results, analyzing token consumption tradeoffs, and mapping the generated Knowledge Graphs. This sits strictly outside the core pipeline; no production or pipeline logic should be executed from or rely on Jupyter notebooks.
- **`src/config/` (Configuration Management):** The single source of truth for the framework. Houses the unified YAML settings and Pydantic loaders to ensure that parameters (like LLM temperature or chunk sizes) remain identical across all comparative runs.
- **`src/telemetry/` (Observability Layer):** Contains the LiteLLM callbacks and tracking decorators. This code intercepts all LLM traffic to enforce the tagging rules defined in Section 4.
- **`src/memory/` (Core Abstraction Layer):** Houses the foundational memory interfaces. The base class lives here, alongside the concrete implementations for Vector RAG (Model A) and GraphRAG (Model B). The agent must only ever interact with the base interface.
- **`src/agent/` (Orchestration Layer):** Contains the LangGraph state definitions, routing logic, and the highly restricted toolset (`SearchMemory`, `UpdateMemory`).
- **`src/pipelines/` (Execution Layer):** The top-level scripts that string the other components together. Separated into three distinct runtimes: Data Ingestion (`01`), Agentic QA (`02`), and Evaluation (`03`).

## 6. Development Rules for Coding Agent

1. **Never hardcode model names:** Always pull the model name (e.g., `ollama/mixtral`, `gpt-4o`) and parameters (`top_k`, `temperature`) from the unified config.
2. **Abstract the Memory:** When writing agent tools, rely on the abstract methods of the memory interface. Let the config decide which memory subclass is instantiated.
3. **Type Hinting:** Use strict Python type hinting (Python 3.10+) and Pydantic for data structures.
4. **Think Step-by-Step:** When asked to implement a feature, first identify which pipeline (Ingest, QA, Evaluate) and which domain (Memory, Agent, Telemetry) it belongs to.
5. **Dependency management:** This Project uses uv for packet management
