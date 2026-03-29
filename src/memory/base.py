from __future__ import annotations

from abc import ABC, abstractmethod


class BaseMemory(ABC):
    """
    Abstract interface for all memory backends.

    The LangGraph agent interacts with memory EXCLUSIVELY through this interface.
    It must not know whether the underlying implementation is Vector RAG, GraphRAG,
    or a dummy in-memory store.

    Subclass contract:
        - All three abstract methods must be implemented.
        - search() must return plain strings, never backend-specific objects
          (e.g., LlamaIndex Documents, embedding vectors, node IDs).
        - Token costs incurred by implementations must be tracked via the
          telemetry tracker before returning.
    """

    @abstractmethod
    def ingest_documents(self, documents: list[str]) -> None:
        """
        Process and persist a list of raw text documents into the memory store.

        Phase A (offline ingestion) operation. May involve chunking, embedding,
        and/or LLM-based entity extraction depending on the backend.

        Args:
            documents: Raw text strings to ingest. Each is treated as one
                       document unit before any chunking the backend applies.
        """
        ...

    @abstractmethod
    def search(self, query: str) -> list[str]:
        """
        Retrieve relevant text passages from the memory store.

        Phase B (test-time retrieval) operation. The number of results returned
        is determined by the backend's configured top_k.

        Args:
            query: Natural language query string from the agent.

        Returns:
            Text passages most relevant to the query, ranked by relevance.
            Returns an empty list if no relevant passages are found.
            Never raises on empty results.
        """
        ...

    @abstractmethod
    def update_fact(self, fact: str) -> None:
        """
        Inject a new fact learned at test-time into the memory store.

        Phase B (test-time update) operation triggered by the agent's
        UpdateMemory tool. Must integrate with the same store that search()
        reads from so subsequent searches can return the new fact.

        Args:
            fact: A single natural language statement to persist.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """
        Clear all persisted data from the memory store.

        Must be called at the start of every ingestion run to ensure a clean
        baseline. After reset(), store_size (if present) must be 0 and
        search() must return [] for any query.
        """
        ...

    def get_backend_name(self) -> str:
        """Returns the string identifier for this backend. Used for telemetry tagging."""
        return self.__class__.__name__.lower()
