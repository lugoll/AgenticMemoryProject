from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel, Field


class Entity(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)  # unique chunk MD5 hashes


class Triplet(BaseModel):
    subject: str      # canonical entity name (resolved at ingest)
    predicate: str    # short verb phrase — edge label for graph structure
    description: str  # self-contained rich phrase — embedded for semantic search
    object: str       # canonical entity name or literal value (graph traversal endpoint)
    source_hash: str


class StorageBackend(ABC):
    """
    Abstract storage layer for VecGraph entities, triplets, and the FAISS vector index.

    Implementations of this interface are the only place where persistence
    details live — the memory class itself is agnostic to whether data is on
    local disk, a database, or a remote store.

    Index storage contract: the raw float32 matrix is stored (not a FAISS index
    object), so this class has no faiss dependency. The FAISS index is rebuilt
    in memory from the vectors by the caller.
    """

    @abstractmethod
    def load_entities(self) -> dict[str, Entity]:
        """Return all entities keyed by their normalised name."""
        ...

    @abstractmethod
    def save_entities(self, entities: dict[str, Entity]) -> None: ...

    @abstractmethod
    def load_triplets(self) -> list[Triplet]: ...

    @abstractmethod
    def save_triplets(self, triplets: list[Triplet]) -> None: ...

    @abstractmethod
    def load_index(self) -> tuple[np.ndarray | None, list[str]]:
        """
        Return (vectors, index_keys).

        vectors: float32 array of shape [N, D], or None if no index persisted.
        index_keys: parallel list of "trip:{idx}" strings.
        """
        ...

    @abstractmethod
    def save_index(self, vectors: np.ndarray, index_keys: list[str]) -> None: ...

    @abstractmethod
    def clear(self) -> None:
        """Delete all persisted state."""
        ...
