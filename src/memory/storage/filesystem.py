from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base import Entity, StorageBackend, Triplet


class FilesystemStorage(StorageBackend):
    """
    Local filesystem storage: JSON for entities/triplets, numpy .npy for vectors.

    The FAISS index object is not persisted — it is rebuilt in memory from the
    raw float32 matrix on every load, keeping this class free of faiss as a
    compile-time dependency.

    Directory layout::

        <base_dir>/
            entities.json       — {normalised_key: Entity.model_dump()}
            triplets.json       — [Triplet.model_dump(), ...]
            faiss_vectors.npy   — float32 array [N, D] (L2-normalised)
            faiss_keys.json     — parallel list["trip:{idx}"]
    """

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._entities_path = base_dir / "entities.json"
        self._triplets_path = base_dir / "triplets.json"
        self._vectors_path = base_dir / "faiss_vectors.npy"
        self._keys_path = base_dir / "faiss_keys.json"

    def _ensure_dir(self) -> None:
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def load_entities(self) -> dict[str, Entity]:
        if not self._entities_path.exists():
            return {}
        raw = json.loads(self._entities_path.read_text("utf-8"))
        return {k: Entity.model_validate(v) for k, v in raw.items()}

    def save_entities(self, entities: dict[str, Entity]) -> None:
        self._ensure_dir()
        self._entities_path.write_text(
            json.dumps(
                {k: v.model_dump() for k, v in entities.items()},
                ensure_ascii=False,
                indent=2,
            ),
            "utf-8",
        )

    def load_triplets(self) -> list[Triplet]:
        if not self._triplets_path.exists():
            return []
        raw = json.loads(self._triplets_path.read_text("utf-8"))
        return [Triplet.model_validate(r) for r in raw]

    def save_triplets(self, triplets: list[Triplet]) -> None:
        self._ensure_dir()
        self._triplets_path.write_text(
            json.dumps(
                [t.model_dump() for t in triplets],
                ensure_ascii=False,
                indent=2,
            ),
            "utf-8",
        )

    def load_index(self) -> tuple[np.ndarray | None, list[str]]:
        if not self._vectors_path.exists() or not self._keys_path.exists():
            return None, []
        vectors = np.load(str(self._vectors_path)).astype(np.float32)
        keys: list[str] = json.loads(self._keys_path.read_text("utf-8"))
        return vectors, keys

    def save_index(self, vectors: np.ndarray, index_keys: list[str]) -> None:
        self._ensure_dir()
        np.save(str(self._vectors_path), vectors.astype(np.float32))
        self._keys_path.write_text(json.dumps(index_keys, ensure_ascii=False), "utf-8")

    def clear(self) -> None:
        for path in (
            self._entities_path,
            self._triplets_path,
            self._vectors_path,
            self._keys_path,
        ):
            if path.exists():
                path.unlink()
