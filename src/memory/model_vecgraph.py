from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import litellm
import numpy as np

from src.config.settings import ExperimentConfig, VecGraphConfig

from .base import BaseMemory
from .storage import Entity, FilesystemStorage, StorageBackend, Triplet

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """\
Extract all information from the text below as Subject-Relation-Object triplets.

Return a JSON array where each element has exactly four keys:
- "subject": the entity the statement is about (1-5 words, Title Case)
- "predicate": a short verb phrase summarising the relation (1-4 words)
- "description": a self-contained descriptive phrase conveying full semantic meaning \
(e.g. "won the Nobel Prize in Physics in 1903 for her research on radioactivity" rather than "won")
- "object": the target entity or value (1-5 words, Title Case for named entities)

Rules:
- Use consistent, specific names: "Nikola Tesla" not "Tesla" or "the inventor".
- The description must make sense standalone without subject or object as context.
- Return [] if nothing can be extracted.
- Return ONLY the JSON array — no markdown fences, no extra text.

Text: {text}

Example:
[
  {{"subject": "Marie Curie", "predicate": "was born in", "description": "was born in Warsaw in 1867", "object": "Warsaw"}},
  {{"subject": "Marie Curie", "predicate": "won", "description": "won the Nobel Prize in Physics in 1903 for her pioneering research on radioactivity", "object": "Nobel Prize in Physics"}}
]"""

_LM_ENTITY_SELECT_PROMPT = """\
Decide which of the candidates below, if any, is the EXACT same real-world entity as the new name.

New name: "{candidate}"

Candidates:
{candidates_block}

Rules:
- Reply with the NUMBER of the matching candidate if the new name is merely an alias \
or alternate spelling for that entity (e.g. "USA" → 1 for "United States", "Tesla" → 2 for "Nikola Tesla").
- Reply with 0 if the new name is a distinct entity from all candidates, \
even if closely related (e.g. "Smiljan" vs "Nikola Tesla", "Paris" vs "France").
- Reply with a single integer only — no other text."""


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping word-boundary chunks."""
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - chunk_overlap
    return chunks


class VecGraphMemory(BaseMemory):
    """
    Vector-Graph memory backend (pure triplet model).

    Ingest (Phase A):
        Documents are chunked and passed to the configured LLM, which extracts
        all information as Subject-Predicate-Object triplets with a rich
        description per edge. Triplets are stored in an append-only JSON ledger.
        After all documents are processed, near-duplicate triplets are removed by
        cosine-similarity comparison on their embedded descriptions, and a shared
        FAISS index is built over all remaining triplets.

        LLM calls tagged:       phase="ingest",  actor="graph_extract"
        Embedding calls tagged: phase="ingest",  actor="vector_embed"

    Retrieval (Phase B):
        A query is embedded and searched against the shared FAISS index.
        Triplet hits surface the matched triplet plus all other triplets that
        share a subject or object entity (1-hop graph neighbourhood).

        Embedding calls tagged: phase="retrieval_overhead", actor="vector_embed"

    update_fact() (Phase B):
        New fact string → LLM extraction → append to ledger → full FAISS rebuild.

        LLM calls tagged:       phase="agent_reasoning", actor="graph_extract"
        Embedding calls tagged: phase="agent_reasoning", actor="vector_embed"

    Entity resolution:
        - sim >= entity_lm_lower → candidate enters LLM selection pool
        - LLM receives top-k candidates (entity_lm_top_k) and picks by index (0 = none)
        - sim < entity_lm_lower → new entity, no LLM call

    Storage:
        Backed by an injectable StorageBackend (default: FilesystemStorage).
    """

    def __init__(
        self,
        config: ExperimentConfig,
        storage: StorageBackend | None = None,
    ) -> None:
        self._config = config
        self._vg_cfg: VecGraphConfig = config.vecgraph or VecGraphConfig()
        base_dir = (
            Path(config.memory_path)
            if config.memory_path
            else Path("data/processed/vecgraph_default")
        )
        self._storage: StorageBackend = storage or FilesystemStorage(base_dir)

        self._entities: dict[str, Entity] = {}
        self._triplets: list[Triplet] = []
        self._index = None          # faiss.IndexFlatIP | None
        self._index_keys: list[str] = []

        # Entity-name dedup support
        self._name_vecs: dict[str, np.ndarray] = {}   # entity_key → L2-normalised name embedding
        self._alias_to_key: dict[str, str] = {}       # normalised alias → canonical entity_key

        self._load_state()

    # ---- Initialisation ----

    def _load_state(self) -> None:
        self._entities = self._storage.load_entities()
        self._triplets = self._storage.load_triplets()
        vectors, keys = self._storage.load_index()
        if vectors is not None and len(keys) > 0 and vectors.shape[0] > 0:
            self._rebuild_faiss_from_vectors(vectors, keys)
        self._rebuild_alias_index()

    def _rebuild_alias_index(self) -> None:
        self._alias_to_key = {}
        for entity_key, entity in self._entities.items():
            for alias in entity.aliases:
                self._alias_to_key[self._normalize_name(alias)] = entity_key

    def _rebuild_faiss_from_vectors(self, vectors: np.ndarray, keys: list[str]) -> None:
        import faiss

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors.astype(np.float32))
        self._index = index
        self._index_keys = list(keys)

    # ---- Private helpers ----

    def _normalize_name(self, name: str) -> str:
        return name.strip().lower()

    def _embed(self, texts: list[str], phase: str, run_id: str) -> np.ndarray:
        response = litellm.embedding(
            model=self._config.embedding.model,
            input=texts,
            metadata={
                "phase": phase,
                "actor": "vector_embed",
                "variant_name": self._config.variant_name,
                "run_id": run_id,
            },
        )
        return np.array([item["embedding"] for item in response.data], dtype=np.float32)

    def _lm_select_entity(
        self,
        candidate_name: str,
        candidates: list[tuple[str, str]],  # [(entity_key, canonical_name), ...]
        phase: str,
        run_id: str,
    ) -> str | None:
        """Present up to entity_lm_top_k candidates to the LLM and return the matching key.

        The LLM replies with the 1-based index of the matching candidate, or 0 for no match.
        Returns the entity_key of the chosen candidate, or None.
        """
        candidates_block = "\n".join(
            f'  {i + 1}. "{name}"' for i, (_, name) in enumerate(candidates)
        )
        prompt = _LM_ENTITY_SELECT_PROMPT.format(
            candidate=candidate_name,
            candidates_block=candidates_block,
        )
        response = litellm.completion(
            model=self._config.llm.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4,
            metadata={
                "phase": phase,
                "actor": "graph_extract",
                "variant_name": self._config.variant_name,
                "run_id": run_id,
            },
        )
        answer = (response.choices[0].message.content or "").strip()  # type: ignore[union-attr]
        try:
            idx = int(answer)
        except ValueError:
            return None
        if idx < 1 or idx > len(candidates):
            return None
        return candidates[idx - 1][0]

    def _find_similar_entity(
        self, normalized_key: str, candidate_name: str, phase: str, run_id: str
    ) -> str | None:
        """Return the canonical entity_key for candidate_name if the LLM selects a match.

        All entities with sim >= entity_lm_lower are collected, sorted by similarity
        descending, and the top entity_lm_top_k are presented to the LLM in a single
        call. The LLM picks the matching one (by index) or replies 0 for no match.

        sim < entity_lm_lower → no LLM call, new entity created.
        """
        lm_lower = self._vg_cfg.entity_lm_lower
        top_k = self._vg_cfg.entity_lm_top_k

        if lm_lower <= 0.0 or not self._entities:
            return None

        existing_keys = [k for k in self._entities if k != normalized_key]
        if not existing_keys:
            return None

        # Batch-embed the candidate + all uncached entity names in one API call
        names_to_embed: list[str] = [candidate_name]
        keys_to_fill: list[str] = []
        for k in existing_keys:
            if k not in self._name_vecs:
                names_to_embed.append(self._entities[k].name)
                keys_to_fill.append(k)

        raw = self._embed(names_to_embed, phase=phase, run_id=run_id)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = (raw / norms).astype(np.float32)

        cand_vec = normed[0]
        for i, k in enumerate(keys_to_fill):
            self._name_vecs[k] = normed[i + 1]

        # Collect all candidates at or above lm_lower, sorted by similarity descending
        scored: list[tuple[float, str]] = [
            (float(np.dot(cand_vec, self._name_vecs[k])), k)
            for k in existing_keys
            if k in self._name_vecs
        ]
        candidates = [
            (k, self._entities[k].name)
            for sim, k in sorted(scored, key=lambda x: x[0], reverse=True)[:top_k]
            if sim >= lm_lower
        ]

        if not candidates:
            return None

        matched_key = self._lm_select_entity(candidate_name, candidates, phase, run_id)
        if matched_key is not None:
            logger.info(
                "VecGraphMemory: LLM selected merge '%s' → '%s' (%d candidate(s) shown)",
                candidate_name, self._entities[matched_key].name, len(candidates),
            )
            return matched_key

        logger.info(
            "VecGraphMemory: LLM found no match for '%s' among %d candidate(s) — new entity",
            candidate_name, len(candidates),
        )
        return None

    def _resolve_entity_key(self, name: str) -> str:
        norm = self._normalize_name(name)
        if norm in self._entities:
            return norm
        return self._alias_to_key.get(norm, norm)

    def _get_or_create_entity(
        self, name: str, phase: str, run_id: str
    ) -> tuple[str, str]:
        """Return (entity_key, canonical_name) for *name*, creating the entity if needed."""
        norm = self._normalize_name(name)

        if norm in self._entities:
            return norm, self._entities[norm].name

        if norm in self._alias_to_key:
            key = self._alias_to_key[norm]
            return key, self._entities[key].name

        similar_key = self._find_similar_entity(norm, name, phase=phase, run_id=run_id)
        if similar_key is not None:
            entity = self._entities[similar_key]
            if name not in entity.aliases:
                entity.aliases.append(name)
            self._alias_to_key[norm] = similar_key
            return similar_key, entity.name

        self._entities[norm] = Entity(name=name)
        return norm, name

    def _extract(
        self, text: str, phase: str, run_id: str
    ) -> list[tuple[str, str, str, str]]:
        """Call LLM to extract (subject, predicate, description, object) tuples."""
        prompt = _EXTRACTION_PROMPT.format(text=text)
        response = litellm.completion(
            model=self._config.llm.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=self._vg_cfg.extraction_max_tokens,
            metadata={
                "phase": phase,
                "actor": "graph_extract",
                "variant_name": self._config.variant_name,
                "run_id": run_id,
            },
        )
        raw = response.choices[0].message.content or "[]"  # type: ignore[union-attr]
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("VecGraphMemory: could not parse LLM response: %r", raw[:200])
            return []

        if not isinstance(parsed, list):
            return []

        result: list[tuple[str, str, str, str]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            subj = item.get("subject", "")
            pred = item.get("predicate", "")
            desc = item.get("description", "")
            obj = item.get("object", "")
            if subj and pred and desc and obj:
                result.append((
                    str(subj).strip(),
                    str(pred).strip(),
                    str(desc).strip(),
                    str(obj).strip(),
                ))
        return result

    def _append_triplets(
        self,
        extracted: list[tuple[str, str, str, str]],
        source_hash: str,
        phase: str,
        run_id: str,
    ) -> None:
        seen_triplets: set[tuple[str, str, str]] = {
            (
                self._resolve_entity_key(t.subject),
                t.description.lower().strip(),
                self._resolve_entity_key(t.object),
            )
            for t in self._triplets
        }

        for subject, predicate, description, obj in extracted:
            subj_key, canonical_subj = self._get_or_create_entity(subject, phase, run_id)
            obj_key, canonical_obj = self._get_or_create_entity(obj, phase, run_id)

            dedup_key = (subj_key, description.lower().strip(), obj_key)
            if dedup_key in seen_triplets:
                continue
            seen_triplets.add(dedup_key)

            # Track source on both endpoint entities (for chunk-level dedup guard)
            for key in (subj_key, obj_key):
                entity = self._entities[key]
                if source_hash not in entity.sources:
                    entity.sources.append(source_hash)

            self._triplets.append(
                Triplet(
                    subject=canonical_subj,
                    predicate=predicate,
                    description=description,
                    object=canonical_obj,
                    source_hash=source_hash,
                )
            )

        self._storage.save_entities(self._entities)
        self._storage.save_triplets(self._triplets)

    def _save_and_rebuild_index(self, phase: str, run_id: str) -> None:
        import faiss

        if not self._triplets:
            self._index = None
            self._index_keys = []
            self._storage.save_index(np.empty((0, 1), dtype=np.float32), [])
            return

        # --- 1. Collect texts to embed: "{subject} {description}" ---
        all_texts = [f"{t.subject} {t.description}" for t in self._triplets]

        # --- 2. Embed in batches ---
        batch_size = self._config.embedding.batch_size
        vecs_list: list[np.ndarray] = []
        for start in range(0, len(all_texts), batch_size):
            batch = all_texts[start : start + batch_size]
            vecs_list.append(self._embed(batch, phase=phase, run_id=run_id))
        raw_vectors = np.concatenate(vecs_list, axis=0).astype(np.float32)

        # --- 3. L2-normalise ---
        norms = np.linalg.norm(raw_vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        norm_vectors = (raw_vectors / norms).astype(np.float32)

        # --- 4. Global semantic dedup across all triplets ---
        threshold = self._vg_cfg.dedup_threshold
        keep_mask = [True] * len(self._triplets)
        sims = norm_vectors @ norm_vectors.T  # [N, N]

        for i in range(len(self._triplets)):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, len(self._triplets)):
                if not keep_mask[j]:
                    continue
                if float(sims[i, j]) > threshold:
                    keep_mask[j] = False

        # --- 5. Prune triplets list to survivors ---
        self._triplets = [t for t, keep in zip(self._triplets, keep_mask) if keep]
        self._storage.save_triplets(self._triplets)

        # --- 6. Build final vectors + keys ---
        kept_indices = [i for i, keep in enumerate(keep_mask) if keep]
        final_vectors = norm_vectors[kept_indices]
        final_keys = [f"trip:{j}" for j in range(len(self._triplets))]

        # --- 7. Build FAISS index ---
        index = faiss.IndexFlatIP(final_vectors.shape[1])
        index.add(final_vectors)
        self._index = index
        self._index_keys = final_keys
        self._storage.save_index(final_vectors, final_keys)

        logger.debug(
            "VecGraphMemory: index built — %d entities, %d triplets (after dedup)",
            len(self._entities),
            len(self._triplets),
        )

    # ---- BaseMemory interface ----

    def ingest_documents(self, documents: list[str]) -> None:
        chunk_size = self._config.ingestion.chunk_size
        chunk_overlap = self._config.ingestion.chunk_overlap
        total_chunks = 0

        for doc in documents:
            stripped = doc.strip()
            if not stripped:
                continue
            for chunk in _chunk_text(stripped, chunk_size, chunk_overlap):
                source_hash = hashlib.md5(chunk.encode("utf-8")).hexdigest()
                extracted = self._extract(chunk, phase="ingest", run_id="ingest")
                self._append_triplets(extracted, source_hash, phase="ingest", run_id="ingest")
                total_chunks += 1

        logger.debug(
            "VecGraphMemory: ingested %d documents → %d chunks → building index",
            len(documents),
            total_chunks,
        )
        self._save_and_rebuild_index(phase="ingest", run_id="ingest")

    def search(self, query: str) -> list[str]:
        if not query.strip():
            return []
        if self._index is None or self._index.ntotal == 0:
            return []

        top_k = self._config.retrieval.top_k

        q_vec = self._embed([query.strip()], phase="retrieval_overhead", run_id="search")
        q_norms = np.linalg.norm(q_vec, axis=1, keepdims=True)
        q_norms = np.where(q_norms == 0, 1.0, q_norms)
        q_normalized = (q_vec / q_norms).astype(np.float32)

        _distances, indices = self._index.search(q_normalized, k=min(top_k, self._index.ntotal))

        context: dict[str, None] = {}      # ordered set via dict keys
        matched_entity_keys: list[str] = []

        for idx in indices[0]:
            if idx < 0 or idx >= len(self._index_keys):
                continue
            key = self._index_keys[idx]
            if not key.startswith("trip:"):
                continue
            trip_idx = int(key.split(":", 1)[1])
            if trip_idx >= len(self._triplets):
                continue
            trip = self._triplets[trip_idx]
            context[f"[{trip.subject}] {trip.description} [{trip.object}]"] = None

            for name in (trip.subject, trip.object):
                entity_key = self._resolve_entity_key(name)
                if entity_key not in matched_entity_keys:
                    matched_entity_keys.append(entity_key)

        # 1-hop expansion: all triplets sharing an endpoint with matched entities
        for trip in self._triplets:
            subj_key = self._resolve_entity_key(trip.subject)
            obj_key = self._resolve_entity_key(trip.object)
            if subj_key in matched_entity_keys or obj_key in matched_entity_keys:
                context[f"[{trip.subject}] {trip.description} [{trip.object}]"] = None

        results = list(context.keys())
        cap = top_k * 4
        logger.debug(
            "VecGraphMemory: query=%r → %d context strings (capped at %d)",
            query,
            len(results),
            cap,
        )
        return results[:cap]

    def update_fact(self, fact: str) -> None:
        if not fact.strip():
            return
        source_hash = hashlib.md5(fact.strip().encode("utf-8")).hexdigest()
        extracted = self._extract(fact.strip(), phase="agent_reasoning", run_id="update_fact")
        self._append_triplets(extracted, source_hash, phase="agent_reasoning", run_id="update_fact")
        self._save_and_rebuild_index(phase="agent_reasoning", run_id="update_fact")

    def reset(self) -> None:
        self._entities = {}
        self._triplets = []
        self._index = None
        self._index_keys = []
        self._name_vecs = {}
        self._alias_to_key = {}
        self._storage.clear()
        logger.debug("VecGraphMemory: reset")

    def get_backend_name(self) -> str:
        return "vecgraph"

    # ---- Introspection ----

    @property
    def entity_count(self) -> int:
        return len(self._entities)

    @property
    def triplet_count(self) -> int:
        return len(self._triplets)

    @property
    def index_size(self) -> int:
        return self._index.ntotal if self._index is not None else 0
