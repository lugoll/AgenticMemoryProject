"""Tests for VecGraphMemory entity-deduplication behaviour.

Regression suite for the "Nikola Tesla bug" where topically-related but
distinct entities (Smiljan, Zagreb, Croatia, …) were incorrectly merged into
the first ingested entity because:

  (a) entity_name_dedup_threshold=0.88 triggered a silent auto-merge without
      any LLM confirmation, and
  (b) entity_lm_lower=0.82 sat below the ~0.83-0.88 cosine-similarity band
      that passage-level embedding models (nomic-embed-text) produce for
      domain-co-occurring proper nouns.

After the fix:
  • entity_name_dedup_threshold is gone — no auto-merge code path exists.
  • entity_lm_lower=0.90 guards every LLM confirmation call.
  • The LLM prompt includes counter-examples to prevent false YES responses.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.config.settings import VecGraphConfig
from src.memory.model_vecgraph import VecGraphMemory
from src.memory.storage.base import StorageBackend


# ---------------------------------------------------------------------------
# Test doubles & helpers
# ---------------------------------------------------------------------------

class InMemoryStorage(StorageBackend):
    """Fully in-memory StorageBackend — no filesystem touches in tests."""

    def __init__(self) -> None:
        self._entities: dict = {}
        self._triplets: list = []
        self._vectors = None
        self._keys: list = []

    def load_entities(self) -> dict:
        return dict(self._entities)

    def save_entities(self, entities: dict) -> None:
        self._entities = dict(entities)

    def load_triplets(self) -> list:
        return list(self._triplets)

    def save_triplets(self, triplets: list) -> None:
        self._triplets = list(triplets)

    def load_index(self) -> tuple:
        return self._vectors, list(self._keys)

    def save_index(self, vectors, keys: list) -> None:
        self._vectors = vectors.copy() if vectors is not None else None
        self._keys = list(keys)

    def clear(self) -> None:
        self._entities = {}
        self._triplets = []
        self._vectors = None
        self._keys = []


# Orthogonal unit vectors — cosine similarity between any pair is exactly 0.
# Used to guarantee that no entity exceeds entity_lm_lower (0.90).
_ORTH_VECS: dict[str, list[float]] = {
    "Nikola Tesla":                          [1, 0, 0, 0, 0, 0, 0, 0],
    "Smiljan":                               [0, 1, 0, 0, 0, 0, 0, 0],
    "Zagreb":                                [0, 0, 1, 0, 0, 0, 0, 0],
    "Croatia":                               [0, 0, 0, 1, 0, 0, 0, 0],
    "Alternating Current Electrical System": [0, 0, 0, 0, 1, 0, 0, 0],
    "Inventor":                              [0, 0, 0, 0, 0, 1, 0, 0],
    "AC System":                             [0, 0, 0, 0, 0, 0, 1, 0],
}

# High-similarity pair: cosine("Tesla", "Nikola Tesla") ≈ 0.95 >= entity_lm_lower.
# [0.95, sqrt(1-0.95²)] is a unit vector at angle ≈ 18° from [1, 0, 0, 0].
_HIGH_SIM_VECS: dict[str, list[float]] = {
    "Nikola Tesla": [1.0, 0.0, 0.0, 0.0],
    "Tesla":        [0.95, 0.312, 0.0, 0.0],
    "Inventor":     [0.0, 1.0, 0.0, 0.0],
    "AC System":    [0.0, 0.0, 1.0, 0.0],
}


def _embed_fn(mapping: dict[str, list[float]]):
    """Return a litellm.embedding side-effect function backed by *mapping*.

    Unknown texts receive a unique orthogonal vector so they never accidentally
    exceed the similarity threshold.
    """
    dim = len(next(iter(mapping.values()))) if mapping else 8
    _counter = [0]

    def _embed(model, input, metadata=None, **_kw):
        resp = MagicMock()
        resp.data = []
        for text in input:
            if text in mapping:
                vec = list(mapping[text])
            else:
                v = [0.0] * dim
                v[_counter[0] % dim] = 1.0
                _counter[0] += 1
                vec = v
            resp.data.append({"embedding": vec})
        return resp

    return _embed


def _completion(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices[0].message.content = content
    return resp


def _make_config(lm_lower: float = 0.90) -> MagicMock:
    cfg = MagicMock()
    cfg.memory_path = None
    cfg.variant_name = "test"
    cfg.llm.model = "ollama/test-llm"
    cfg.embedding.model = "ollama/test-embed"
    cfg.embedding.batch_size = 32
    cfg.ingestion.chunk_size = 512
    cfg.ingestion.chunk_overlap = 64
    cfg.retrieval.top_k = 5
    cfg.vecgraph = VecGraphConfig(entity_lm_lower=lm_lower)
    return cfg


def _triplets_json(*triplets: dict) -> str:
    return json.dumps(list(triplets))


# ---------------------------------------------------------------------------
# Config structure
# ---------------------------------------------------------------------------

class TestVecGraphConfigStructure:
    def test_entity_name_dedup_threshold_field_removed(self) -> None:
        """The auto-merge field must no longer exist on VecGraphConfig."""
        cfg = VecGraphConfig()
        assert not hasattr(cfg, "entity_name_dedup_threshold")

    def test_default_lm_lower_above_false_positive_band(self) -> None:
        """Default entity_lm_lower must sit above the ~0.83-0.88 nomic-embed
        similarity band for domain-co-occurring proper nouns."""
        cfg = VecGraphConfig()
        assert cfg.entity_lm_lower >= 0.90

    def test_entity_lm_top_k_default(self) -> None:
        cfg = VecGraphConfig()
        assert cfg.entity_lm_top_k == 5


# ---------------------------------------------------------------------------
# Below threshold — LLM must never be invoked
# ---------------------------------------------------------------------------

class TestEntityDedupBelowThreshold:
    """Entities whose name cosine-sim falls below entity_lm_lower are always
    created as separate entities without any LLM call."""

    def _ingest(self, extraction_json: str, lm_lower: float = 0.90):
        cfg = _make_config(lm_lower=lm_lower)
        with (
            patch("src.memory.model_vecgraph.litellm.completion") as mock_llm,
            patch(
                "src.memory.model_vecgraph.litellm.embedding",
                side_effect=_embed_fn(_ORTH_VECS),
            ),
            patch.object(VecGraphMemory, "_save_and_rebuild_index"),
        ):
            mock_llm.return_value = _completion(extraction_json)
            mem = VecGraphMemory(config=cfg, storage=InMemoryStorage())
            mem.ingest_documents(["Nikola Tesla was born in Smiljan."])
        return mem, mock_llm

    def test_both_entities_created(self) -> None:
        extraction = _triplets_json({
            "subject": "Nikola Tesla", "predicate": "was born in",
            "description": "was born in Smiljan in 1856", "object": "Smiljan",
        })
        mem, _ = self._ingest(extraction)
        assert mem.entity_count == 2
        assert "nikola tesla" in mem._entities
        assert "smiljan" in mem._entities

    def test_lm_confirmation_never_called(self) -> None:
        """max_tokens=4 is the signature of an entity-confirmation call."""
        extraction = _triplets_json({
            "subject": "Nikola Tesla", "predicate": "was born in",
            "description": "was born in Smiljan in 1856", "object": "Smiljan",
        })
        _, mock_llm = self._ingest(extraction)
        confirmation_calls = [
            c for c in mock_llm.call_args_list
            if c.kwargs.get("max_tokens") == 4
        ]
        assert confirmation_calls == []

    def test_triplet_preserves_correct_subject_and_object(self) -> None:
        """Core regression: subject and object must resolve to different entities."""
        extraction = _triplets_json({
            "subject": "Nikola Tesla", "predicate": "was born in",
            "description": "was born in Smiljan in 1856", "object": "Smiljan",
        })
        mem, _ = self._ingest(extraction)
        assert len(mem._triplets) == 1
        trip = mem._triplets[0]
        assert trip.subject == "Nikola Tesla"
        assert trip.object == "Smiljan"

    def test_lm_lower_1_completely_disables_lm_gate(self) -> None:
        """entity_lm_lower=1.0 means the LLM is never asked regardless of sim."""
        extraction = _triplets_json({
            "subject": "Nikola Tesla", "predicate": "also known as",
            "description": "also commonly referred to as Tesla", "object": "Tesla",
        })
        # Even though cos-sim(Tesla, Nikola Tesla)=0.95 > 0.90, lm_lower=1.0 wins
        cfg = _make_config(lm_lower=1.0)
        with (
            patch("src.memory.model_vecgraph.litellm.completion") as mock_llm,
            patch(
                "src.memory.model_vecgraph.litellm.embedding",
                side_effect=_embed_fn(_HIGH_SIM_VECS),
            ),
            patch.object(VecGraphMemory, "_save_and_rebuild_index"),
        ):
            mock_llm.return_value = _completion(extraction)
            mem = VecGraphMemory(config=cfg, storage=InMemoryStorage())
            mem.ingest_documents(["Nikola Tesla, also known as Tesla, was a genius."])

        confirmation_calls = [
            c for c in mock_llm.call_args_list
            if c.kwargs.get("max_tokens") == 4
        ]
        assert confirmation_calls == []
        assert mem.entity_count == 2


# ---------------------------------------------------------------------------
# At/above threshold — LLM gate
# ---------------------------------------------------------------------------

class TestEntityDedupLLMGate:
    """When cos-sim >= entity_lm_lower the LLM IS invoked; merge iff it says YES."""

    _EXTRACTION_ALIAS = _triplets_json({
        "subject": "Nikola Tesla", "predicate": "also known as",
        "description": "also commonly referred to as Tesla", "object": "Tesla",
    })

    def _ingest_high_sim(self, lm_answer: str):
        cfg = _make_config(lm_lower=0.90)
        with (
            patch("src.memory.model_vecgraph.litellm.completion") as mock_llm,
            patch(
                "src.memory.model_vecgraph.litellm.embedding",
                side_effect=_embed_fn(_HIGH_SIM_VECS),
            ),
            patch.object(VecGraphMemory, "_save_and_rebuild_index"),
        ):
            mock_llm.side_effect = [
                _completion(self._EXTRACTION_ALIAS),  # extraction call
                _completion(lm_answer),               # entity-selection call
            ]
            mem = VecGraphMemory(config=cfg, storage=InMemoryStorage())
            mem.ingest_documents(["Nikola Tesla, also known as Tesla, invented AC."])
        return mem, mock_llm

    def test_lm_selection_called_when_sim_above_threshold(self) -> None:
        _, mock_llm = self._ingest_high_sim("0")
        selection_calls = [
            c for c in mock_llm.call_args_list
            if c.kwargs.get("max_tokens") == 4
        ]
        assert len(selection_calls) == 1

    def test_lm_zero_keeps_entities_separate(self) -> None:
        mem, _ = self._ingest_high_sim("0")
        assert mem.entity_count == 2
        assert "nikola tesla" in mem._entities
        assert "tesla" in mem._entities
        assert "Tesla" not in mem._entities["nikola tesla"].aliases

    def test_lm_one_merges_candidate_into_existing(self) -> None:
        mem, _ = self._ingest_high_sim("1")
        assert mem.entity_count == 1
        assert "nikola tesla" in mem._entities

    def test_lm_one_alias_registered(self) -> None:
        mem, _ = self._ingest_high_sim("1")
        assert "Tesla" in mem._entities["nikola tesla"].aliases

    def test_lm_selects_second_candidate(self) -> None:
        """LLM can pick candidate 2, not just candidate 1.

        Two entities are pre-seeded in storage. Their cosine similarity to the
        new candidate name puts Marie Curie first (sim ≈ 0.994) and Nikola Tesla
        second (sim ≈ 0.912) in the sorted candidate list presented to the LLM.
        The LLM replies "2" → the candidate merges into Nikola Tesla.
        """
        from src.memory.storage.base import Entity

        vecs = {
            "Nikola Tesla": [1.0,  0.0,  0.0, 0.0],
            "Marie Curie":  [0.95, 0.31, 0.0, 0.0],
            # dot([0.91,0.41], [0.95,0.31]) ≈ 0.994  → Marie Curie listed first
            # dot([0.91,0.41], [1.0, 0.0])  ≈ 0.912  → Nikola Tesla listed second
            "candidate":    [0.91, 0.41, 0.0, 0.0],
            "Person":       [0.0,  0.0,  1.0, 0.0],
        }
        extraction = _triplets_json({
            "subject": "candidate", "predicate": "is",
            "description": "is a notable person", "object": "Person",
        })

        # Pre-seed storage so both entities exist before ingest
        storage = InMemoryStorage()
        storage._entities = {
            "nikola tesla": Entity(name="Nikola Tesla"),
            "marie curie":  Entity(name="Marie Curie"),
        }

        cfg = _make_config(lm_lower=0.90)
        with (
            patch("src.memory.model_vecgraph.litellm.completion") as mock_llm,
            patch(
                "src.memory.model_vecgraph.litellm.embedding",
                side_effect=_embed_fn(vecs),
            ),
            patch.object(VecGraphMemory, "_save_and_rebuild_index"),
        ):
            mock_llm.side_effect = [
                _completion(extraction),  # extraction call
                _completion("2"),         # selection: Nikola Tesla is second candidate
            ]
            mem = VecGraphMemory(config=cfg, storage=storage)
            mem.ingest_documents(["candidate is a notable person."])

        # "candidate" merged into "Nikola Tesla" (the second candidate shown to LLM)
        assert "candidate" not in mem._entities
        assert "candidate" in mem._entities["nikola tesla"].aliases

    def test_lm_yes_triplet_uses_canonical_name(self) -> None:
        """After merge the triplet subject must carry the canonical name, not the alias.

        Two triplets are ingested:
          (Nikola Tesla, was a, …, Inventor)      ← establishes the canonical entity
          (Tesla,        invented, …, AC System)  ← Tesla merges into Nikola Tesla
        The second triplet's subject must resolve to "Nikola Tesla".
        """
        extraction = _triplets_json(
            {
                "subject": "Nikola Tesla", "predicate": "was a",
                "description": "was a Serbian-American inventor", "object": "Inventor",
            },
            {
                "subject": "Tesla", "predicate": "invented",
                "description": "invented the alternating current electrical system",
                "object": "AC System",
            },
        )
        cfg = _make_config(lm_lower=0.90)
        with (
            patch("src.memory.model_vecgraph.litellm.completion") as mock_llm,
            patch(
                "src.memory.model_vecgraph.litellm.embedding",
                side_effect=_embed_fn(_HIGH_SIM_VECS),
            ),
            patch.object(VecGraphMemory, "_save_and_rebuild_index"),
        ):
            mock_llm.side_effect = [
                _completion(extraction),  # extraction
                _completion("1"),         # Tesla == Nikola Tesla (only candidate)
            ]
            mem = VecGraphMemory(config=cfg, storage=InMemoryStorage())
            mem.ingest_documents(["Nikola Tesla was an inventor. Tesla invented AC."])

        tesla_triplet = next(t for t in mem._triplets if "invented" in t.predicate)
        assert tesla_triplet.subject == "Nikola Tesla"
        assert tesla_triplet.object == "AC System"


# ---------------------------------------------------------------------------
# Original bug regression — full Tesla scenario
# ---------------------------------------------------------------------------

class TestOriginalTeslaBugRegression:
    """End-to-end regression for the exact entity collapse reported in the bug:
    Smiljan, Zagreb, Croatia, and ACES all landing as aliases of Nikola Tesla."""

    _EXTRACTION = _triplets_json(
        {
            "subject": "Nikola Tesla", "predicate": "was born in",
            "description": "was born in Smiljan, a village in modern-day Croatia",
            "object": "Smiljan",
        },
        {
            "subject": "Zagreb", "predicate": "is capital of",
            "description": "is the capital city of Croatia",
            "object": "Croatia",
        },
        {
            "subject": "Nikola Tesla", "predicate": "invented",
            "description": "invented the alternating current electrical system",
            "object": "Alternating Current Electrical System",
        },
    )

    def _ingest(self) -> tuple[VecGraphMemory, MagicMock]:
        cfg = _make_config(lm_lower=0.90)
        with (
            patch("src.memory.model_vecgraph.litellm.completion") as mock_llm,
            patch(
                "src.memory.model_vecgraph.litellm.embedding",
                side_effect=_embed_fn(_ORTH_VECS),
            ),
            patch.object(VecGraphMemory, "_save_and_rebuild_index"),
        ):
            mock_llm.return_value = _completion(self._EXTRACTION)
            mem = VecGraphMemory(config=cfg, storage=InMemoryStorage())
            mem.ingest_documents([
                "Nikola Tesla was born in Smiljan, a village in Croatia. "
                "Zagreb is the capital of Croatia. "
                "Tesla invented the alternating current electrical system."
            ])
        return mem, mock_llm

    def test_all_five_entities_created(self) -> None:
        mem, _ = self._ingest()
        names = {e.name for e in mem._entities.values()}
        expected = {
            "Nikola Tesla", "Smiljan", "Zagreb",
            "Croatia", "Alternating Current Electrical System",
        }
        assert expected.issubset(names)

    def test_nikola_tesla_has_no_unrelated_aliases(self) -> None:
        mem, _ = self._ingest()
        tesla = mem._entities["nikola tesla"]
        bad = {"Smiljan", "Zagreb", "Croatia", "Alternating Current Electrical System"}
        assert not bad.intersection(set(tesla.aliases))

    def test_triplets_have_correct_distinct_subjects_and_objects(self) -> None:
        """No triplet may have subject == object == 'Nikola Tesla' due to a false merge."""
        mem, _ = self._ingest()
        for trip in mem._triplets:
            assert not (trip.subject == trip.object == "Nikola Tesla"), (
                f"Self-loop triplet detected (entity collapse bug): {trip}"
            )

    def test_born_in_triplet_object_is_smiljan(self) -> None:
        mem, _ = self._ingest()
        born_in = next(t for t in mem._triplets if "born" in t.description)
        assert born_in.subject == "Nikola Tesla"
        assert born_in.object == "Smiljan"

    def test_capital_triplet_subject_is_zagreb(self) -> None:
        mem, _ = self._ingest()
        capital = next(t for t in mem._triplets if "capital" in t.description)
        assert capital.subject == "Zagreb"
        assert capital.object == "Croatia"

    def test_no_lm_confirmation_called_for_orthogonal_entities(self) -> None:
        """With orthogonal name vectors (sim=0) the LLM gate must never fire."""
        _, mock_llm = self._ingest()
        confirmation_calls = [
            c for c in mock_llm.call_args_list
            if c.kwargs.get("max_tokens") == 4
        ]
        assert confirmation_calls == []
