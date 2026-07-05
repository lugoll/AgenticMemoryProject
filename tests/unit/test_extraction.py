from __future__ import annotations

from src.memory.base import _chunk_text
from src.memory.extraction import (
    _ALLOWED_PREDICATES_SET,
    _normalize_predicate,
    parse_triples,
)
from src.memory.model_bm25 import _sanitise_query


class TestParseTriples:
    def test_valid_json(self) -> None:
        raw = '{"triples": [{"subject": "A", "predicate": "born_in", "object": "B"}]}'
        assert parse_triples(raw) == [
            {"subject": "A", "predicate": "born_in", "object": "B"}
        ]

    def test_truncated_json_regex_fallback(self) -> None:
        raw = (
            '{"triples": [{"subject": "A", "predicate": "born_in", "object": "B"},'
            ' {"subject": "C", "pred'
        )
        assert parse_triples(raw) == [
            {"subject": "A", "predicate": "born_in", "object": "B"}
        ]

    def test_garbage_returns_empty(self) -> None:
        assert parse_triples("not json at all") == []
        assert parse_triples("") == []

    def test_list_object_is_split(self) -> None:
        raw = (
            '{"triples": [{"subject": "Film X", "predicate": "starred_in",'
            ' "object": "Actor A, Actor B, Actor C"}]}'
        )
        triples = parse_triples(raw)
        assert [t["object"] for t in triples] == ["Actor A", "Actor B", "Actor C"]

    def test_short_pair_object_kept_intact(self) -> None:
        raw = (
            '{"triples": [{"subject": "A", "predicate": "located_in",'
            ' "object": "New York, USA"}]}'
        )
        assert parse_triples(raw)[0]["object"] == "New York, USA"

    def test_invented_predicate_is_normalised(self) -> None:
        raw = (
            '{"triples": [{"subject": "A", "predicate": "executive produced",'
            ' "object": "B"}]}'
        )
        assert parse_triples(raw)[0]["predicate"] == "produced_by"


class TestNormalizePredicate:
    def test_exact_match_passthrough(self) -> None:
        assert _normalize_predicate("born_in", _ALLOWED_PREDICATES_SET) == "born_in"

    def test_alias(self) -> None:
        assert _normalize_predicate("studied_at", _ALLOWED_PREDICATES_SET) == "educated_at"

    def test_token_overlap(self) -> None:
        assert _normalize_predicate("born_within", _ALLOWED_PREDICATES_SET) in _ALLOWED_PREDICATES_SET

    def test_no_overlap_falls_back(self) -> None:
        assert _normalize_predicate("zzz_qqq", _ALLOWED_PREDICATES_SET) == "related_to"


class TestLuceneSanitiseQuery:
    def test_specials_stripped_tokens_or_joined(self) -> None:
        assert _sanitise_query('who is "Bob"? (really)') == '"who" OR "is" OR "Bob" OR "really"'

    def test_empty_after_stripping(self) -> None:
        assert _sanitise_query('?*:"') == ""

    def test_plain_words(self) -> None:
        assert _sanitise_query("hotel designer") == '"hotel" OR "designer"'


class TestChunkText:
    def test_short_text_single_chunk(self) -> None:
        assert _chunk_text("a b c", chunk_size=10, chunk_overlap=2) == ["a b c"]

    def test_overlapping_chunks(self) -> None:
        words = " ".join(str(i) for i in range(10))
        chunks = _chunk_text(words, chunk_size=4, chunk_overlap=2)
        assert chunks[0] == "0 1 2 3"
        assert chunks[1] == "2 3 4 5"

    def test_empty_text(self) -> None:
        assert _chunk_text("   ", chunk_size=4, chunk_overlap=2) == []
