from __future__ import annotations

import re

from src.memory.traversal import _BeamConfig, _beam_search


# ── test doubles ──────────────────────────────────────────────────────────────

def _make_fetch(edges: list[tuple[str, str, str]]):
    """Fake ``fetch_edges``: return the rows _beam_search expects for the given tails.

    Traversal is undirected, so an edge (s, p, o) is incident to both s and o. The
    row's ``anchor`` is the frontier node it was fetched for and ``neighbor`` is the
    other endpoint (the extended path's next tail); ``s p o`` keep their stored order.
    """
    def fetch(tails: list[str]) -> list[dict]:
        wanted = set(tails)
        rows: list[dict] = []
        for s, p, o in edges:
            if s in wanted and s != o:
                rows.append({"anchor": s, "s": s, "p": p, "o": o, "neighbor": o})
            if o in wanted and s != o:
                rows.append({"anchor": o, "s": s, "p": p, "o": o, "neighbor": s})
        return rows

    return fetch


def _tokenize(text: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9]", " ", text.lower()).split())


def _token_overlap(query: str, chains: list[str]) -> list[float]:
    """Stand-in for the cross-encoder: |query ∩ chain| token overlap."""
    q = _tokenize(query)
    return [float(len(q & _tokenize(c))) for c in chains]


def _cfg(max_hops=3, beam_width=10, max_per_tail=3, rerank_top_n=10) -> _BeamConfig:
    return _BeamConfig(
        max_hops=max_hops,
        beam_width=beam_width,
        max_per_tail=max_per_tail,
        rerank_top_n=rerank_top_n,
    )


QUERY = "guest on VH1 Big in 06 known by nickname New York"


# ── tests ─────────────────────────────────────────────────────────────────────

class TestBeamSearch:
    def test_bridge_chain_assembles_from_two_seeds(self) -> None:
        """The 2-hop chain touching both query entities beats the hub siblings.

        Tiffany is reachable only by hopping: from the ``New York`` seed via
        ``known_as`` and from the ``VH1`` seed via ``appeared_in``. Only her chain
        matches the whole query, so it must appear even though five other guests
        share the identical "appeared in VH1" edge.
        """
        edges = [
            ("Tiffany Pollard", "known_as", "New York"),
            ("Tiffany Pollard", "appeared_in", "VH1 Big in '06 Awards"),
            ("Guest A", "appeared_in", "VH1 Big in '06 Awards"),
            ("Guest B", "appeared_in", "VH1 Big in '06 Awards"),
            ("Guest C", "appeared_in", "VH1 Big in '06 Awards"),
            ("Guest D", "appeared_in", "VH1 Big in '06 Awards"),
            ("Guest E", "appeared_in", "VH1 Big in '06 Awards"),
        ]
        seeds = ["New York", "VH1 Big in '06 Awards",
                 "Guest A", "Guest B", "Guest C", "Guest D", "Guest E"]

        triples, stats = _beam_search(
            QUERY, seeds, _cfg(), _make_fetch(edges), _token_overlap
        )

        # Output is deduplicated triples flattened from the winning chains; the two
        # facts that make Tiffany the answer must both be present.
        assert "Tiffany Pollard known as New York" in triples
        assert "Tiffany Pollard appeared in VH1 Big in '06 Awards" in triples
        assert stats["hops"] >= 2
        assert stats["paths_kept"] == len(triples)

    def test_per_node_cap_limits_hub(self) -> None:
        """A hub cannot contribute more than max_per_tail paths (fan-in or fan-out)."""
        # Fan-in: 10 seeds all converge on one hub tail.
        hub = "Hub"
        edges = [(f"Seed {i}", "related_to", hub) for i in range(10)]
        seeds = [f"Seed {i}" for i in range(10)]

        triples, _ = _beam_search(
            QUERY, seeds, _cfg(max_hops=1, max_per_tail=3),
            _make_fetch(edges), lambda q, cs: [1.0] * len(cs),  # constant scores
        )
        # All 10 candidate paths share tail=Hub; the cap keeps 3 → 3 distinct triples.
        assert len(triples) == 3
        assert all(t.endswith("related to Hub") for t in triples)

    def test_cycle_is_not_revisited(self) -> None:
        """A path never steps back onto a node it already contains."""
        edges = [("A", "rel", "B"), ("B", "rel", "A")]
        triples, _ = _beam_search(
            "A B", ["A"], _cfg(max_hops=3),
            _make_fetch(edges), _token_overlap,
        )
        # A→B is fine; B→A would revisit A, so the walk terminates — no chain grows
        # unboundedly and the flattened output stays deduplicated.
        assert triples == list(dict.fromkeys(triples))
        assert all("rel" in t for t in triples)

    def test_output_triples_are_deduplicated(self) -> None:
        """Triples shared across overlapping chains appear once in the output."""
        edges = [
            ("Alice", "born_in", "Paris"),
            ("Paris", "capital_of", "France"),
        ]
        triples, _ = _beam_search(
            "Alice born in Paris capital of France", ["Alice"],
            _cfg(), _make_fetch(edges), _token_overlap,
        )
        assert "Alice born in Paris" in triples
        assert "Paris capital of France" in triples
        # "Alice born in Paris" is a prefix of the 2-hop chain and also its own 1-hop
        # chain, but it is emitted exactly once.
        assert triples.count("Alice born in Paris") == 1

    def test_empty_seeds_returns_empty(self) -> None:
        triples, stats = _beam_search(
            QUERY, [], _cfg(), _make_fetch([]), _token_overlap
        )
        assert triples == []
        assert stats["paths_kept"] == 0

    def test_no_edges_returns_empty(self) -> None:
        triples, _ = _beam_search(
            QUERY, ["Lonely"], _cfg(), _make_fetch([]), _token_overlap
        )
        assert triples == []
