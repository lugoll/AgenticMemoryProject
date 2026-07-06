from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class _BeamPath:
    """One reasoning chain grown outward from a single seed entity.

    tail    — current node; the next hop expands from here.
    nodes   — every entity visited on THIS path (cycle prevention).
    triples — ordered "s p o" display strings, one per hop.
    origin  — the seed the path started from (provenance, step 1; kept for the
              future collision/bridge work, unused by the ranking today).
    src     — node the final hop expanded FROM (fan-out cap key); "" for seeds.
    """

    tail: str
    nodes: frozenset[str]
    triples: tuple[str, ...]
    origin: str
    src: str = ""

    @property
    def chain_text(self) -> str:
        """The whole chain as one string — the unit the cross-encoder scores."""
        return ". ".join(self.triples)

    def extend(self, triple: str, neighbor: str) -> "_BeamPath":
        return _BeamPath(
            tail=neighbor,
            nodes=self.nodes | {neighbor},
            triples=self.triples + (triple,),
            origin=self.origin,
            src=self.tail,
        )


# Edge row shape returned by ``fetch_edges`` — mirrors the Cypher projection so the
# Neo4j wrapper and the test fakes agree. ``anchor`` is the frontier node the edge was
# fetched for (maps the edge back to the path whose ``tail == anchor``); ``neighbor``
# is the other endpoint and becomes the extended path's new tail. ``s p o`` render the
# triple in its stored direction.
FetchEdges = Callable[[list[str]], Iterable[dict[str, Any]]]
ScoreChains = Callable[[str, list[str]], list[float]]


@dataclass
class _BeamConfig:
    """The subset of graph config the beam traversal needs (duck-typed from GraphCfg)."""

    max_hops: int
    beam_width: int
    max_per_tail: int
    rerank_top_n: int


def _triple_display(s: str, predicate: str, o: str) -> str:
    """Render a stored triple as human/reranker-readable text.

    Predicates are stored verbatim as snake_case relationship types; ``.lower()`` is
    a safety net in case a future store version normalises the casing.
    """
    predicate = (predicate or "related_to").lower()
    return f"{s} {predicate.replace('_', ' ')} {o}"


def _beam_search(
    query: str,
    seed_nodes: list[str],
    g: _BeamConfig,
    fetch_edges: FetchEdges,
    score_chains: ScoreChains,
) -> tuple[list[str], dict]:
    """Beam-guided, chain-scored graph traversal (used by the vectorgraph variant).

    Grows reasoning chains outward from the seeds and, at each hop, scores the *whole
    chain* against the query and keeps a per-node-capped top-K beam. Scoring chains
    (not single triples) lets a triple be kept for *completing* a relevant path;
    capping per node stops a high-degree hub from filling the beam with near-identical
    siblings in either direction (fan-in by tail, fan-out by source). The eval settled
    this as the best configuration for vectorgraph's noisy chunk-anchored seeds; the
    graph variant keeps the simpler flat single-triple rerank (BaseMemory._expand_triples).

    Output is the deduplicated individual triples of the top chains, NOT the chains
    themselves: emitting full chains repeats shared triples and that run-on redundancy
    measurably degrades the small agent model (verified — it refused questions whose
    answer was present).

    All I/O is injected so the algorithm unit-tests without Neo4j or a GPU:
      ``fetch_edges(tails)``   → one batched edge fetch per hop.
      ``score_chains(q, [c])`` → cross-encoder relevance of each chain to the query.
    """
    seeds = [s for s in seed_nodes if s]
    stats = {
        "cypher_queries": 0,
        "paths_scored": 0,
        "paths_kept": 0,
        "hops": 0,
        "bfs_ms": 0.0,
        "rerank_ms": 0.0,
    }
    if not seeds:
        return [], stats

    # Zero-length paths: one per seed, no edge yet.
    beam = [_BeamPath(tail=s, nodes=frozenset({s}), triples=(), origin=s) for s in seeds]

    # Output pool keyed by the path's triple tuple so identical chains reached from
    # different seeds collapse to one entry, keeping the best score seen.
    pool: dict[tuple[str, ...], tuple[float, _BeamPath]] = {}

    for hop in range(1, g.max_hops + 1):
        tails = sorted({p.tail for p in beam})
        if not tails:
            break

        t0 = time.perf_counter()
        rows = list(fetch_edges(tails))
        stats["bfs_ms"] += (time.perf_counter() - t0) * 1000
        stats["cypher_queries"] += 1
        stats["hops"] = hop

        # Group edges by the frontier node they were fetched for.
        edges_by_anchor: dict[str, list[tuple[str, str]]] = {}
        for row in rows:
            s, o = row["s"], row["o"]
            if s is None or o is None:
                continue
            edges_by_anchor.setdefault(row["anchor"], []).append(
                (_triple_display(s, row["p"], o), row["neighbor"])
            )

        # Extend every live path by every edge of its tail (skip cycles).
        candidates: list[_BeamPath] = []
        for p in beam:
            for triple, neighbor in edges_by_anchor.get(p.tail, []):
                if neighbor is not None and neighbor in p.nodes:
                    continue
                candidates.append(p.extend(triple, neighbor))
        if not candidates:
            break

        t0 = time.perf_counter()
        scores = score_chains(query, [c.chain_text for c in candidates])
        stats["rerank_ms"] += (time.perf_counter() - t0) * 1000
        stats["paths_scored"] += len(candidates)

        ranked = sorted(zip(candidates, scores), key=lambda cs: cs[1], reverse=True)

        # Per-node-capped top-K: a hub can contribute at most ``max_per_tail`` paths in
        # EITHER direction, so it cannot fill the beam and starve the answer:
        #   - by tail (``cand.tail``): bounds fan-IN, many paths converging on a hub.
        #   - by source (``cand.src``): bounds fan-OUT, one hub tail exploding into its
        #     dozens of neighbours (the "guest → VH1 → other guest" hub-walk that
        #     otherwise dominates, since each such path ends at a distinct tail).
        next_beam: list[_BeamPath] = []
        per_tail: Counter[str] = Counter()
        per_src: Counter[str] = Counter()
        for cand, score in ranked:
            if per_tail[cand.tail] >= g.max_per_tail or per_src[cand.src] >= g.max_per_tail:
                continue
            next_beam.append(cand)
            per_tail[cand.tail] += 1
            per_src[cand.src] += 1
            key = cand.triples
            if key not in pool or score > pool[key][0]:
                pool[key] = (score, cand)
            if len(next_beam) >= g.beam_width:
                break
        beam = next_beam

    triples = _finalize(pool, g.rerank_top_n)
    stats["paths_kept"] = len(triples)
    return triples, stats


def _finalize(
    pool: dict[tuple[str, ...], tuple[float, _BeamPath]], top_n: int
) -> list[str]:
    """Flatten the ranked chains into a deduplicated list of individual triples.

    Ranking is a plain sort of the pooled chain scores — comparable across hops (same
    model, same query) so no second rerank pass. Each chain contributes its triples in
    traversal order, first occurrence wins, capped at ``top_n`` triples.
    """
    ordered = sorted(pool.values(), key=lambda sp: sp[0], reverse=True)

    seen: set[str] = set()
    out: list[str] = []
    for _score, path in ordered:
        for triple in path.triples:
            if triple in seen:
                continue
            seen.add(triple)
            out.append(triple)
            if len(out) >= top_n:
                return out
    return out
