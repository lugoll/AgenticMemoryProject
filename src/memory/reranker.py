from __future__ import annotations

import logging

from src.config.cfg import Config, resolve_device

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-encoder reranking shared by every retrieval variant.

    Scores (query, candidate) pairs jointly with a sentence-transformers
    CrossEncoder and keeps the ``graph.rerank_top_n`` most relevant candidates.
    No LLM call and zero tokens → preserves the zero-cost-retrieval property;
    the wall-clock cost is reported by the callers via retrieval_overhead
    telemetry events.

    One implementation for all variants keeps the ranking stage identical
    across the comparison (cross-variant validity). The model is loaded lazily
    on the device from ``config.device`` (auto → cuda when available), so
    ingest runs and variants that never rerank skip the model-load cost.
    """

    def __init__(self, config: Config) -> None:
        self._model_name: str = config.graph.rerank_model
        self._top_n: int = config.graph.rerank_top_n
        self._device: str = resolve_device(config.device)
        self._model = None  # lazy CrossEncoder

    def score(self, query: str, candidates: list[str]) -> list[float]:
        """Cross-encoder relevance of each candidate to the query (no truncation).

        Used by the beam traversal, which needs raw per-chain scores to prune its
        frontier; ``rerank`` layers sort+truncate on top.
        """
        if not candidates:
            return []
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info(
                "CrossEncoderReranker: loading %s on %s", self._model_name, self._device
            )
            self._model = CrossEncoder(self._model_name, device=self._device)
        return [float(s) for s in self._model.predict([(query, c) for c in candidates])]

    def rerank(self, query: str, candidates: list[str]) -> list[str]:
        """Return the top_n candidates ranked by cross-encoder relevance to query."""
        scores = self.score(query, candidates)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[: self._top_n]]
