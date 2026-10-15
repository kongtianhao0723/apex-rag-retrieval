"""Pluggable second-stage rerankers."""

from __future__ import annotations

from typing import Protocol

from apex_rag.models import SearchResult


class Reranker(Protocol):
    """Protocol for reranking a candidate list."""

    def score(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Return candidates ordered by reranker score."""


class CrossEncoderReranker:
    """Sentence Transformers Cross-Encoder adapter."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "Install the models extra to use Cross-Encoder reranking: "
                "python -m pip install -e '.[models]'"
            ) from exc
        self.model = CrossEncoder(model_name)

    def score(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Score query/passage pairs and sort descending."""
        pairs = [(query, result.chunk.content) for result in results]
        scores = self.model.predict(pairs)
        for result, score in zip(results, scores):
            result.rerank_score = float(score)
        return sorted(results, key=lambda result: result.rerank_score, reverse=True)


class LexicalReranker:
    """Deterministic fallback used only when model dependencies are unavailable."""

    def score(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Score token overlap for offline smoke tests."""
        query_tokens = set(query.lower().split())
        max_rrf = max((result.rrf_score for result in results), default=1.0)
        for result in results:
            passage_tokens = set(result.chunk.content.lower().split())
            overlap = len(query_tokens & passage_tokens) / max(len(query_tokens), 1)
            result.rerank_score = 0.7 * (result.rrf_score / max_rrf) + 0.3 * overlap
        return sorted(results, key=lambda result: (result.rerank_score, result.rrf_score), reverse=True)
