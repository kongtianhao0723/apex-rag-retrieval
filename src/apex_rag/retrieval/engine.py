"""Local retrieval engine used for reproducible evaluation and API fallback.

The engine uses TF-IDF for sparse retrieval and a truncated SVD projection as a
dependency-light dense baseline. When Sentence Transformers is installed, the
embedding adapter can be swapped in without changing the retrieval interface.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import replace
from time import perf_counter
from typing import Iterable

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from apex_rag.models import Chunk, SearchResult


def _normalize_text(text: str) -> str:
    """Normalize policy text for the local lexical index."""
    return re.sub(r"[^a-z0-9%.$-]+", " ", text.lower()).strip()


class RetrievalEngine:
    """A deterministic local Dense/Sparse/Hybrid retrieval implementation."""

    def __init__(self, chunks: list[Chunk], rrf_k: int = 60) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required")
        self.chunks = chunks
        self.rrf_k = rrf_k
        corpus = [_normalize_text(f"{c.title} {c.category} {c.card_tier} {c.content}") for c in chunks]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.sparse_matrix = normalize(self.vectorizer.fit_transform(corpus))
        max_components = max(1, min(128, self.sparse_matrix.shape[1] - 1, len(chunks) - 1))
        if max_components >= 1 and self.sparse_matrix.shape[1] > 1 and len(chunks) > 2:
            self.svd = TruncatedSVD(n_components=max_components, random_state=42)
            self.dense_matrix = normalize(self.svd.fit_transform(self.sparse_matrix))
        else:
            self.svd = None
            self.dense_matrix = self.sparse_matrix.toarray()

    def _query_vectors(self, query: str) -> tuple[np.ndarray, np.ndarray]:
        sparse = normalize(self.vectorizer.transform([_normalize_text(query)]))
        if self.svd is None:
            dense = sparse.toarray()
        else:
            dense = normalize(self.svd.transform(sparse))
        return dense[0], sparse

    @staticmethod
    def _filter(chunk: Chunk, filters: dict[str, str | int] | None) -> bool:
        if not filters:
            return True
        return all(
            str(getattr(chunk, key, "")) == str(value)
            for key, value in filters.items()
            if key in {"category", "card_tier", "effective_year"}
        )

    def dense_search(self, query: str, top_k: int = 20, filters: dict[str, str | int] | None = None) -> list[SearchResult]:
        """Return cosine-ranked dense candidates."""
        dense_query, _ = self._query_vectors(query)
        scores = self.dense_matrix @ dense_query
        indexes = [i for i, c in enumerate(self.chunks) if self._filter(c, filters)]
        indexes.sort(key=lambda i: float(scores[i]), reverse=True)
        return [SearchResult(self.chunks[i], rank + 1, dense_score=float(scores[i])) for rank, i in enumerate(indexes[:top_k])]

    def sparse_search(self, query: str, top_k: int = 20, filters: dict[str, str | int] | None = None) -> list[SearchResult]:
        """Return TF-IDF-ranked sparse candidates."""
        _, sparse_query = self._query_vectors(query)
        scores = (self.sparse_matrix @ sparse_query.T).toarray().ravel()
        indexes = [i for i, c in enumerate(self.chunks) if self._filter(c, filters)]
        indexes.sort(key=lambda i: float(scores[i]), reverse=True)
        return [SearchResult(self.chunks[i], rank + 1, sparse_score=float(scores[i])) for rank, i in enumerate(indexes[:top_k])]

    def hybrid_search(self, query: str, top_k: int = 20, final_top_n: int = 5,
                      rerank: bool = True, filters: dict[str, str | int] | None = None) -> list[SearchResult]:
        """Fuse dense and sparse rankings with RRF and optionally rerank."""
        dense = self.dense_search(query, top_k, filters)
        sparse = self.sparse_search(query, top_k, filters)
        merged: dict[str, SearchResult] = {}
        dense_by_id = {r.chunk.chunk_id: r for r in dense}
        sparse_by_id = {r.chunk.chunk_id: r for r in sparse}
        for result in [*dense, *sparse]:
            key = result.chunk.chunk_id
            if key not in merged:
                merged[key] = replace(result)
            item = merged[key]
            item.dense_score = dense_by_id.get(key, result).dense_score
            item.sparse_score = sparse_by_id.get(key, result).sparse_score
            dense_rank = dense_by_id.get(key).rank if key in dense_by_id else None
            sparse_rank = sparse_by_id.get(key).rank if key in sparse_by_id else None
            item.rrf_score = sum(1.0 / (self.rrf_k + rank) for rank in (dense_rank, sparse_rank) if rank)
        candidates = sorted(merged.values(), key=lambda r: r.rrf_score, reverse=True)[:top_k]
        if rerank:
            query_tokens = set(_normalize_text(query).split())
            max_rrf = max((result.rrf_score for result in candidates), default=1.0)
            for result in candidates:
                text_tokens = set(_normalize_text(f"{result.chunk.title} {result.chunk.content}").split())
                overlap = len(query_tokens & text_tokens) / max(len(query_tokens), 1)
                result.rerank_score = 0.7 * (result.rrf_score / max_rrf) + 0.3 * overlap
            candidates.sort(key=lambda r: (r.rerank_score, r.rrf_score), reverse=True)
        return [replace(result, rank=index + 1) for index, result in enumerate(candidates[:final_top_n])]

    def timed_search(self, query: str, mode: str, **kwargs: object) -> tuple[list[SearchResult], float]:
        """Run a search and return elapsed milliseconds."""
        start = perf_counter()
        if mode == "dense":
            kwargs.pop("final_top_n", None)
            results = self.dense_search(query, **kwargs)
        elif mode == "hybrid":
            results = self.hybrid_search(query, rerank=False, **kwargs)
        elif mode == "enhanced":
            results = self.hybrid_search(query, rerank=True, **kwargs)
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")
        return results, (perf_counter() - start) * 1000
