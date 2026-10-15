"""PostgreSQL-backed Dense/Sparse/Hybrid retrieval."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Any

import numpy as np

from apex_rag.db import PostgresStore
from apex_rag.embeddings import create_embedder
from apex_rag.models import Chunk, SearchResult
from apex_rag.retrieval.reranker import CrossEncoderReranker, LexicalReranker, Reranker


class PostgresRetrievalEngine:
    """Explicit PostgreSQL retrieval engine matching the local engine interface."""

    def __init__(self, database_url: str, embedding_model: str, reranker_model: str,
                 rrf_k: int = 60, hnsw_ef_search: int = 40,
                 use_model_reranker: bool = True, table_name: str = "documents",
                 index_type: str = "hnsw", ivfflat_probes: int = 1) -> None:
        self.store = PostgresStore(database_url, table_name=table_name)
        self.index_type = index_type.lower()
        if self.index_type == "hnsw":
            self.store.set_hnsw_search(hnsw_ef_search)
        elif self.index_type == "ivfflat":
            self.store.set_ivfflat_probes(ivfflat_probes)
        else:
            raise ValueError("index_type must be hnsw or ivfflat")
        self.embedder = create_embedder(embedding_model)
        self.rrf_k = rrf_k
        if use_model_reranker:
            try:
                self.reranker: Reranker = CrossEncoderReranker(reranker_model)
            except RuntimeError:
                self.reranker = LexicalReranker()
        else:
            self.reranker = LexicalReranker()

    def close(self) -> None:
        """Close the PostgreSQL connection."""
        self.store.close()

    @staticmethod
    def _filter_sql(filters: dict[str, str | int] | None) -> tuple[str, list[Any]]:
        """Build a safe metadata predicate from allow-listed columns."""
        if not filters:
            return "", []
        allowed = {"category", "card_tier", "effective_year"}
        clauses: list[str] = []
        params: list[Any] = []
        for key, value in filters.items():
            if key in allowed:
                clauses.append(f"{key} = %s")
                params.append(value)
        return (" AND " + " AND ".join(clauses)) if clauses else "", params

    @staticmethod
    def _row_to_result(row: tuple[Any, ...], score_name: str) -> SearchResult:
        """Convert a database result row to the shared result model."""
        chunk = Chunk(
            chunk_id=str(row[0]), parent_doc_id=str(row[1]), title=str(row[2]),
            category=str(row[3] or ""), card_tier=str(row[4] or ""),
            effective_year=int(row[5]), content=str(row[6]), strategy=str(row[8] or "original"),
        )
        result = SearchResult(chunk=chunk)
        setattr(result, f"{score_name}_score", float(row[7] or 0.0))
        return result

    def dense_search(self, query: str, top_k: int = 20,
                     filters: dict[str, str | int] | None = None) -> list[SearchResult]:
        """Run pgvector cosine search."""
        embedding = self.embedder.encode([query])[0]
        predicate, filter_params = self._filter_sql(filters)
        sql = (
            "SELECT chunk_id, parent_doc_id, title, category, card_tier, effective_year, content, "
            f"1 - (embedding <=> %s) AS score, chunking_strategy FROM {self.store.table_name} "
            f"WHERE embedding IS NOT NULL{predicate} ORDER BY embedding <=> %s LIMIT %s"
        )
        params = [embedding, *filter_params, embedding, top_k]
        with self.store.connection.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        return [replace(self._row_to_result(row, "dense"), rank=index + 1) for index, row in enumerate(rows)]

    def sparse_search(self, query: str, top_k: int = 20,
                      filters: dict[str, str | int] | None = None) -> list[SearchResult]:
        """Run PostgreSQL English full-text search."""
        predicate, filter_params = self._filter_sql(filters)
        sql = (
            "SELECT chunk_id, parent_doc_id, title, category, card_tier, effective_year, content, "
            "ts_rank_cd(search_vector, websearch_to_tsquery('english', %s)) AS score, chunking_strategy "
            f"FROM {self.store.table_name} WHERE search_vector @@ websearch_to_tsquery('english', %s)"
            f"{predicate} ORDER BY score DESC LIMIT %s"
        )
        params = [query, query, *filter_params, top_k]
        with self.store.connection.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        return [replace(self._row_to_result(row, "sparse"), rank=index + 1) for index, row in enumerate(rows)]

    def hybrid_search(self, query: str, top_k: int = 20, final_top_n: int = 5,
                      rerank: bool = True,
                      filters: dict[str, str | int] | None = None) -> list[SearchResult]:
        """Fuse pgvector and PostgreSQL full-text rankings using RRF."""
        dense = self.dense_search(query, top_k, filters)
        sparse = self.sparse_search(query, top_k, filters)
        merged: dict[str, SearchResult] = {}
        dense_by_id = {result.chunk.chunk_id: result for result in dense}
        sparse_by_id = {result.chunk.chunk_id: result for result in sparse}
        for result in [*dense, *sparse]:
            key = result.chunk.chunk_id
            if key not in merged:
                merged[key] = replace(result)
            item = merged[key]
            item.dense_score = dense_by_id.get(key, result).dense_score
            item.sparse_score = sparse_by_id.get(key, result).sparse_score
            ranks = [mapping[key].rank for mapping in (dense_by_id, sparse_by_id) if key in mapping]
            item.rrf_score = sum(1.0 / (self.rrf_k + rank) for rank in ranks)
        candidates = sorted(merged.values(), key=lambda result: result.rrf_score, reverse=True)[:top_k]
        if rerank:
            candidates = self.reranker.score(query, candidates)
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
