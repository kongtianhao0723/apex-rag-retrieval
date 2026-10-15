"""Controlled chunking and PostgreSQL index experiments.

The runner deliberately uses the supplied documents and evaluation queries only.
Each configuration is isolated in a temporary PostgreSQL table so experiments do
not change the production ``documents`` table used by the browser demo.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from apex_rag.db import PostgresStore
from apex_rag.embeddings import create_embedder
from apex_rag.ingestion.chunking import (
    hierarchical_chunks,
    original_chunks,
    sliding_window_chunks,
)
from apex_rag.models import Chunk, SearchResult
from apex_rag.retrieval.postgres import PostgresRetrievalEngine


@dataclass(frozen=True)
class IndexConfig:
    """One reproducible PostgreSQL vector-index configuration."""

    name: str
    index_type: str
    m: int = 16
    ef_construction: int = 128
    ef_search: int = 80
    lists: int = 10
    probes: int = 1


CHUNK_INDEX = IndexConfig("hnsw_m16_efc128_efs80", "hnsw")
INDEX_CONFIGS = (
    IndexConfig("hnsw_m16_efc128_efs40", "hnsw", ef_search=40),
    IndexConfig("hnsw_m16_efc128_efs80", "hnsw", ef_search=80),
    IndexConfig("hnsw_m16_efc128_efs160", "hnsw", ef_search=160),
    IndexConfig("hnsw_m32_efc256_efs80", "hnsw", m=32, ef_construction=256),
    IndexConfig("ivfflat_l10_p1", "ivfflat", lists=10, probes=1),
    IndexConfig("ivfflat_l10_p5", "ivfflat", lists=10, probes=5),
)


def _unique_parent_results(results: Iterable[SearchResult]) -> list[SearchResult]:
    """Collapse multiple chunks from one source document before scoring."""
    seen: set[str] = set()
    unique: list[SearchResult] = []
    for result in results:
        parent = result.chunk.parent_doc_id
        if parent not in seen:
            seen.add(parent)
            unique.append(result)
    return unique


def _metrics(results: list[SearchResult], relevant: set[str], k: int) -> dict[str, float]:
    """Compute parent-document metrics with chunk de-duplication."""
    from apex_rag.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank

    unique = _unique_parent_results(results)
    return {
        "recall_at_k": recall_at_k(unique, relevant, k),
        "precision_at_k": precision_at_k(unique, relevant, k),
        "mrr": reciprocal_rank(unique, relevant),
        "ndcg_at_k": ndcg_at_k(unique, relevant, k),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    metrics = {name: statistics.fmean(float(row[name]) for row in rows)
               for name in ("recall_at_k", "precision_at_k", "mrr", "ndcg_at_k", "latency_ms")}
    latencies = sorted(float(row["latency_ms"]) for row in rows)
    metrics["p50_ms"] = latencies[len(latencies) // 2]
    metrics["p99_ms"] = latencies[min(len(latencies) - 1, int(len(latencies) * 0.99))]
    return metrics


def _run_configuration(
    database_url: str,
    embedding_model: str,
    reranker_model: str,
    table_name: str,
    chunks: list[Chunk],
    embeddings: Any,
    queries: list[dict[str, Any]],
    index: IndexConfig,
    k: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    """Load one isolated table, run all modes, and return detail/summary rows."""
    store = PostgresStore(database_url, dimension=int(embeddings.shape[1]), table_name=table_name)
    try:
        store.reset_table()
        store.upsert_chunks(chunks, embeddings)
        store.create_index(index.index_type, index.m, index.ef_construction,
                           lists=index.lists, drop_existing=True)
    finally:
        store.close()

    engine = PostgresRetrievalEngine(
        database_url, embedding_model, reranker_model,
        rrf_k=60, hnsw_ef_search=index.ef_search,
        table_name=table_name, index_type=index.index_type,
        ivfflat_probes=index.probes,
    )
    details: list[dict[str, Any]] = []
    try:
        for mode in ("dense", "hybrid", "enhanced"):
            mode_rows: list[dict[str, Any]] = []
            for query in queries:
                results, latency = engine.timed_search(
                    query["query"], mode, top_k=20, final_top_n=k
                )
                metrics = _metrics(results, set(query["relevant_doc_ids"]), k)
                row = {
                    "configuration": index.name,
                    "mode": mode,
                    "query_id": query["query_id"],
                    "result_doc_ids": ",".join(r.chunk.parent_doc_id for r in _unique_parent_results(results)),
                    "latency_ms": latency,
                    **metrics,
                }
                details.append(row)
                mode_rows.append(row)
    finally:
        engine.close()
    # Reconstruct summaries from query rows.
    summary_by_mode = {
        mode: _summary([r for r in details if r["mode"] == mode])
        for mode in ("dense", "hybrid", "enhanced")
    }
    return details, summary_by_mode


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["configuration", "chunking_strategy", "chunk_count", "mode", "query_id",
              "recall_at_k", "precision_at_k", "mrr", "ndcg_at_k", "latency_ms",
              "p50_ms", "p99_ms", "result_doc_ids"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_experiments(
    database_url: str,
    embedding_model: str,
    reranker_model: str,
    documents: list[Any],
    queries: list[dict[str, Any]],
    report_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Run chunking and index experiments and write CSV/Markdown evidence."""
    report_dir.mkdir(parents=True, exist_ok=True)
    embedder = create_embedder(embedding_model)
    chunk_sets = {
        "original": original_chunks(documents),
        "sliding_window": sliding_window_chunks(documents, window_size=40, overlap=10),
        "hierarchical": hierarchical_chunks(documents),
    }
    embeddings_by_strategy = {
        strategy: embedder.encode([f"{c.title}. {c.category}. {c.content}" for c in chunks])
        for strategy, chunks in chunk_sets.items()
    }
    chunk_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    chunk_summaries: list[dict[str, Any]] = []
    index_summaries: list[dict[str, Any]] = []

    for strategy, chunks in chunk_sets.items():
        table = f"experiment_{strategy}"
        details, summaries = _run_configuration(
            database_url, embedding_model, reranker_model, table, chunks,
            embeddings_by_strategy[strategy], queries, CHUNK_INDEX,
        )
        for row in details:
            row.update({"chunking_strategy": strategy, "chunk_count": len(chunks)})
        chunk_rows.extend(details)
        for mode, values in summaries.items():
            chunk_summaries.append({"configuration": CHUNK_INDEX.name, "chunking_strategy": strategy,
                                    "chunk_count": len(chunks), "mode": mode, **values})

    chunks = chunk_sets["original"]
    for index in INDEX_CONFIGS:
        table = "experiment_index"
        details, summaries = _run_configuration(
            database_url, embedding_model, reranker_model, table, chunks,
            embeddings_by_strategy["original"], queries, index,
        )
        for row in details:
            row.update({"chunking_strategy": "original", "chunk_count": len(chunks)})
        index_rows.extend(details)
        for mode, values in summaries.items():
            index_summaries.append({"configuration": index.name, "chunking_strategy": "original",
                                    "chunk_count": len(chunks), "mode": mode, **values})

    _write_csv(report_dir / "chunking_experiment.csv", chunk_rows)
    _write_csv(report_dir / "index_experiment.csv", index_rows)
    _write_markdown(report_dir / "experiments.md", chunk_summaries, index_summaries)
    return {"chunking": chunk_summaries, "index": index_summaries}


def _write_markdown(path: Path, chunk_rows: list[dict[str, Any]], index_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Chunking and Index Experiments", "",
        "本报告使用现有 150 条知识库文档和 30 条评估查询，所有指标按 `parent_doc_id` 去重后计算。",
        "实验表使用独立 PostgreSQL 表，不会覆盖浏览器演示使用的 `documents` 表。", "",
        "## 分块策略对比", "",
        "| Strategy | Chunks | Mode | Recall@5 | Precision@5 | MRR | nDCG@5 | Mean ms | P50 ms | P99 ms |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in chunk_rows:
        lines.append(f"| {row['chunking_strategy']} | {row['chunk_count']} | {row['mode']} | {row['recall_at_k']:.4f} | {row['precision_at_k']:.4f} | {row['mrr']:.4f} | {row['ndcg_at_k']:.4f} | {row['latency_ms']:.3f} | {row['p50_ms']:.3f} | {row['p99_ms']:.3f} |")
    lines.extend(["", "## 向量索引与参数对比", "",
                  "| Configuration | Mode | Recall@5 | Precision@5 | MRR | nDCG@5 | Mean ms | P50 ms | P99 ms |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in index_rows:
        lines.append(f"| {row['configuration']} | {row['mode']} | {row['recall_at_k']:.4f} | {row['precision_at_k']:.4f} | {row['mrr']:.4f} | {row['ndcg_at_k']:.4f} | {row['latency_ms']:.3f} | {row['p50_ms']:.3f} | {row['p99_ms']:.3f} |")
    lines.extend(["", "## 解释与边界", "",
                  "当前数据集只有 150 条文档，近似索引可能被 PostgreSQL 选择性地退化为顺序扫描，因此索引之间的差异只代表本演示规模。生产环境应在目标数据量和硬件上重新压测。",
                  "最终演示配置应优先选择 Recall@5/MRR 较高且 P99 可接受的组合；报告不把当前延迟外推到千万级数据。", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
