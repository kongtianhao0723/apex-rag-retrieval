"""Command-line entry points for ingestion and evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from time import perf_counter
from pathlib import Path

from apex_rag.config import settings
from apex_rag.answering import Judgment, build_answerer, extractive_fallback, score_answer
from apex_rag.embeddings import create_embedder
from apex_rag.evaluation.benchmark import run_benchmark
from apex_rag.evaluation.experiments import run_experiments
from apex_rag.ingestion.chunking import original_chunks
from apex_rag.ingestion.loader import load_documents, load_queries
from apex_rag.retrieval.engine import RetrievalEngine


def _build_engine() -> RetrievalEngine:
    documents = load_documents(settings.data_dir / "kb_documents.json")
    if settings.backend == "postgres":
        from apex_rag.retrieval.postgres import PostgresRetrievalEngine

        return PostgresRetrievalEngine(
            settings.database_url,
            settings.embedding_model,
            settings.reranker_model,
            settings.rrf_k,
            settings.hnsw_ef_search,
        )
    return RetrievalEngine(original_chunks(documents), rrf_k=settings.rrf_k)


def ingest() -> None:
    """Validate input data and write reproducible local index metadata."""
    documents = load_documents(settings.data_dir / "kb_documents.json")
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "document_count": len(documents),
        "chunking_strategy": "original",
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
        "index_type": settings.index_type,
        "hnsw_m": settings.hnsw_m,
        "hnsw_ef_construction": settings.hnsw_ef_construction,
        "hnsw_ef_search": settings.hnsw_ef_search,
        "ivfflat_lists": settings.ivfflat_lists,
        "ivfflat_probes": settings.ivfflat_probes,
        "recommended_configuration": "original + HNSW(m=16, ef_construction=128, ef_search=40)",
        "dense_top_k": settings.dense_top_k,
        "sparse_top_k": settings.sparse_top_k,
        "final_top_n": settings.final_top_n,
        "rrf_k": settings.rrf_k,
        "backend": settings.backend,
    }
    if settings.backend == "postgres":
        from apex_rag.db import PostgresStore

        embedder = create_embedder(settings.embedding_model)
        chunks = original_chunks(documents)
        texts = [f"{chunk.title}. {chunk.category}. {chunk.content}" for chunk in chunks]
        embeddings = embedder.encode(texts)
        store = PostgresStore(settings.database_url, dimension=embedder.dimension)
        try:
            count = store.upsert_chunks(chunks, embeddings)
            store.create_index(settings.index_type, settings.hnsw_m, settings.hnsw_ef_construction)
        finally:
            store.close()
        config["embedding_dimension"] = embedder.dimension
        print(f"Imported {count} chunks into PostgreSQL with {settings.index_type} index.")
    else:
        print(f"Validated {len(documents)} documents (local fallback; set BACKEND=postgres for pgvector).")
    (settings.data_dir / "index_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def evaluate() -> None:
    """Run the benchmark and write a concise Markdown summary."""
    engine = _build_engine()
    queries = load_queries(settings.data_dir / "eval_queries.json")
    try:
        summary = run_benchmark(engine, queries, settings.report_dir, k=settings.final_top_n)
    finally:
        close = getattr(engine, "close", None)
        if close is not None:
            close()
    report = [
        "# 检索评估结果",
        "",
        "本报告由给定的 150 篇文档和 30 条查询生成。",
        (f"后端：`postgres`。本次使用 PostgreSQL 16 + pgvector、配置的 "
         f"`{settings.index_type.upper()}` 索引，以及 HNSW `ef_search={settings.hnsw_ef_search}`。"
         if settings.backend == "postgres" else
         "后端：`local`。本次使用可复现的本地离线检索。"),
        "",
        "| 模式 | Recall@5 | Precision@5 | MRR | nDCG@5 | 平均毫秒 | P50 毫秒 | P99 毫秒 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, values in summary.items():
        report.append(f"| {mode} | {values['recall_at_k']:.4f} | {values['precision_at_k']:.4f} | {values['mrr']:.4f} | {values['ndcg_at_k']:.4f} | {values['latency_ms']:.3f} | {values['p50_ms']:.3f} | {values['p99_ms']:.3f} |")
    interpretation = (
        "Enhanced 在向量检索基线上增加了 RRF 融合和配置的 Cross-Encoder 重排序。"
        "本报告中的耗时来自真实的 PostgreSQL + pgvector。"
        "逐题结果见 `metrics.csv`，质量与耗时对比图见 `figures/benchmark_summary.png`。"
        if settings.backend == "postgres" else
        "Enhanced 在向量检索基线上增加了 RRF 融合和可复现的本地重排序。"
        "安装 `.[models]` 并将 `BACKEND` 设为 `postgres` 后，可使用配置的 Sentence Transformers 模型和 PostgreSQL 耗时。"
        "逐题结果见 `metrics.csv`，质量与耗时对比图见 `figures/benchmark_summary.png`。"
    )
    report.extend(["", "## 解读", "", interpretation])
    (settings.report_dir / "results.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


def experiments() -> None:
    """Run the controlled chunking and vector-index comparison suite."""
    if settings.backend != "postgres":
        raise RuntimeError("The experiment suite requires BACKEND=postgres and pgvector")
    documents = load_documents(settings.data_dir / "kb_documents.json")
    queries = load_queries(settings.data_dir / "eval_queries.json")
    summaries = run_experiments(
        settings.database_url, settings.embedding_model, settings.reranker_model,
        documents, queries, settings.report_dir,
    )
    print(f"Chunking configurations: {len(summaries['chunking'])}")
    print(f"Index configurations: {len(summaries['index'])}")
    print(f"Reports written to {settings.report_dir / 'experiments.md'}")


def evaluate_answers() -> None:
    """Generate DeepSeek answers for the 30 labelled queries and score them."""
    if settings.backend != "postgres":
        raise RuntimeError("Answer evaluation requires BACKEND=postgres")
    answerer = build_answerer(settings.deepseek_api_key, settings.deepseek_model, settings.deepseek_base_url)
    if answerer is None:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")
    queries = load_queries(settings.data_dir / "eval_queries.json")
    engine = _build_engine()
    rows: list[dict[str, object]] = []
    try:
        for query in queries:
            results, retrieval_latency = engine.timed_search(
                query["query"], "enhanced", top_k=20, final_top_n=settings.final_top_n
            )
            started = perf_counter()
            try:
                answer = answerer.answer(query["query"], results)
                error = ""
                source = "deepseek"
            except RuntimeError as exc:
                answer = extractive_fallback(results)
                error = str(exc)
                source = "retrieval_fallback"
            answer_latency = (perf_counter() - started) * 1000
            metrics = score_answer(answer, query["gold_response"]) if answer else None
            judge_started = perf_counter()
            try:
                judgment = answerer.judge(query["query"], results, answer)
            except RuntimeError as exc:
                judgment = Judgment(None, None, error=str(exc))
            judge_latency = (perf_counter() - judge_started) * 1000
            rows.append({
                "query_id": query["query_id"], "query": query["query"],
                "gold_response": query["gold_response"], "answer": answer,
                "source": source, "error": error, "retrieval_latency_ms": retrieval_latency,
                "answer_latency_ms": answer_latency, "judge_latency_ms": judge_latency,
                **(metrics.as_dict() if metrics else {"token_precision": 0.0, "token_recall": 0.0,
                    "token_f1": 0.0, "numeric_coverage": 0.0, "score": 0.0}),
                **judgment.as_dict(),
            })
    finally:
        close = getattr(engine, "close", None)
        if close is not None:
            close()
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = settings.report_dir / "answer_evaluation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    successful = [row for row in rows if row["source"] == "deepseek"]
    fallback = [row for row in rows if row["source"] == "retrieval_fallback"]
    available = [row for row in rows if row["answer"]]

    def _mean(selected: list[dict[str, object]], key: str) -> float | None:
        values = [float(row[key]) for row in selected if row.get(key) not in ("", None)]
        return statistics.fmean(values) if values else None

    def _cell(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    mean_score = _mean(available, "score")
    mean_f1 = _mean(available, "token_f1")
    mean_numeric = _mean(available, "numeric_coverage")
    report = [
        "# DeepSeek Answer Evaluation", "",
        f"Model: `{settings.deepseek_model}`; queries: {len(rows)}; DeepSeek answers: {len(successful)}; "
        f"retrieval fallback: {len(fallback)}; answers available: {len(available)}.",
        "Token F1、Numeric coverage 和 Composite score 仍是对照 `gold_response` 的词面回归分，公式为 "
        "0.7 × Token F1 + 0.3 × Numeric coverage。",
        "Context Relevance 与 Faithfulness 由同一 DeepSeek 模型做 LLM 评判：相关性 = 被标为有助于回答的检索篇数 / 检索篇数；"
        "忠实度 = 被上下文支持的原子陈述数 / 陈述总数。评判失败时该题分数留空，不中断其余题目。", "",
        "| Metric | All answers | DeepSeek only | Fallback only |",
        "|---|---:|---:|---:|",
        f"| Token F1 | {_cell(mean_f1)} | {_cell(_mean(successful, 'token_f1'))} | {_cell(_mean(fallback, 'token_f1'))} |",
        f"| Numeric coverage | {_cell(mean_numeric)} | {_cell(_mean(successful, 'numeric_coverage'))} | {_cell(_mean(fallback, 'numeric_coverage'))} |",
        f"| Composite score | {_cell(mean_score)} | {_cell(_mean(successful, 'score'))} | {_cell(_mean(fallback, 'score'))} |",
        f"| Context Relevance | {_cell(_mean(available, 'context_relevance'))} | {_cell(_mean(successful, 'context_relevance'))} | {_cell(_mean(fallback, 'context_relevance'))} |",
        f"| Faithfulness | {_cell(_mean(available, 'faithfulness'))} | {_cell(_mean(successful, 'faithfulness'))} | {_cell(_mean(fallback, 'faithfulness'))} |",
        "",
        "逐题答案、标准答案、分块判定、陈述判定、错误信息和耗时见 `reports/answer_evaluation.csv`。回答失败时不影响原有检索结果。",
    ]
    (settings.report_dir / "answer_evaluation.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("ingest", "evaluate", "experiments", "evaluate-answers"))
    args = parser.parse_args()
    {"ingest": ingest, "evaluate": evaluate, "experiments": experiments,
     "evaluate-answers": evaluate_answers}[args.command]()


if __name__ == "__main__":
    main()
