"""Run baseline and enhanced retrieval experiments."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from apex_rag.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from apex_rag.models import SearchResult
from apex_rag.retrieval.engine import RetrievalEngine


def run_benchmark(engine: RetrievalEngine, queries: list[dict[str, Any]], report_dir: Path, k: int = 5) -> dict[str, dict[str, float]]:
    """Run all modes and write per-query CSV plus summary metrics."""
    report_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    summary: dict[str, dict[str, float]] = {}
    for mode in ("dense", "hybrid", "enhanced"):
        mode_rows: list[dict[str, float]] = []
        for query in queries:
            relevant = set(query["relevant_doc_ids"])
            results, latency = engine.timed_search(query["query"], mode, top_k=20, final_top_n=k)
            metrics = {
                "recall_at_k": recall_at_k(results, relevant, k),
                "precision_at_k": precision_at_k(results, relevant, k),
                "mrr": reciprocal_rank(results, relevant),
                "ndcg_at_k": ndcg_at_k(results, relevant, k),
                "latency_ms": latency,
            }
            mode_rows.append(metrics)
            rows.append({"query_id": query["query_id"], "mode": mode, **metrics,
                         "result_doc_ids": ",".join(r.chunk.parent_doc_id for r in results)})
        summary[mode] = {
            metric: statistics.fmean(row[metric] for row in mode_rows)
            for metric in ("recall_at_k", "precision_at_k", "mrr", "ndcg_at_k", "latency_ms")
        }
        latencies = sorted(row["latency_ms"] for row in mode_rows)
        summary[mode]["p50_ms"] = latencies[len(latencies) // 2]
        summary[mode]["p99_ms"] = latencies[min(len(latencies) - 1, int(len(latencies) * 0.99))]
    with (report_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_summary_figure(summary, report_dir)
    return summary


def _write_summary_figure(summary: dict[str, dict[str, float]], report_dir: Path) -> None:
    """Write a compact quality and latency comparison figure."""
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    modes = list(summary)
    positions = list(range(len(modes)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar([i - 0.15 for i in positions], [summary[m]["recall_at_k"] for m in modes], width=0.3, label="Recall@5")
    axes[0].bar([i + 0.15 for i in positions], [summary[m]["mrr"] for m in modes], width=0.3, label="MRR")
    axes[0].set_xticks(positions, modes)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Retrieval quality")
    axes[0].legend()
    axes[1].bar([i - 0.15 for i in positions], [summary[m]["p50_ms"] for m in modes], width=0.3, label="P50 ms")
    axes[1].bar([i + 0.15 for i in positions], [summary[m]["p99_ms"] for m in modes], width=0.3, label="P99 ms")
    axes[1].set_xticks(positions, modes)
    axes[1].set_ylabel("milliseconds")
    axes[1].set_title("Retrieval latency")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "benchmark_summary.png", dpi=160)
    plt.close(fig)
