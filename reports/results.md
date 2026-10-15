# 检索评估结果

本报告由给定的 150 篇文档和 30 条查询生成。
后端：`postgres`。本次使用 PostgreSQL 16 + pgvector、配置的 `HNSW` 索引，以及 HNSW `ef_search=40`。

| 模式 | Recall@5 | Precision@5 | MRR | nDCG@5 | 平均毫秒 | P50 毫秒 | P99 毫秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 0.9444 | 0.2533 | 0.9733 | 0.9403 | 10.190 | 10.129 | 21.272 |
| hybrid | 0.9444 | 0.2533 | 0.9733 | 0.9403 | 10.854 | 11.025 | 13.107 |
| enhanced | 0.9667 | 0.2667 | 0.9611 | 0.9348 | 93.711 | 94.699 | 108.530 |

## 解读

Enhanced 在向量检索基线上增加了 RRF 融合和配置的 Cross-Encoder 重排序。本报告中的耗时来自真实的 PostgreSQL + pgvector。逐题结果见 `metrics.csv`，质量与耗时对比图见 `figures/benchmark_summary.png`。
