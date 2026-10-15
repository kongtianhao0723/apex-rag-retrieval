# 技术评估验收标准矩阵

本文记录当前仓库的真实实现状态与可复核证据。验收数据严格限定为 `data/kb_documents.json` 的 150 条文档和 `data/eval_queries.json` 的 30 条查询；没有扩充或改写数据。

| 评分项 | 权重 | 状态 | 实现与验收证据 | 当前结论 |
|---|---:|---|---|---|
| 向量存储与导入 | 25 | 已完成 | PostgreSQL 16 + pgvector；`python -m apex_rag.cli ingest`；`sql/init.sql`；`data/index_config.json` | 150 条原始知识文档生成 150 个 384 维向量，元数据和全文检索字段入库；HNSW/IVFFlat 均有实测 |
| 混合检索与重排序 | 25 | 已完成 | `src/apex_rag/retrieval/postgres.py`；网页 `http://127.0.0.1:18000/`；`/docs` | Dense、PostgreSQL FTS、RRF、Cross-Encoder、三种元数据过滤均可运行；Enhanced 对当前标签集 Recall@5=0.9667 |
| 定量评估与呈现 | 20 | 已完成 | `python -m apex_rag.cli evaluate`；`reports/results.md`、`reports/metrics.csv`、`reports/figures/benchmark_summary.png` | 三种检索模式均报告 Recall@5、Precision@5、MRR、nDCG@5、Mean/P50/P99；30 条逐题结果可追溯 |
| 分块与索引实验 | 核心要求，纳入评估/工程项 | 已完成 | `python -m apex_rag.cli experiments`；`reports/experiments.md`、`reports/chunking_experiment.csv`、`reports/index_experiment.csv` | 三种分块、六种索引参数组合；完整报告每组质量和延迟；实验表隔离于正式演示表 |
| 代码质量与工程 | 20 | 已完成 | `pytest -q`；`python -m compileall -q src`；`Makefile` | 配置集中管理、模块化实现、CLI 一键链路和回归测试可运行 |
| 战略路线图 | 10 | 已完成 | `docs/roadmap.md` | 覆盖千万级扩展、缓存、时效治理、PII、访问控制、可观测性 |
| 加分项 | +20 | 部分完成 | `docker compose up --build -d` 同时启动 PostgreSQL 与 FastAPI（`/api/v1/search`、Swagger、检索页面）；DeepSeek 回答层以及 Context Relevance / Faithfulness 自动评判已实现；Ragas 库、GraphRAG、Iceberg 未接入 | 两项 LLM 指标由 DeepSeek 按检索篇和原子陈述逐条判定后本地计分。回答失败不影响检索 |

## 关键实验结论

- 当前完整 Enhanced 演示推荐 `original + HNSW(m=16, ef_construction=128, ef_search=40)`。在 HNSW `ef_search=40/80/160` 以及 m/ef_construction 扩大到 `32/256` 时，当前数据的检索质量一致；选择 40 是因为没有观察到质量损失且搜索负担更低。
- 40/10 滑窗策略的 Dense Recall@5 为 0.9778、MRR 为 1.0000，是纯 Dense 检索下的最佳分块结果；但它的 Enhanced 指标不如原始块，因此综合链路仍推荐原始块。
- IVFFlat `lists=10, probes=1` 在当前集合造成明显召回下降；`probes=5` 追平 HNSW，当前规模无明显速度收益。
- Enhanced 的重排会提高部分查询的召回，但 nDCG/MRR 未全面超过 Dense/Hybrid。结论应以具体指标为准，不能宣称每项指标都提升。

## 验收边界

当前证据可以证明本机演示数据上的功能完整、流程可复现和指标可测。150 条记录太少，索引参数差异可能被顺序扫描掩盖；模型耗时受 CPU、缓存及运行时状态影响。不能据此承诺千万级吞吐、延迟或线上质量，生产部署前需要使用目标硬件、真实规模和并发压测。

`gold_response` 保留在评估数据中，核心验收仍是检索质量。DeepSeek 回答层是检索后的可选后处理。Context Relevance 与 Faithfulness 由同一模型自动评判，定义对齐常见 RAG 评估，但没有接入 Ragas 库。GraphRAG 和 Iceberg 仍未实现。

当前已接入生成式回答层：API 从 `.env` 读取 DeepSeek 密钥，使用 `deepseek-flash` 根据 Top-5 检索上下文生成回答。请求关闭思考模式，避免思考过程占满生成额度后正文为空。`python -m apex_rag.cli evaluate-answers` 会同时记录词面回归分，以及上下文相关性和回答忠实度。最近一次 30 题全部由 DeepSeek 直接回答，没有检索原文回退，也没有评判留空；词面综合分 0.7147，Context Relevance 均值 0.3067，Faithfulness 均值 1.0000。

## 一键命令

```powershell
docker compose up --build -d
python -m apex_rag.cli evaluate
python -m apex_rag.cli experiments
pytest -q
```

`docker compose up --build -d` 会启动 PostgreSQL 和 API，并在容器内完成导入。评估和实验仍在本机 Python 中运行，结果写入 `reports/`。

`make all` 会依次运行导入、正式评估和全套分块/索引实验；实验运行时间明显长于单次评估。
