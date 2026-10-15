# 企业级向量检索与 RAG 系统开发文档

## 1. 文档目的

本文档用于在正式编码前确认项目范围、技术路线、验收标准和实施顺序。项目目标是基于现有信用卡知识库和标准评估查询，构建一个可运行、可复现、可量化评估的企业级检索服务。

本文档对应当前可运行实现。实现状态与实测结果以 `docs/ACCEPTANCE_MATRIX.md`、`reports/results.md` 和 `reports/experiments.md` 为准。

## 2. 项目背景与目标

### 2.1 业务场景

Apex Card Services 的客服坐席需要查询信用卡账单分期、APR、盗刷争议、退款、积分权益和紧急换卡等规则。系统需要为坐席提供低延迟、可解释的相关知识检索结果，作为后续 RAG 回答生成的上下文来源。

### 2.2 当前数据

- `data/kb_documents.json`：150 条知识库文档。
- `data/eval_queries.json`：30 条评估查询，包含相关文档 ID 和标准答案。
- 文档字段：`doc_id`、`title`、`category`、`card_tier`、`effective_year`、`content`。
- 评估查询字段：`query_id`、`query`、`relevant_doc_ids`、`gold_response`。

### 2.3 核心目标

1. 将知识库导入 PostgreSQL + pgvector。
2. 实现 Dense、Sparse 和 Hybrid 三种检索模式。
3. 实现 RRF 结果融合和第二阶段重排序。
4. 对比 Baseline 与 Enhanced 检索质量和延迟。
5. 对比原始语义文档块与固定窗口分块策略。
6. 提供可复现的一键运行流程和结果报告。
7. 编写面向生产扩展、缓存、数据治理和 PII 防护的架构路线图。
8. 在同一份 150 条文档和 30 条查询上完成原始块、40/10 滑动窗口、层级分块，以及 HNSW/IVFFlat 参数实验。

## 3. 范围边界

### 3.1 本阶段必须完成

- pgvector 数据导入和 HNSW/IVFFlat 索引。
- Embedding 生成和元数据保存。
- PostgreSQL `tsvector` 稀疏检索。
- Dense + Sparse 混合检索。
- RRF 融合。
- Cross-Encoder 或 FlashRank 重排序。
- Baseline/Enhanced 评估。
- Recall@K、Precision@K、MRR、nDCG@K、P50/P99 延迟。
- 分块策略对比。
- README、结果报告、路线图和一键运行脚本。

### 3.2 暂不作为核心目标

- 完整的生产级 LLM 对话生成服务。
- 训练领域专属 Embedding 或 Reranker 模型。
- 大规模分布式向量数据库集群。
- 复杂的 Agent、工作流和前端界面。

LLM-as-a-Judge、GraphRAG、Apache Iceberg 和完整 FastAPI 服务作为核心流程稳定后的加分项。

## 4. 技术选型

### 4.1 核心技术栈

| 层次 | 选择 | 原因 |
|---|---|---|
| 数据库 | PostgreSQL + pgvector | 满足评估要求，可同时保存业务元数据、向量和全文检索字段 |
| Python 数据库访问 | psycopg + pgvector-python | 依赖简单、可直接控制 SQL 和索引参数 |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` | 本地可运行、384 维、适合当前英文数据集 |
| 稀疏检索 | PostgreSQL `tsvector` + GIN | 不额外引入搜索服务，便于复现 |
| 融合 | 自定义 RRF | 逻辑透明，便于展示和调参 |
| 重排序 | `cross-encoder/ms-marco-MiniLM-L6-v2` | 适合 Top-20 到 Top-5 的二阶段排序 |
| 轻量替代 | FlashRank | CPU 资源有限时降低重排序延迟 |
| 配置 | pydantic-settings + `.env` | 集中管理数据库、模型和检索参数 |
| API | FastAPI | 加分项，便于暴露 `/api/v1/search` |
| 容器 | Docker Compose | 统一 PostgreSQL/pgvector 运行环境 |
| 评估 | 自定义检索指标，Ragas 为可选项 | 自定义指标直接对应评估矩阵，Ragas 用于 LLM 质量扩展 |

Docker 不是硬性依赖。无 Docker 时使用本地后端完成开发、功能验证和离线评估；正式 pgvector 评分可以连接原生安装或远程托管的 PostgreSQL + pgvector。

### 4.2 外部开源项目的使用方式

- [pgvector](https://github.com/pgvector/pgvector)：直接使用官方扩展。
- [pgvector-python](https://github.com/pgvector/pgvector-python)：用于 Python 类型转换和数据库写入。
- [Sentence Transformers](https://github.com/huggingface/sentence-transformers)：用于 Embedding 和 Cross-Encoder。
- [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank)：作为可切换的轻量重排序器。
- [RAGLite](https://github.com/superlinear-ai/raglite)：参考其 PostgreSQL 混合检索、RRF 和分块设计，不默认把核心流程封装成黑盒。
- [LlamaIndex PGVectorStore](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/vector_stores/llama-index-vector-stores-postgres)：作为可选对照后端。
- [Ragas](https://github.com/vibrantlabsai/ragas)：用于加分项中的 Context Relevance 和 Faithfulness 评估。

## 5. 系统架构

```text
JSON 数据
   |
   v
校验与分块
   |
   +--> Embedding 模型 --> pgvector embedding
   |
   +--> 文本规范化 --> PostgreSQL tsvector
   |
   +--> 元数据 ----------------------+
                                      v
                              PostgreSQL documents
                                      |
                    +-----------------+-----------------+
                    |                                   |
              Dense Search                         Sparse Search
                    |                                   |
                    +------------ RRF -----------------+
                                      |
                              Candidate Top-K
                                      |
                               Re-ranker
                                      |
                                Final Top-N
                                      |
                         API / Benchmark / Report
```

## 6. 数据库设计

建议的核心表：

```sql
CREATE TABLE documents (
    chunk_id TEXT PRIMARY KEY,
    parent_doc_id TEXT NOT NULL,
    title TEXT NOT NULL,
    category TEXT,
    card_tier TEXT,
    effective_year INT,
    content TEXT NOT NULL,
    embedding VECTOR(384),
    search_vector TSVECTOR,
    chunking_strategy TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

建议索引：

```sql
CREATE INDEX documents_embedding_hnsw_idx
ON documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 128);

CREATE INDEX documents_search_vector_idx
ON documents USING gin (search_vector);
```

索引参数必须通过实验配置，而不是写死为唯一答案。至少比较不同 `ef_search` 值，并记录 Recall 与延迟的变化。

## 7. 检索设计

### 7.1 Baseline

Baseline 仅使用 pgvector Dense Search：

```sql
SELECT chunk_id, parent_doc_id, title, content,
       1 - (embedding <=> :query_embedding) AS score
FROM documents
WHERE chunking_strategy = :strategy
ORDER BY embedding <=> :query_embedding
LIMIT :top_k;
```

### 7.2 Sparse Search

使用 PostgreSQL 全文检索：

```sql
SELECT chunk_id, parent_doc_id, title, content,
       ts_rank_cd(search_vector, websearch_to_tsquery('english', :query)) AS score
FROM documents
WHERE search_vector @@ websearch_to_tsquery('english', :query)
ORDER BY score DESC
LIMIT :top_k;
```

### 7.3 Hybrid Search

Dense 和 Sparse 各取 Top-20，通过 RRF 合并：

```text
rrf_score(document) = sum(1 / (rrf_k + rank_i))
```

默认参数：

- Dense Top-K：20。
- Sparse Top-K：20。
- `rrf_k`：60。
- 重排序候选集：20。
- 最终返回 Top-N：5。

这些参数需要在报告中说明，并保留配置入口。

### 7.4 重排序

对混合检索后的候选文档形成 `(query, passage)` 对，使用 Cross-Encoder 或 FlashRank 计算相关性，再返回 Top-5。输出中保留 Dense 分数、Sparse 分数、RRF 分数和 reranker 分数，方便调试和结果解释。

### 7.5 元数据过滤

第一版支持可选过滤条件：

- `category`
- `card_tier`
- `effective_year`

过滤条件必须在向量检索和全文检索中保持一致，避免两个检索分支使用不同候选集合。

## 8. 分块策略实验

### 策略 A：原始知识库块

直接使用现有的 150 条文档作为检索单元，保留完整标题和元数据。由于当前文档长度约 27–78 个英文词，该策略预计是强基线。

### 策略 B：固定滑动窗口

- 窗口大小：40 个词。
- 重叠：10 个词。
- 每个子块保存 `parent_doc_id`。

### 策略 C：语义/层级分块

按标题、产品、类别、规则详情等字段组织内容，并在每个块中保留上下文标题和关键元数据。

评估时按 `parent_doc_id` 去重，避免同一原始文档的多个子块重复计数。

## 9. 评估方案

### 9.1 对比系统

| 系统 | Dense | Sparse | RRF | Reranker |
|---|---:|---:|---:|---:|
| Baseline | 是 | 否 | 否 | 否 |
| Hybrid | 是 | 是 | 是 | 否 |
| Enhanced | 是 | 是 | 是 | 是 |

### 9.2 指标

- `Recall@K`
- `Precision@K`
- `MRR`
- `nDCG@K`
- P50 检索延迟
- P99 检索延迟

评估必须使用 `relevant_doc_ids` 作为唯一检索标签来源。`gold_response` 可以用于后续回答生成或 Ragas 评估，不作为核心检索指标的替代品。

### 9.3 实验记录

每次实验至少记录：

- Embedding 模型及维度。
- 分块策略。
- 索引类型和参数。
- Dense/Sparse/RRF/Reranker 参数。
- 每条查询的结果和延迟。
- 汇总指标。
- 运行时间和机器环境。

最终报告至少包含 Baseline、Hybrid、Enhanced 三组结果和一张延迟分布图。

## 10. 项目目录

```text
.
├── data/
├── docs/
│   ├── DEVELOPMENT_PLAN.md
│   └── roadmap.md
├── src/
│   ├── config.py
│   ├── db.py
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── chunking.py
│   │   ├── embedder.py
│   │   └── pipeline.py
│   ├── retrieval/
│   │   ├── dense.py
│   │   ├── sparse.py
│   │   ├── hybrid.py
│   │   └── reranker.py
│   └── evaluation/
│       ├── metrics.py
│       ├── benchmark.py
│       └── report.py
├── reports/
│   ├── results.md
│   ├── metrics.csv
│   └── figures/
├── tests/
├── .env.example
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── README.md
```

## 11. 命令设计

目标是提供以下命令：

```bash
make up          # 启动 PostgreSQL + pgvector
make install     # 安装 Python 依赖
make ingest      # 导入文档、生成 Embedding、构建索引
make evaluate    # 运行全部评估实验
make report      # 生成 Markdown、CSV 和图表
make test        # 运行单元测试
make all         # 一键执行主要流程
```

当前实验命令为 `python -m apex_rag.cli experiments`（或 `make experiments`），会生成分块策略和向量索引参数的独立对比报告。`make all` 已包含该命令。

如果实现 FastAPI 加分项，再提供：

```bash
make api
```

接口初版：

```text
POST /api/v1/search
```

请求至少支持：`query`、`top_k`、`mode`、`category`、`card_tier`、`effective_year`。

响应至少包含：`doc_id`、`title`、`content`、`scores`、`rank` 和 `metadata`。

## 12. 实施阶段

### 阶段 0：方案确认

- 确认 Embedding 和 reranker 模型。
- 确认是否允许 Docker。
- 确认是否需要优先完成 FastAPI。
- 确认默认 Top-K、Top-N 和评估 K 值。

### 阶段 1：运行环境与数据库

- 初始化 Python 项目。
- 编写 Docker Compose。
- 启动 PostgreSQL + pgvector。
- 建表和执行扩展检查。

### 阶段 2：数据导入与索引

- 数据校验。
- 分块策略实现。
- Embedding 生成。
- `tsvector` 生成。
- HNSW/IVFFlat 索引构建。

### 阶段 3：检索流水线

- Dense Search。
- Sparse Search。
- RRF 融合。
- Reranker。
- 元数据过滤。

### 阶段 4：评估和调优

- Baseline/Hybrid/Enhanced 对比。
- HNSW 参数实验。
- 分块策略实验。
- 延迟采集。
- 生成表格和图表。

### 阶段 5：交付文档

- README。
- `reports/results.md`。
- `docs/roadmap.md`。
- 测试和一键运行验证。

### 阶段 6：可选加分

- FastAPI + Swagger。
- Ragas。
- GraphRAG。
- Apache Iceberg PoC。

## 13. 风险与处理方式

### 模型下载或网络不可用

支持本地模型目录配置，并在 README 中说明模型缓存方式。若 Cross-Encoder 下载失败，可以先运行无重排序版本，再切换 FlashRank。

### 当前数据量太小，延迟不具代表性

报告中区分“当前 150 条数据的实测延迟”和“生产规模的扩展建议”，不把小数据集结果直接外推到 1,000 万条记录。

### 关键词检索对数字和产品名敏感

保留标题、卡片层级、类别和年份等字段进入 `search_vector`，并通过 Dense + Sparse 融合提高精确匹配能力。

### 分块后同一文档重复命中

使用 `parent_doc_id` 聚合和去重，评估时按原始文档 ID 计算指标。

### 评估标签较少

严格使用提供的 30 条查询和相关文档 ID，不自行添加未标注的“相关”结果；在报告中说明标签规模和局限性。

## 14. 已落地决策

以下决策已用于当前演示实现：

1. 使用本地 `all-MiniLM-L6-v2`（384 维）和 `ms-marco-MiniLM-L6-v2` Cross-Encoder。
2. Docker Compose 只负责提供 PostgreSQL + pgvector，不是 Python 运行时硬依赖。
3. FastAPI、Swagger 和浏览器检索界面已经纳入演示交付。
4. 核心评估统一使用 `K=5`，并保存逐查询结果。
5. 当前交付检索上下文，不把未实现的 LLM 生成、Ragas、GraphRAG 或 Iceberg 宣称为完成项。

## 15. 初步建议

在没有额外约束的情况下，建议采用：

- PostgreSQL + pgvector + Docker Compose。
- 本地 `all-MiniLM-L6-v2` Embedding。
- PostgreSQL `tsvector` 做 Sparse Search。
- 自定义 RRF 做融合。
- `cross-encoder/ms-marco-MiniLM-L6-v2` 做重排序。
- 先实现检索 API 和评估脚本，再决定是否加入完整 LLM 回答生成。

这个组合能够直接覆盖评估中的向量存储、混合检索、重排序、定量评估、工程质量和可复现性要求，同时保留后续替换模型和扩展服务的空间。
