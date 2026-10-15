# Apex Card Services RAG Retrieval

这是信用卡客服 Co-Pilot 的向量检索评估项目。当前实现包含可离线运行的 Dense/Sparse/Hybrid/Enhanced 检索基线，并提供 PostgreSQL + pgvector 的真实导入适配器。

## 快速开始

```bash
python -m pip install -e ".[dev]"
python -m apex_rag.cli ingest
python -m apex_rag.cli evaluate
python -m apex_rag.cli experiments
```

结果写入 `reports/results.md` 和 `reports/metrics.csv`，索引参数写入 `data/index_config.json`。`experiments` 会额外生成 `reports/experiments.md`、`reports/chunking_experiment.csv` 和 `reports/index_experiment.csv`，覆盖三种分块策略与 HNSW/IVFFlat 参数对比。

## DeepSeek 回答层

检索完成后可以追加 DeepSeek 回答，不改变原有 Dense、Hybrid、Enhanced 检索结果。配置写在本机 `.env`（`.env` 已被 `.gitignore` 忽略）：

```ini
DEEPSEEK_API_KEY=你的密钥
DEEPSEEK_MODEL=deepseek-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com
ENABLE_ANSWERING=true
```

启动 API 后，浏览器页面会在检索结果上方显示回答；接口 `/api/v1/search` 会额外返回 `answer`、`answer_model`、`answer_latency_ms`。若 DeepSeek 暂时不可用，响应仍保留完整检索结果，并返回 `answer_error`。

使用 30 条标准查询评估回答：

```powershell
python -m apex_rag.cli evaluate-answers
```

报告写入 `reports/answer_evaluation.md` 和 `reports/answer_evaluation.csv`。Token F1 和数字覆盖率仍对照标准答案。同一命令还会用 DeepSeek 自动评判 Context Relevance（相关检索篇数 / 检索篇数）和 Faithfulness（被上下文支持的陈述数 / 陈述总数）。

如果把 `BACKEND` 设为 `local`，这条路径不依赖 Docker 或 PostgreSQL，可完成离线算法验证。当前演示验收使用 PostgreSQL + pgvector；PowerShell 本地回退方式如下：

```powershell
$env:BACKEND="local"
python -m apex_rag.cli ingest
python -m apex_rag.cli evaluate
```

正式演示推荐配置记录在 `data/index_config.json`：原始文档块、HNSW `m=16`、`ef_construction=128`、`ef_search=40`。完整实验报告见 `reports/experiments.md`。

## Docker 部署

从 GitHub 克隆后，一条命令同时启动 PostgreSQL + pgvector 和 FastAPI。容器内会等待数据库就绪、导入 150 篇文档、创建 HNSW 索引，然后监听 18000 端口。没有 `.env` 也可以检索。

```powershell
git clone https://github.com/kongtianhao0723/apex-rag-retrieval.git
cd apex-rag-retrieval
docker compose up --build -d
```

打开 `http://127.0.0.1:18000/` 使用检索页面，打开 `http://127.0.0.1:18000/docs` 查看 Swagger。检索接口是 `POST /api/v1/search`。

需要 DeepSeek 回答时，在克隆目录复制 `.env.example` 为 `.env`，填入 `DEEPSEEK_API_KEY`，再执行 `docker compose up -d`。密钥只留在本机，不会进入仓库或镜像。

```powershell
docker compose logs -f api
docker compose down
```

`docker compose down` 会停止容器并保留数据库卷。第一次构建会下载 CPU 版 PyTorch 和两个 Sentence Transformers 模型，需要能访问 Docker Hub、PyTorch 和 Hugging Face。模型在构建时下载，启动后离线加载。

## PostgreSQL + pgvector

`sql/init.sql` 会创建 pgvector 扩展、文档表和全文索引。只想在本机 Python 里连数据库、不启动 API 容器时：

```powershell
docker compose up -d postgres
python -m pip install -e ".[postgres,models]"
$env:BACKEND="postgres"
python -m apex_rag.cli ingest
```

容器内的数据库地址是 `postgres:5432`。本机 Python 继续使用 `.env` 里的 `localhost:5432`。

## 文档

- [开发文档](docs/DEVELOPMENT_PLAN.md)
- [验收标准矩阵](docs/ACCEPTANCE_MATRIX.md)
- [生产路线图](docs/roadmap.md)
- [当前结果](reports/results.md)

## 浏览器检索界面

使用上面的 `docker compose up --build -d` 后，直接打开 `http://127.0.0.1:18000/`。如果要在本机 Python 里启动，而不是用 API 容器：

```powershell
python -m pip install -e ".[api]"
python -m uvicorn apex_rag.api:app --host 127.0.0.1 --port 18000
```

网页会调用 `/api/v1/search`，展示文档内容、元数据、Dense/Sparse/RRF/重排分数和查询耗时；`/docs` 仍可用于 API 调试。
