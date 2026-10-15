# DeepSeek Answer Evaluation

Model: `deepseek-flash`; queries: 30; DeepSeek answers: 30; retrieval fallback: 0; answers available: 30.
Token F1、Numeric coverage 和 Composite score 仍是对照 `gold_response` 的词面回归分，公式为 0.7 × Token F1 + 0.3 × Numeric coverage。
Context Relevance 与 Faithfulness 由同一 DeepSeek 模型做 LLM 评判：相关性 = 被标为有助于回答的检索篇数 / 检索篇数；忠实度 = 被上下文支持的原子陈述数 / 陈述总数。评判失败时该题分数留空，不中断其余题目。

| Metric | All answers | DeepSeek only | Fallback only |
|---|---:|---:|---:|
| Token F1 | 0.6486 | 0.6486 | n/a |
| Numeric coverage | 0.8689 | 0.8689 | n/a |
| Composite score | 0.7147 | 0.7147 | n/a |
| Context Relevance | 0.3067 | 0.3067 | n/a |
| Faithfulness | 1.0000 | 1.0000 | n/a |

逐题答案、标准答案、分块判定、陈述判定、错误信息和耗时见 `reports/answer_evaluation.csv`。回答失败时不影响原有检索结果。
