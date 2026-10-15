"""DeepSeek answer generation and reference-answer scoring.

The answer layer is deliberately separate from retrieval.  A provider failure
returns a structured error and never prevents the caller from receiving ranked
retrieval results.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from apex_rag.models import SearchResult


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.$%/-][a-z0-9]+)*", re.IGNORECASE)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from",
    "how", "if", "in", "is", "it", "of", "on", "or", "the", "to", "what", "when",
    "which", "with", "will", "can", "i", "my", "you", "your", "under", "than", "this",
}


@dataclass(frozen=True)
class AnswerScore:
    """Deterministic reference-answer comparison values in the [0, 1] range."""

    token_precision: float
    token_recall: float
    token_f1: float
    numeric_coverage: float
    score: float

    def as_dict(self) -> dict[str, float]:
        return {
            "token_precision": self.token_precision,
            "token_recall": self.token_recall,
            "token_f1": self.token_f1,
            "numeric_coverage": self.numeric_coverage,
            "score": self.score,
        }


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text) if token.lower() not in _STOPWORDS]


def score_answer(answer: str, reference: str) -> AnswerScore:
    """Score lexical fact overlap with the supplied gold answer.

    This is a transparent regression score, not an LLM judge.  Numeric facts
    receive a separate coverage score because policy answers are number-heavy.
    """
    predicted = _tokens(answer)
    expected = _tokens(reference)
    predicted_counts: dict[str, int] = {}
    expected_counts: dict[str, int] = {}
    for token in predicted:
        predicted_counts[token] = predicted_counts.get(token, 0) + 1
    for token in expected:
        expected_counts[token] = expected_counts.get(token, 0) + 1
    overlap = sum(min(predicted_counts.get(token, 0), count) for token, count in expected_counts.items())
    precision = overlap / max(len(predicted), 1)
    recall = overlap / max(len(expected), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    expected_numbers = set(re.findall(r"\d+(?:\.\d+)?%?", reference))
    answer_numbers = set(re.findall(r"\d+(?:\.\d+)?%?", answer))
    numeric_coverage = len(expected_numbers & answer_numbers) / max(len(expected_numbers), 1)
    # A small numeric-fact bonus/penalty keeps the score useful for policy data
    # while retaining lexical transparency for manual review.
    score = 0.7 * f1 + 0.3 * numeric_coverage
    return AnswerScore(precision, recall, f1, numeric_coverage, score)


def extractive_fallback(results: Iterable[SearchResult]) -> str:
    """Return grounded context when the remote provider is temporarily unavailable."""
    items = list(results)
    if not items:
        return "No matching policy context was retrieved."
    top = items[0].chunk
    return f"Based on the retrieved policy ({top.parent_doc_id}, {top.title}): {top.content}"


@dataclass(frozen=True)
class Judgment:
    """LLM verdicts reduced to context relevance and faithfulness in [0, 1]."""

    context_relevance: float | None
    faithfulness: float | None
    chunk_verdicts: tuple[dict[str, Any], ...] = ()
    claim_verdicts: tuple[dict[str, Any], ...] = ()
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Serialize scores and the verdicts used to compute them."""
        return {
            "context_relevance": "" if self.context_relevance is None else self.context_relevance,
            "faithfulness": "" if self.faithfulness is None else self.faithfulness,
            "chunk_verdicts": json.dumps(list(self.chunk_verdicts), ensure_ascii=False),
            "claim_verdicts": json.dumps(list(self.claim_verdicts), ensure_ascii=False),
            "judge_error": self.error,
        }


def format_context(results: Iterable[SearchResult]) -> str:
    """Render retrieved chunks with stable document ids for prompting."""
    parts: list[str] = []
    for result in results:
        chunk = result.chunk
        parts.append(
            f"[{chunk.parent_doc_id}] {chunk.title}\n"
            f"Category: {chunk.category}; Card tier: {chunk.card_tier}; Year: {chunk.effective_year}\n"
            f"{chunk.content}"
        )
    return "\n\n".join(parts)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Pull the JSON object out of a model response, including fenced output."""
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("judge response did not contain a JSON object")
    data = json.loads(candidate[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("judge response must be a JSON object")
    return data


def interpret_judgment(raw: str, chunk_ids: list[str], answer: str) -> Judgment:
    """Compute relevance and faithfulness from structured chunk and claim verdicts.

    Context relevance is the share of retrieved documents marked relevant.
    Faithfulness is the share of atomic answer claims marked supported.
    A non-empty answer with no claims is a judge failure, so faithfulness stays empty.
    """
    if not chunk_ids and not answer.strip():
        return Judgment(0.0, 0.0)
    data = _extract_json_object(raw)
    chunks = data.get("chunks", [])
    claims = data.get("claims", [])
    if not isinstance(chunks, list) or not isinstance(claims, list):
        raise ValueError("chunks and claims must be lists")
    relevant_ids = {
        str(item.get("doc_id", ""))
        for item in chunks
        if isinstance(item, dict) and _as_bool(item.get("relevant"))
    }
    chunk_verdicts = tuple(
        {"doc_id": doc_id, "relevant": doc_id in relevant_ids} for doc_id in chunk_ids
    )
    context_relevance = (
        sum(1 for item in chunk_verdicts if item["relevant"]) / len(chunk_ids) if chunk_ids else 0.0
    )
    claim_verdicts = tuple(
        {"claim": str(item.get("claim", "")), "supported": _as_bool(item.get("supported"))}
        for item in claims
        if isinstance(item, dict) and str(item.get("claim", "")).strip()
    )
    if not answer.strip():
        return Judgment(context_relevance, 0.0, chunk_verdicts, claim_verdicts)
    if not claim_verdicts:
        return Judgment(
            context_relevance, None, chunk_verdicts, claim_verdicts,
            "judge returned no claims for a non-empty answer",
        )
    faithfulness = sum(1 for item in claim_verdicts if item["supported"]) / len(claim_verdicts)
    return Judgment(context_relevance, faithfulness, chunk_verdicts, claim_verdicts)


class DeepSeekAnswerer:
    """Minimal OpenAI-compatible DeepSeek chat-completions client."""

    def __init__(self, api_key: str, model: str = "deepseek-flash",
                 base_url: str = "https://api.deepseek.com", timeout: float = 45.0) -> None:
        if not api_key.strip():
            raise ValueError("DeepSeek API key is empty")
        self.api_key = api_key.strip()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _complete(self, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        """Send one chat completion and return the assistant text."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Thinking shares this token budget. Leaving it on can finish with an empty answer.
            "thinking": {"type": "disabled"},
        }
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            # Do not include response bodies: they may contain request metadata.
            raise RuntimeError(f"DeepSeek request failed with HTTP {exc.code}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError("DeepSeek request could not be completed") from exc
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("DeepSeek returned an unexpected response") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("DeepSeek returned an empty answer")
        return content.strip()

    def _retry(self, operation: Any, retries: int) -> Any:
        last_error: Exception | None = None
        for attempt in range(max(0, retries) + 1):
            try:
                return operation()
            except (RuntimeError, ValueError) as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(0.5 * (attempt + 1))
        assert last_error is not None
        if isinstance(last_error, RuntimeError):
            raise last_error
        raise RuntimeError(str(last_error)) from last_error

    def _answer_once(self, query: str, results: Iterable[SearchResult]) -> str:
        """Make one answer request grounded only in retrieved context chunks."""
        system = (
            "You are an accurate credit-card customer-service assistant. "
            "Answer using only the supplied retrieved context. Do not invent policy details. "
            "If the context is insufficient, say so clearly. Answer in the same language as the user. "
            "Give a concise direct answer and preserve exact amounts, percentages, dates, and limits."
        )
        user = f"User question:\n{query}\n\nRetrieved context:\n{format_context(results)}"
        return self._complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.1,
            max_tokens=500,
        )

    def answer(self, query: str, results: Iterable[SearchResult], retries: int = 2) -> str:
        """Generate an answer, retrying transient provider failures."""
        return self._retry(lambda: self._answer_once(query, results), retries)

    def _judge_once(self, query: str, results: list[SearchResult], answer: str) -> Judgment:
        chunk_ids = [result.chunk.parent_doc_id for result in results]
        system = (
            "You are a strict RAG evaluator. Return only a JSON object with keys "
            "\"chunks\" and \"claims\". "
            "chunks is a list of {\"doc_id\": string, \"relevant\": boolean}. "
            "Mark relevant true only when that passage helps answer the question. "
            "Include every supplied doc_id exactly once. "
            "claims is a list of atomic statements from the answer, each "
            "{\"claim\": string, \"supported\": boolean}. "
            "supported is true only when the statement is entailed by the retrieved context. "
            "Do not add prose or markdown."
        )
        user = (
            f"Question:\n{query}\n\nRetrieved context:\n{format_context(results)}\n\n"
            f"Answer:\n{answer}"
        )
        raw = self._complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=800,
        )
        return interpret_judgment(raw, chunk_ids, answer)

    def judge(
        self, query: str, results: Iterable[SearchResult], answer: str, retries: int = 2
    ) -> Judgment:
        """Score context relevance and faithfulness, retrying transient failures."""
        items = list(results)
        chunk_ids = [result.chunk.parent_doc_id for result in items]
        if not chunk_ids:
            return Judgment(0.0, 0.0)
        return self._retry(lambda: self._judge_once(query, items, answer), retries)


def build_answerer(api_key: str, model: str, base_url: str) -> DeepSeekAnswerer | None:
    """Return a configured answerer, or None when answering is disabled."""
    return DeepSeekAnswerer(api_key, model, base_url) if api_key.strip() else None
