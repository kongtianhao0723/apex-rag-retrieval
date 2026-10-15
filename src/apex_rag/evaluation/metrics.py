"""Standard information-retrieval metrics."""

import math
from typing import Iterable


def _ids(results: Iterable[object]) -> list[str]:
    return [getattr(getattr(result, "chunk"), "parent_doc_id") for result in results]


def recall_at_k(results: Iterable[object], relevant: set[str], k: int) -> float:
    """Compute Recall@K using parent document IDs."""
    found = set(_ids(list(results)[:k]))
    return len(found & relevant) / len(relevant) if relevant else 0.0


def precision_at_k(results: Iterable[object], relevant: set[str], k: int) -> float:
    """Compute Precision@K using parent document IDs."""
    ids = _ids(list(results)[:k])
    return sum(doc_id in relevant for doc_id in ids) / max(k, 1)


def reciprocal_rank(results: Iterable[object], relevant: set[str]) -> float:
    """Compute reciprocal rank of the first relevant result."""
    for index, doc_id in enumerate(_ids(results), start=1):
        if doc_id in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(results: Iterable[object], relevant: set[str], k: int) -> float:
    """Compute binary-relevance nDCG@K."""
    ids = _ids(list(results)[:k])
    dcg = sum((1.0 / math.log2(index + 2)) for index, doc_id in enumerate(ids) if doc_id in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    return dcg / idcg if idcg else 0.0
