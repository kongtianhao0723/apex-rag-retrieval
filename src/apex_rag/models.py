"""Typed domain models used by ingestion, retrieval, and evaluation."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """A source knowledge-base document."""

    doc_id: str
    title: str
    category: str
    card_tier: str
    effective_year: int
    content: str


@dataclass(frozen=True)
class Chunk:
    """A searchable chunk retaining its source document identity."""

    chunk_id: str
    parent_doc_id: str
    title: str
    category: str
    card_tier: str
    effective_year: int
    content: str
    strategy: str


@dataclass
class SearchResult:
    """A ranked retrieval result with diagnostic scores."""

    chunk: Chunk
    rank: int = 0
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the result for reports and APIs."""
        return {
            "doc_id": self.chunk.parent_doc_id,
            "chunk_id": self.chunk.chunk_id,
            "title": self.chunk.title,
            "content": self.chunk.content,
            "category": self.chunk.category,
            "card_tier": self.chunk.card_tier,
            "effective_year": self.chunk.effective_year,
            "rank": self.rank,
            "scores": {
                "dense": self.dense_score,
                "sparse": self.sparse_score,
                "rrf": self.rrf_score,
                "rerank": self.rerank_score,
                **self.scores,
            },
        }
