"""Load and validate the supplied JSON datasets."""

import json
from pathlib import Path
from typing import Any

from apex_rag.models import Document


def load_documents(path: Path) -> list[Document]:
    """Load knowledge-base documents and validate required fields."""
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list in {path}")
    required = {"doc_id", "title", "category", "card_tier", "effective_year", "content"}
    documents: list[Document] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError(f"Document {index} is missing one of {sorted(required)}")
        documents.append(
            Document(
                doc_id=str(item["doc_id"]),
                title=str(item["title"]),
                category=str(item["category"]),
                card_tier=str(item["card_tier"]),
                effective_year=int(item["effective_year"]),
                content=str(item["content"]),
            )
        )
    return documents


def load_queries(path: Path) -> list[dict[str, Any]]:
    """Load and validate evaluation queries."""
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list in {path}")
    required = {"query_id", "query", "relevant_doc_ids", "gold_response"}
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError(f"Query {index} is missing one of {sorted(required)}")
    return raw
