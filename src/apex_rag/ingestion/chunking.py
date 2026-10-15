"""Chunking strategies used for the controlled retrieval experiment."""

import re

from apex_rag.models import Chunk, Document


def original_chunks(documents: list[Document]) -> list[Chunk]:
    """Represent each supplied document as one searchable chunk."""
    return [
        Chunk(d.doc_id, d.doc_id, d.title, d.category, d.card_tier, d.effective_year, d.content, "original")
        for d in documents
    ]


def sliding_window_chunks(
    documents: list[Document], window_size: int = 40, overlap: int = 10
) -> list[Chunk]:
    """Split documents into word windows while preserving parent IDs."""
    if window_size <= overlap or overlap < 0:
        raise ValueError("window_size must be greater than non-negative overlap")
    chunks: list[Chunk] = []
    step = window_size - overlap
    for document in documents:
        words = document.content.split()
        for start in range(0, len(words), step):
            text = " ".join(words[start : start + window_size])
            if not text:
                continue
            chunk_id = f"{document.doc_id}-chunk-{start // step}"
            chunks.append(
                Chunk(chunk_id, document.doc_id, document.title, document.category,
                      document.card_tier, document.effective_year, text, "sliding_window")
            )
            if start + window_size >= len(words):
                break
    return chunks


def hierarchical_chunks(documents: list[Document]) -> list[Chunk]:
    """Create sentence-level chunks with title and metadata context."""
    chunks: list[Chunk] = []
    for document in documents:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", document.content) if s.strip()]
        for index, sentence in enumerate(sentences):
            context = f"Title: {document.title}. Category: {document.category}. {sentence}"
            chunks.append(
                Chunk(f"{document.doc_id}-semantic-{index}", document.doc_id, document.title,
                      document.category, document.card_tier, document.effective_year,
                      context, "hierarchical")
            )
    return chunks
