"""Embedding model adapters with an explicit local fallback."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Protocol shared by local and Sentence Transformers embedders."""

    dimension: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into a two-dimensional float array."""


class SentenceTransformerEmbedder:
    """Adapter around sentence-transformers for production runs."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "Install the models extra to use Sentence Transformers: "
                "python -m pip install -e '.[models]'"
            ) from exc
        self.model = SentenceTransformer(model_name)
        self.dimension = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode and L2-normalize text embeddings."""
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False))


def create_embedder(model_name: str) -> SentenceTransformerEmbedder:
    """Create the configured production embedder."""
    return SentenceTransformerEmbedder(model_name)
