"""Optional PostgreSQL + pgvector persistence layer."""

from __future__ import annotations

import re
from typing import Any, Iterable

import numpy as np

from apex_rag.models import Chunk


class PostgresStore:
    """Small explicit database adapter used by production ingestion/search."""

    _IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")

    def __init__(self, database_url: str, dimension: int = 384,
                 table_name: str = "documents") -> None:
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Install PostgreSQL dependencies with: python -m pip install -e '.[postgres]'"
            ) from exc
        self.psycopg = psycopg
        self.register_vector = register_vector
        self.connection = psycopg.connect(database_url)
        self.register_vector(self.connection)
        self.dimension = dimension
        if not self._IDENTIFIER.fullmatch(table_name):
            raise ValueError(f"Unsafe PostgreSQL table name: {table_name}")
        self.table_name = table_name

    def ensure_table(self) -> None:
        """Create the configured document table when it does not exist."""
        table = self.table_name
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    chunk_id TEXT PRIMARY KEY,
                    parent_doc_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    category TEXT,
                    card_tier TEXT,
                    effective_year INTEGER,
                    content TEXT NOT NULL,
                    embedding VECTOR({int(self.dimension)}),
                    search_vector TSVECTOR,
                    chunking_strategy TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS {table}_search_vector_idx "
                f"ON {table} USING gin (search_vector)"
            )
        self.connection.commit()

    def reset_table(self) -> None:
        """Remove all rows and vector indexes from the configured table."""
        table = self.table_name
        with self.connection.cursor() as cursor:
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
        self.connection.commit()
        self.ensure_table()

    def close(self) -> None:
        """Close the database connection."""
        self.connection.close()

    def upsert_chunks(self, chunks: Iterable[Chunk], embeddings: np.ndarray) -> int:
        """Upsert chunks and vectors in one transaction."""
        rows = list(chunks)
        if len(rows) != len(embeddings):
            raise ValueError("Chunk and embedding counts must match")
        with self.connection.cursor() as cursor:
            for chunk, embedding in zip(rows, embeddings):
                text = f"{chunk.title} {chunk.category} {chunk.card_tier} {chunk.content}"
                cursor.execute(
                    f"""
                    INSERT INTO {self.table_name}
                        (chunk_id, parent_doc_id, title, category, card_tier,
                         effective_year, content, embedding, search_vector, chunking_strategy)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, to_tsvector('english', %s), %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        parent_doc_id = EXCLUDED.parent_doc_id,
                        title = EXCLUDED.title,
                        category = EXCLUDED.category,
                        card_tier = EXCLUDED.card_tier,
                        effective_year = EXCLUDED.effective_year,
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        search_vector = EXCLUDED.search_vector,
                        chunking_strategy = EXCLUDED.chunking_strategy
                    """,
                    (chunk.chunk_id, chunk.parent_doc_id, chunk.title, chunk.category,
                     chunk.card_tier, chunk.effective_year, chunk.content,
                     embedding, text, chunk.strategy),
                )
        self.connection.commit()
        return len(rows)

    def create_index(self, index_type: str = "hnsw", m: int = 16,
                     ef_construction: int = 128, lists: int = 10,
                     index_name: str | None = None,
                     drop_existing: bool = False) -> None:
        """Create the configured approximate vector index."""
        table = self.table_name
        if drop_existing:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "SELECT indexname FROM pg_indexes WHERE tablename = %s "
                    "AND indexname LIKE %s", (table, f"{table}_embedding_%")
                )
                for (existing,) in cursor.fetchall():
                    cursor.execute(f"DROP INDEX IF EXISTS {existing}")
            self.connection.commit()
        index_name = index_name or f"{table}_embedding_{index_type.lower()}_idx"
        if not self._IDENTIFIER.fullmatch(index_name):
            raise ValueError(f"Unsafe PostgreSQL index name: {index_name}")
        if index_type.lower() == "hnsw":
            sql = (
                f"CREATE INDEX IF NOT EXISTS {index_name} "
                f"ON {table} USING hnsw (embedding vector_cosine_ops) "
                f"WITH (m = {int(m)}, ef_construction = {int(ef_construction)})"
            )
        elif index_type.lower() == "ivfflat":
            sql = (
                f"CREATE INDEX IF NOT EXISTS {index_name} "
                f"ON {table} USING ivfflat (embedding vector_cosine_ops) "
                f"WITH (lists = {int(lists)})"
            )
        else:
            raise ValueError("index_type must be hnsw or ivfflat")
        with self.connection.cursor() as cursor:
            cursor.execute(sql)
        self.connection.commit()

    def set_hnsw_search(self, ef_search: int) -> None:
        """Set HNSW candidate breadth for the current session."""
        value = int(ef_search)
        if value < 1:
            raise ValueError("hnsw.ef_search must be positive")
        with self.connection.cursor() as cursor:
            # PostgreSQL SET does not accept bind parameters for the value.
            cursor.execute(f"SET hnsw.ef_search = {value}")
        self.connection.commit()

    def distinct_values(self, column: str) -> list[str]:
        """Return sorted non-empty values for an allow-listed metadata column."""
        if column not in {"category", "card_tier"}:
            raise ValueError("column must be category or card_tier")
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"SELECT DISTINCT {column} FROM {self.table_name} "
                f"WHERE {column} IS NOT NULL AND {column} <> '' ORDER BY {column}"
            )
            return [str(row[0]) for row in cursor.fetchall()]

    def set_ivfflat_probes(self, probes: int) -> None:
        """Set IVFFlat list probes for the current session."""
        value = int(probes)
        if value < 1:
            raise ValueError("ivfflat.probes must be positive")
        with self.connection.cursor() as cursor:
            cursor.execute(f"SET ivfflat.probes = {value}")
        self.connection.commit()
