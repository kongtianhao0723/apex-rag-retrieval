CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    chunk_id TEXT PRIMARY KEY,
    parent_doc_id TEXT NOT NULL,
    title TEXT NOT NULL,
    category TEXT,
    card_tier TEXT,
    effective_year INTEGER,
    content TEXT NOT NULL,
    embedding VECTOR(384),
    search_vector TSVECTOR,
    chunking_strategy TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS documents_search_vector_idx
    ON documents USING gin (search_vector);
