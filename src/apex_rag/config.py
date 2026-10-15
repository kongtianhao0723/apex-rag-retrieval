"""Centralized application configuration."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment and optional .env file."""

    database_url: str = "postgresql://rag:rag@localhost:5432/rag"
    backend: str = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-flash"
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_timeout: float = 45.0
    enable_answering: bool = True
    embedding_dimension: int = 384
    dense_top_k: int = 20
    sparse_top_k: int = 20
    final_top_n: int = 5
    rrf_k: int = 60
    index_type: str = "hnsw"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 128
    hnsw_ef_search: int = 80
    ivfflat_lists: int = 10
    ivfflat_probes: int = 5
    data_dir: Path = Path("data")
    report_dir: Path = Path("reports")

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", case_sensitive=False)


settings = Settings()
