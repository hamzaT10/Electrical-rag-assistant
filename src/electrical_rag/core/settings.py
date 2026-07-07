from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Path("Data")
    vectorstore_dir: Path = Path("vectorstore")
    vector_backend: str = Field(default="faiss", pattern="^(faiss|qdrant)$")
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "electrical_rag_chunks"
    upload_dir: Path = Path("Data/uploads")
    max_upload_size_mb: int = Field(default=25, ge=1, le=200)

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    preload_rag_service: bool = True
    warmup_embedding_model: bool = True
    chunk_size: int = Field(default=800, ge=100, le=4096)
    chunk_overlap: int = Field(default=120, ge=0, le=1024)
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    rag_min_retrieval_score: float = Field(default=0.0, ge=0.0, le=1.0)
    enable_reranker: bool = False
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_chunks_per_source_page: int = Field(default=2, ge=1, le=10)
    max_context_chars: int = Field(default=3500, ge=0, le=50000)
    max_chunk_chars: int = Field(default=900, ge=0, le=20000)

    enable_ocr_fallback: bool = False
    ocr_languages: str = "en"

    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_api_key: str = "lm-studio"
    lmstudio_model: str = "local-model"
    lmstudio_health_timeout_seconds: float = Field(default=3.0, ge=0.5, le=30.0)
    llm_temperature: float = Field(default=0.1, ge=0.0, le=1.0)

    app_log_level: str = "INFO"
    app_log_format: str = Field(default="json", pattern="^(json|text)$")
    app_release: str = "local"
    enable_metrics: bool = True
    enable_langfuse: bool = False
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"
    langfuse_environment: str = "development"
    langfuse_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    langfuse_capture_content: bool = False
    frontend_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    database_url: str = "postgresql+psycopg://electrical_rag:electrical_rag@localhost:5432/electrical_rag"
    redis_url: str = "redis://localhost:6379/0"
    enable_chat_cache: bool = True
    chat_cache_ttl_seconds: int = Field(default=3600, ge=1, le=86400)
    enable_rate_limit: bool = True
    rate_limit_requests: int = Field(default=30, ge=1, le=10000)
    rate_limit_window_seconds: int = Field(default=60, ge=1, le=86400)
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    @property
    def ocr_language_list(self) -> list[str]:
        return [item.strip() for item in self.ocr_languages.split(",") if item.strip()]

    @property
    def frontend_origin_list(self) -> list[str]:
        return [item.strip() for item in self.frontend_origins.split(",") if item.strip()]
