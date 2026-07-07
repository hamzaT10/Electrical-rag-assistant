from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    document_id: int | None = Field(default=None, ge=1)


class Citation(BaseModel):
    source: str
    page: int | None = None
    score: float | None = None
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    session_id: int | None = None


class ChatSessionSummary(BaseModel):
    id: int
    title: str | None = None
    created_at: datetime


class ChatMessageItem(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime


class ChatSessionMessagesResponse(BaseModel):
    session_id: int
    messages: list[ChatMessageItem]


class IngestionJobCreate(BaseModel):
    source_path: str | None = Field(default=None, max_length=1024)


class IngestionJobResponse(BaseModel):
    id: int
    status: str
    document_id: int | None = None
    source_path: str | None = None
    task_id: str | None = None
    pdf_files: int | None = None
    pages_loaded: int | None = None
    pages_ocr: int | None = None
    chunks_created: int | None = None
    error_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class DocumentUploadResponse(BaseModel):
    filename: str
    saved_path: str
    document_id: int
    job_id: int
    job_status: str


class DocumentSummary(BaseModel):
    id: int
    filename: str
    source_path: str
    status: str
    page_count: int | None = None
    created_at: datetime
    indexed_at: datetime | None = None


class HealthResponse(BaseModel):
    status: str
    vectorstore_ready: bool
    llm_ready: bool
    llm_error: str | None = None
    rag_service_ready: bool
    rag_service_error: str | None = None
    rag_service_startup_seconds: float | None = None
    embedding_dimension: int | None = None
