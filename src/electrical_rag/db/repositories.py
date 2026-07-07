from __future__ import annotations

from datetime import datetime
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from electrical_rag.db.models import (
    ChatSession,
    Document,
    IngestionJob,
    Message,
    RetrievedSource,
    User,
)


class IngestionStatsLike(Protocol):
    pdf_files: int
    pages_loaded: int
    pages_ocr: int
    chunks_created: int


class ChatRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_or_create_demo_user(self) -> User:
        user = self.db.query(User).filter(User.email == "demo@electrical-rag.local").one_or_none()
        if user is not None:
            return user

        user = User(
            email="demo@electrical-rag.local",
            hashed_password="not-used-demo-user",
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def create_chat_session(self, user_id: int, title: str | None = None) -> ChatSession:
        session = ChatSession(user_id=user_id, title=title)
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session

    def create_message(self, session_id: int, role: str, content: str) -> Message:
        message = Message(
            session_id=session_id,
            role=role,
            content=content,
        )
        self.db.add(message)
        self.db.commit()
        self.db.refresh(message)
        return message

    def get_or_create_document(self, source_path: str) -> Document:
        document = self.db.query(Document).filter(Document.source_path == source_path).one_or_none()
        if document is not None:
            return document

        filename = source_path.split("/")[-1].split("\\")[-1]
        document = Document(
            source_path=source_path,
            filename=filename,
            status="indexed",
        )
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        return document

    def save_retrieved_sources(
        self,
        message_id: int,
        citations: list[dict[str, object]],
    ) -> list[RetrievedSource]:
        saved_sources: list[RetrievedSource] = []

        for citation in citations:
            source = str(citation.get("source", "unknown"))
            document = self.get_or_create_document(source)

            page_value = citation.get("page")
            page = page_value if isinstance(page_value, int) else None

            score_value = citation.get("score")
            score = float(score_value) if isinstance(score_value, (int, float)) else None

            retrieved_source = RetrievedSource(
                message_id=message_id,
                document_id=document.id,
                page=page,
                chunk_id=None,
                score=score,
                chunk_text=None,
            )
            self.db.add(retrieved_source)
            saved_sources.append(retrieved_source)

        self.db.commit()
        for source_item in saved_sources:
            self.db.refresh(source_item)

        return saved_sources

    def list_chat_sessions(self, user_id: int, limit: int = 50) -> list[ChatSession]:
        statement = (
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.created_at.desc(), ChatSession.id.desc())
            .limit(limit)
        )
        return list(self.db.scalars(statement).all())

    def get_chat_session(self, session_id: int) -> ChatSession | None:
        return self.db.get(ChatSession, session_id)

    def list_messages(self, session_id: int) -> list[Message]:
        statement = (
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.asc(), Message.id.asc())
        )
        return list(self.db.scalars(statement).all())


class IngestionJobRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_uploaded_document(
        self,
        source_path: str,
        filename: str,
        storage_path: str,
    ) -> Document:
        document = Document(
            source_path=source_path,
            filename=filename,
            storage_path=storage_path,
            status="uploaded",
        )
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        return document

    def get_or_create_ingested_document(
        self,
        source_path: str,
        filename: str,
        storage_path: str,
    ) -> Document:
        document = self.db.query(Document).filter(Document.source_path == source_path).one_or_none()
        if document is None:
            document = Document(
                source_path=source_path,
                filename=filename,
                storage_path=storage_path,
                status="indexing",
            )
            self.db.add(document)
        else:
            document.filename = filename
            document.storage_path = storage_path
            document.status = "indexing"

        self.db.commit()
        self.db.refresh(document)
        return document

    def mark_document_indexing(self, document_id: int) -> Document | None:
        document = self.db.get(Document, document_id)
        if document is None:
            return None

        document.status = "indexing"
        self.db.commit()
        self.db.refresh(document)
        return document

    def mark_document_indexed(
        self,
        document_id: int,
        page_count: int | None = None,
    ) -> Document | None:
        document = self.db.get(Document, document_id)
        if document is None:
            return None

        document.status = "indexed"
        document.page_count = page_count
        document.indexed_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(document)
        return document

    def mark_document_failed(self, document_id: int) -> Document | None:
        document = self.db.get(Document, document_id)
        if document is None:
            return None

        document.status = "failed"
        self.db.commit()
        self.db.refresh(document)
        return document

    def list_documents(self, limit: int = 100, status: str | None = None) -> list[Document]:
        statement = select(Document).order_by(Document.created_at.desc(), Document.id.desc())
        if status is not None:
            statement = statement.where(Document.status == status)
        statement = statement.limit(limit)
        return list(self.db.scalars(statement).all())

    def create_job(
        self,
        source_path: str | None = None,
        document_id: int | None = None,
    ) -> IngestionJob:
        job = IngestionJob(status="pending", source_path=source_path, document_id=document_id)
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        return job

    def set_task_id(self, job_id: int, task_id: str) -> IngestionJob | None:
        job = self.get_job(job_id)
        if job is None:
            return None

        job.task_id = task_id
        self.db.commit()
        self.db.refresh(job)
        return job

    def get_job(self, job_id: int) -> IngestionJob | None:
        return self.db.get(IngestionJob, job_id)

    def list_jobs(self, limit: int = 50) -> list[IngestionJob]:
        statement = (
            select(IngestionJob)
            .order_by(IngestionJob.created_at.desc(), IngestionJob.id.desc())
            .limit(limit)
        )
        return list(self.db.scalars(statement).all())

    def mark_running(self, job_id: int) -> IngestionJob | None:
        job = self.get_job(job_id)
        if job is None:
            return None

        job.status = "running"
        job.started_at = datetime.utcnow()
        job.finished_at = None
        job.error_message = None
        self.db.commit()
        self.db.refresh(job)
        return job

    def mark_completed(self, job_id: int, stats: IngestionStatsLike) -> IngestionJob | None:
        job = self.get_job(job_id)
        if job is None:
            return None

        job.status = "completed"
        job.pdf_files = stats.pdf_files
        job.pages_loaded = stats.pages_loaded
        job.pages_ocr = stats.pages_ocr
        job.chunks_created = stats.chunks_created
        job.finished_at = datetime.utcnow()
        job.error_message = None
        self.db.commit()
        self.db.refresh(job)
        return job

    def mark_failed(self, job_id: int, error_message: str) -> IngestionJob | None:
        job = self.get_job(job_id)
        if job is None:
            return None

        job.status = "failed"
        job.finished_at = datetime.utcnow()
        job.error_message = error_message[:4000]
        self.db.commit()
        self.db.refresh(job)
        return job
