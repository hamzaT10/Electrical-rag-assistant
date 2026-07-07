from __future__ import annotations

from collections.abc import Generator

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from electrical_rag.api.app import app
from electrical_rag.db.repositories import ChatRepository
from electrical_rag.db.session import Base, get_db_session


def test_chat_history_endpoints_return_saved_messages():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine)

    def override_db() -> Generator[Session, None, None]:
        db = session_local()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db_session] = override_db
    try:
        db = session_local()
        repo = ChatRepository(db)
        user = repo.get_or_create_demo_user()
        chat_session = repo.create_chat_session(user.id, title="api history")
        repo.create_message(chat_session.id, "user", "Question")
        repo.create_message(chat_session.id, "assistant", "Answer")
        session_id = chat_session.id
        db.close()

        client = TestClient(app)

        sessions_response = client.get("/chat/sessions")
        assert sessions_response.status_code == 200
        sessions = sessions_response.json()
        assert len(sessions) == 1
        assert sessions[0]["title"] == "api history"

        messages_response = client.get(f"/chat/sessions/{session_id}/messages")
        assert messages_response.status_code == 200
        messages_payload = messages_response.json()
        assert messages_payload["session_id"] == session_id
        assert [item["role"] for item in messages_payload["messages"]] == [
            "user",
            "assistant",
        ]

        missing_response = client.get("/chat/sessions/999/messages")
        assert missing_response.status_code == 404
    finally:
        app.dependency_overrides.clear()
