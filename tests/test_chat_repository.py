from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from electrical_rag.db.repositories import ChatRepository
from electrical_rag.db.session import Base


def test_chat_repository_saves_chat_flow():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    session_local = sessionmaker(bind=engine)
    db = session_local()

    repo = ChatRepository(db)
    user = repo.get_or_create_demo_user()
    chat_session = repo.create_chat_session(user.id, title="test")
    user_message = repo.create_message(chat_session.id, "user", "What is voltage unbalance?")
    assistant_message = repo.create_message(
        chat_session.id,
        "assistant",
        "Voltage unbalance is...",
    )

    sources = repo.save_retrieved_sources(
        assistant_message.id,
        [
            {
                "source": "Guides_Manuals/test.pdf",
                "page": 2,
                "score": 0.91,
            }
        ],
    )

    assert user.email == "demo@electrical-rag.local"
    assert chat_session.user_id == user.id
    assert user_message.role == "user"
    assert assistant_message.role == "assistant"
    assert len(sources) == 1
    assert sources[0].page == 2


def test_chat_repository_lists_sessions_and_messages():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    session_local = sessionmaker(bind=engine)
    db = session_local()

    repo = ChatRepository(db)
    user = repo.get_or_create_demo_user()
    chat_session = repo.create_chat_session(user.id, title="history test")
    repo.create_message(chat_session.id, "user", "Question")
    repo.create_message(chat_session.id, "assistant", "Answer")

    sessions = repo.list_chat_sessions(user.id)
    messages = repo.list_messages(chat_session.id)

    assert len(sessions) == 1
    assert sessions[0].title == "history test"
    assert [message.role for message in messages] == ["user", "assistant"]
