"""SQLite schema works with Chainlit's SQLAlchemyDataLayer; demo login check."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.user import User
from pydantic import SecretStr

from research_assistant.agent.state import SessionState, TurnResult
from research_assistant.auth import check_demo_credentials
from research_assistant.config import Settings
from research_assistant.export import conversation_from_thread, turn_metadata
from research_assistant.persistence import init_db, sqlalchemy_url

TABLES = {"users", "threads", "steps", "elements", "feedbacks"}


def test_init_db_creates_tables_and_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "sub" / "chat.db"
    init_db(db)
    init_db(db)
    with sqlite3.connect(db) as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert names >= TABLES


async def test_data_layer_round_trip(tmp_path: Path) -> None:
    db = init_db(tmp_path / "chat.db")
    dl = SQLAlchemyDataLayer(conninfo=sqlalchemy_url(db))
    try:
        user = await dl.create_user(User(identifier="demo", metadata={"role": "demo"}))
        assert user is not None and user.identifier == "demo"

        await dl.update_thread("t1", name="First question", user_id=user.id)
        result = TurnResult(session_id="t1", turn_id="x", question="Q ε?", answer="A δ.")
        # create_step is queued until the first user message inside a websocket session;
        # call the undecorated method directly.
        create_step = SQLAlchemyDataLayer.create_step.__wrapped__  # type: ignore[attr-defined]
        await create_step(
            dl,
            {
                "id": "s1",
                "threadId": "t1",
                "name": "demo",
                "type": "user_message",
                "output": "Q ε?",
                "createdAt": "2026-11-22T10:00:00Z",
                "streaming": False,
                "metadata": {},
                "command": None,
                "modes": None,
                "tags": None,
            },
        )
        await create_step(
            dl,
            {
                "id": "s2",
                "threadId": "t1",
                "parentId": None,
                "name": "Assistant",
                "type": "assistant_message",
                "output": "A δ.",
                "createdAt": "2026-11-22T10:00:05Z",
                "start": "2026-11-22T10:00:05Z",
                "end": "2026-11-22T10:00:05Z",
                "streaming": False,
                "isError": False,
                "waitForAnswer": False,
                "language": None,
                "metadata": turn_metadata(result, "hybrid", 8),
            },
        )
        await create_step(
            dl,
            {
                "id": "r1",
                "threadId": "t1",
                "parentId": None,
                "name": "Research",
                "type": "run",
                "output": "done",
                "showInput": False,
                "defaultOpen": False,
                "autoCollapse": False,
                "streaming": False,
                "metadata": {},
            },
        )

        thread = await dl.get_thread("t1")
        assert thread is not None
        assert thread["userIdentifier"] == "demo" and thread["name"] == "First question"
        assert {s["id"] for s in thread["steps"]} == {"s1", "s2", "r1"}

        conv = conversation_from_thread(thread)
        assert [m.role for m in conv.messages] == ["user", "assistant"]
        assert conv.messages[1].turn == result and conv.messages[1].strategy == "hybrid"

        # The history sidebar lists the user's threads.
        from chainlit.types import Pagination, ThreadFilter

        page = await dl.list_threads(Pagination(first=10), ThreadFilter(userId=user.id))
        assert [t["id"] for t in page.data] == ["t1"]
    finally:
        await dl.close()


def test_session_state_canary_never_reaches_thread_metadata() -> None:
    # Chainlit persists the user session with a JSON encoder that drops non-serializable
    # values; SessionState must stay non-serializable there (and excludes the canary anyway).
    from chainlit.session import clean_metadata

    session = SessionState()
    cleaned = clean_metadata({"research_session": session, "chat_settings": {"top_k": 8}})
    assert session.canary not in json.dumps(cleaned)
    assert cleaned["chat_settings"] == {"top_k": 8}


# --------------------------------------------------------------------------- auth


def _settings(user: str | None, pw: str | None) -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        demo_username=user,
        demo_password=SecretStr(pw) if pw is not None else None,
    )


@pytest.mark.parametrize(
    ("user", "pw", "ok"),
    [("demo", "s3cret", True), ("demo", "wrong", False), ("other", "s3cret", False)],
)
def test_check_demo_credentials(user: str, pw: str, ok: bool) -> None:
    assert check_demo_credentials(user, pw, _settings("demo", "s3cret")) is ok


@pytest.mark.parametrize(("user", "pw"), [(None, None), ("demo", None), ("demo", ""), ("", "x")])
def test_login_fails_without_configured_credentials(user: str | None, pw: str | None) -> None:
    assert check_demo_credentials(user or "", pw or "", _settings(user, pw)) is False


def test_threads_are_not_auto_tagged() -> None:
    # Regression: with auto_tag_thread the browser's chat profile is passed as a Python list
    # to SQLite, the thread upsert fails silently, and threads get no owner -> empty sidebar.
    import tomllib

    config = Path(__file__).resolve().parents[1] / ".chainlit" / "config.toml"
    features = tomllib.loads(config.read_text(encoding="utf-8"))["features"]
    assert features["auto_tag_thread"] is False


async def test_thread_gets_owner_and_name_without_tags(tmp_path: Path) -> None:
    dl = SQLAlchemyDataLayer(conninfo=sqlalchemy_url(init_db(tmp_path / "chat.db")))
    try:
        user = await dl.create_user(User(identifier="demo"))
        assert user is not None
        await dl.update_thread("t1")  # as create_step does first
        await dl.update_thread("t1", name="Question", user_id=user.id, tags=None)
        thread = await dl.get_thread("t1")
        assert thread is not None
        assert (thread["userIdentifier"], thread["name"]) == ("demo", "Question")
    finally:
        await dl.close()
