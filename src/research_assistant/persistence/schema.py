"""SQLite schema for Chainlit's ``SQLAlchemyDataLayer`` (Chainlit 2.12).

Chainlit only documents a PostgreSQL schema. The columns below are the ones the data layer's
SQL reads and writes (``chainlit/data/sql_alchemy.py`` and the step/element/feedback dicts).
JSON values (metadata, props, generation, tags) are stored as TEXT; the data layer encodes
them itself. Every statement is idempotent, so :func:`init_db` is safe to run repeatedly.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    "id" TEXT PRIMARY KEY,
    "identifier" TEXT NOT NULL UNIQUE,
    "metadata" TEXT NOT NULL DEFAULT '{}',
    "createdAt" TEXT
);

CREATE TABLE IF NOT EXISTS threads (
    "id" TEXT PRIMARY KEY,
    "createdAt" TEXT,
    "name" TEXT,
    "userId" TEXT REFERENCES users ("id") ON DELETE CASCADE,
    "userIdentifier" TEXT,
    "tags" TEXT,
    "metadata" TEXT
);
CREATE INDEX IF NOT EXISTS idx_threads_user ON threads ("userId");

CREATE TABLE IF NOT EXISTS steps (
    "id" TEXT PRIMARY KEY,
    "name" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "threadId" TEXT NOT NULL REFERENCES threads ("id") ON DELETE CASCADE,
    "parentId" TEXT,
    "streaming" BOOLEAN NOT NULL DEFAULT 0,
    "waitForAnswer" BOOLEAN,
    "isError" BOOLEAN,
    "metadata" TEXT,
    "tags" TEXT,
    "input" TEXT,
    "output" TEXT,
    "createdAt" TEXT,
    "command" TEXT,
    "modes" TEXT,
    "start" TEXT,
    "end" TEXT,
    "generation" TEXT,
    "showInput" TEXT,
    "language" TEXT,
    "indent" INTEGER,
    "defaultOpen" BOOLEAN,
    "autoCollapse" BOOLEAN,
    "icon" TEXT
);
CREATE INDEX IF NOT EXISTS idx_steps_thread ON steps ("threadId");

CREATE TABLE IF NOT EXISTS elements (
    "id" TEXT PRIMARY KEY,
    "threadId" TEXT REFERENCES threads ("id") ON DELETE CASCADE,
    "type" TEXT,
    "url" TEXT,
    "chainlitKey" TEXT,
    "name" TEXT NOT NULL,
    "display" TEXT,
    "objectKey" TEXT,
    "size" TEXT,
    "page" INTEGER,
    "language" TEXT,
    "forId" TEXT,
    "mime" TEXT,
    "props" TEXT,
    "autoPlay" BOOLEAN,
    "playerConfig" TEXT
);

CREATE TABLE IF NOT EXISTS feedbacks (
    "id" TEXT PRIMARY KEY,
    "forId" TEXT NOT NULL,
    "threadId" TEXT NOT NULL REFERENCES threads ("id") ON DELETE CASCADE,
    "value" INTEGER NOT NULL,
    "comment" TEXT
);
"""


def init_db(path: Path) -> Path:
    """Create the database file (and parent directory) and all tables if missing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL)
    conn.close()
    logger.info("Chat history schema ready at %s", path)
    return path


def sqlalchemy_url(path: Path) -> str:
    """Async SQLAlchemy URL (aiosqlite driver) for the database at ``path``."""
    return f"sqlite+aiosqlite:///{Path(path).resolve()}"
