"""Create the SQLite chat-history database for the Chainlit UI (idempotent).

Usage: ``uv run python scripts/init_db.py [--db PATH]`` (default: paths.chat_db in settings).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from research_assistant.config import ConfigError, get_settings
from research_assistant.persistence import init_db


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, help="Database file (default: paths.chat_db).")
    args = parser.parse_args(argv)
    try:
        path = args.db or get_settings().paths.resolve().chat_db
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    try:
        init_db(path)
    except (OSError, sqlite3.Error) as exc:
        print(f"Could not create the database at {path}: {exc}", file=sys.stderr)
        return 1
    print(f"Chat history database ready: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
