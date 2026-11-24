"""Startup checks: fail fast with a readable message instead of a stack trace mid-chat.

``scripts/check_setup.py`` (``just check``, run by ``just app``) prints every problem and
exits non-zero; the Chainlit app runs the same checks when it is loaded.

Checks, in order:

1. ``config/settings.yaml`` and the environment parse into :class:`Settings`
   (field-level messages, never the offending values).
2. Secrets the app needs are set: ``OPENROUTER_API_KEY`` and, for the web UI login,
   ``CHAINLIT_AUTH_SECRET``, ``DEMO_USERNAME`` and ``DEMO_PASSWORD`` (presence only).
3. ``config/strategies.yaml`` loads and contains ``agent.default_strategy``.
4. The search index exists and is not empty; otherwise the message names the ingest script.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from research_assistant.config import PROJECT_ROOT, ConfigError, Settings, get_settings
from research_assistant.ingest.index import BM25_FILE, CHROMA_DIR, BM25Store, ChunkIndexError

logger = logging.getLogger(__name__)

Target = Literal["app", "cli"]

INGEST_HINT = (
    "Build it with `uv run just corpus` (download the papers in data/papers.yaml) and then "
    "`uv run just ingest` (or: `uv run python scripts/build_corpus.py` followed by "
    "`uv run python scripts/ingest.py`)."
)


class StartupProblem(BaseModel):
    check: Literal["config", "secrets", "strategies", "index"]
    message: str


class StartupError(RuntimeError):
    """The app cannot start; ``str(exc)`` is a readable, secret-free explanation."""

    def __init__(self, problems: list[StartupProblem]) -> None:
        self.problems = problems
        super().__init__(format_problems(problems))


def format_problems(problems: list[StartupProblem]) -> str:
    lines = ["The research assistant cannot start:"]
    lines += [f"  - [{p.check}] {p.message}" for p in problems]
    return "\n".join(lines)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def check_index(settings: Settings) -> StartupProblem | None:
    """None if the index at ``paths.index_dir`` exists and holds at least one chunk."""
    index_dir = settings.paths.resolve().index_dir
    bm25 = index_dir / BM25_FILE
    chroma = index_dir / CHROMA_DIR
    if not bm25.is_file() or not chroma.is_dir():
        return StartupProblem(
            check="index", message=f"No search index found in {_rel(index_dir)}. {INGEST_HINT}"
        )
    try:
        n = len(BM25Store(bm25))
    except ChunkIndexError as exc:
        return StartupProblem(
            check="index",
            message=f"The search index in {_rel(index_dir)} is unreadable ({exc}). "
            "Rebuild it with `uv run python scripts/ingest.py --rebuild`.",
        )
    if n == 0:
        return StartupProblem(
            check="index", message=f"The search index in {_rel(index_dir)} is empty. {INGEST_HINT}"
        )
    return None


def check_secrets(settings: Settings, target: Target) -> list[StartupProblem]:
    def missing(value: object) -> bool:
        if value is None:
            return True
        get = getattr(value, "get_secret_value", None)
        return not (get() if callable(get) else str(value)).strip()

    names = {"OPENROUTER_API_KEY": settings.openrouter_api_key}
    if target == "app":
        names |= {
            "CHAINLIT_AUTH_SECRET": settings.chainlit_auth_secret,
            "DEMO_USERNAME": settings.demo_username,
            "DEMO_PASSWORD": settings.demo_password,
        }
    absent = [k for k, v in names.items() if missing(v)]
    if not absent:
        return []
    return [
        StartupProblem(
            check="secrets",
            message=f"{', '.join(absent)} not set. Copy .env.example to .env and fill it in "
            "(see the README quickstart).",
        )
    ]


def check_strategies(settings: Settings) -> StartupProblem | None:
    from research_assistant.retrieval.strategies import (  # heavy import, keep lazy
        StrategyConfigError,
        get_strategy,
    )

    try:
        get_strategy(settings.agent.default_strategy, settings)
    except StrategyConfigError as exc:
        return StartupProblem(
            check="strategies",
            message=f"agent.default_strategy in config/settings.yaml is not usable: {exc}",
        )
    return None


def run_checks(target: Target = "app", settings: Settings | None = None) -> list[StartupProblem]:
    """All startup problems (empty list: ready). Never raises for configuration errors."""
    if settings is None:
        try:
            settings = get_settings()
        except ConfigError as exc:
            return [StartupProblem(check="config", message=str(exc))]
    problems = check_secrets(settings, target)
    if (p := check_strategies(settings)) is not None:
        problems.append(p)
    if (p := check_index(settings)) is not None:
        problems.append(p)
    return problems


def ensure_ready(target: Target = "app", settings: Settings | None = None) -> Settings:
    """Settings if every check passes, else raise :class:`StartupError`."""
    problems = run_checks(target, settings)
    if problems:
        raise StartupError(problems)
    return settings or get_settings()


__all__ = [
    "INGEST_HINT",
    "StartupError",
    "StartupProblem",
    "check_index",
    "check_secrets",
    "check_strategies",
    "ensure_ready",
    "format_problems",
    "run_checks",
]
