"""Shared plumbing for agent tools: dependencies, knowledge-base lookup, safe execution.

Every tool function returns a :class:`~research_assistant.models.ToolResult` and never
raises. Unexpected exceptions are logged (with traceback, server side only) and turned into a
short, user-safe message without stack traces, URLs with keys, or internal details.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from research_assistant.config import Settings, get_settings
from research_assistant.generation.prompts import sanitize_source_text
from research_assistant.ingest.arxiv_fetch import (
    ArxivClient,
    ArxivError,
    ArxivHTTPError,
    ArxivNetworkError,
    ArxivNotFoundError,
)
from research_assistant.ingest.index import BM25_FILE, BM25Store, ChunkIndex, ChunkIndexError
from research_assistant.models import ToolResult, ToolSource
from research_assistant.tools.semantic_scholar import (
    SemanticScholarClient,
    SemanticScholarError,
    SemanticScholarNotFoundError,
    SemanticScholarRateLimitError,
    SemanticScholarUnavailableError,
)

logger = logging.getLogger(__name__)

_VERSION_RE = re.compile(r"v\d+$")


def base_id(arxiv_id: str) -> str:
    """``2112.03570v2`` -> ``2112.03570`` (the knowledge base holds one version per paper)."""
    return _VERSION_RE.sub("", arxiv_id.strip())


def clean_text(text: str | None, max_chars: int | None = None) -> str:
    """Normalize untrusted external text (titles, abstracts) and optionally shorten it.

    Removes control characters and anything resembling our spotlighting delimiters; when
    ``max_chars`` is given, cuts at a word boundary and appends an ellipsis.
    """
    if not text:
        return ""
    out = sanitize_source_text(text)
    if max_chars is not None and len(out) > max_chars:
        cut = out[:max_chars].rsplit(" ", 1)[0] or out[:max_chars]
        out = cut.rstrip(" ,;:") + "…"
    return out


class ToolArgs(BaseModel):
    """Base for tool argument schemas: unknown fields are rejected, strings stripped."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# --------------------------------------------------------------------------- knowledge base


class KnowledgeBase(Protocol):
    """Answers "is this paper already indexed?" for versionless arXiv IDs."""

    def paper_ids(self) -> set[str]: ...

    def refresh(self) -> None: ...


class StaticKnowledgeBase:
    """Fixed set of IDs (tests, scripts)."""

    def __init__(self, ids: set[str] | None = None) -> None:
        self._ids = {base_id(i) for i in ids or set()}

    def paper_ids(self) -> set[str]:
        return set(self._ids)

    def refresh(self) -> None:
        pass

    def add(self, arxiv_id: str) -> None:
        self._ids.add(base_id(arxiv_id))


class IndexKnowledgeBase:
    """Paper IDs from the on-disk BM25 store (no Chroma / embedding model needed).

    Loaded lazily once and cached; call :meth:`refresh` after ingesting a paper.
    An unreadable or missing index counts as empty (logged), so lookups never fail a tool.
    """

    def __init__(self, settings: Settings) -> None:
        self._path = settings.paths.resolve().index_dir / BM25_FILE
        self._ids: set[str] | None = None
        self._lock = threading.Lock()

    def paper_ids(self) -> set[str]:
        with self._lock:
            if self._ids is None:
                try:
                    self._ids = BM25Store(self._path).paper_ids() if self._path.exists() else set()
                except ChunkIndexError as exc:
                    logger.warning("Knowledge-base lookup unavailable: %s", exc)
                    self._ids = set()
            return set(self._ids)

    def refresh(self) -> None:
        with self._lock:
            self._ids = None


# --------------------------------------------------------------------------- dependencies


@dataclass
class ToolDeps:
    """Everything the tools need. Clients are created lazily and shared, so one arXiv rate
    limiter covers every arXiv request made through these deps."""

    settings: Settings = field(default_factory=get_settings)
    kb: KnowledgeBase | None = None
    arxiv: ArxivClient | None = None
    semantic_scholar: SemanticScholarClient | None = None
    # Index for confirm_ingest; None -> ingest_paper opens the configured index.
    index: ChunkIndex | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def get_kb(self) -> KnowledgeBase:
        with self._lock:
            if self.kb is None:
                self.kb = IndexKnowledgeBase(self.settings)
            return self.kb

    def get_arxiv(self) -> ArxivClient:
        with self._lock:
            if self.arxiv is None:
                self.arxiv = ArxivClient(self.settings.arxiv)
            return self.arxiv

    def get_semantic_scholar(self) -> SemanticScholarClient:
        with self._lock:
            if self.semantic_scholar is None:
                self.semantic_scholar = SemanticScholarClient.from_settings(self.settings)
            return self.semantic_scholar

    def in_kb(self, arxiv_id: str | None) -> bool:
        return bool(arxiv_id) and base_id(arxiv_id or "") in self.get_kb().paper_ids()


_default_deps: ToolDeps | None = None
_default_lock = threading.Lock()


def default_deps() -> ToolDeps:
    """Process-wide shared deps (one arXiv client => one 3 s rate limit for all tools)."""
    global _default_deps
    with _default_lock:
        if _default_deps is None:
            _default_deps = ToolDeps()
        return _default_deps


# --------------------------------------------------------------------------- safe execution


def format_validation_error(exc: ValidationError) -> str:
    """Field-level messages without echoing input values."""
    parts: list[str] = []
    for err in exc.errors(include_url=False, include_input=False):
        loc = ".".join(str(x) for x in err.get("loc", ()) if x != "__root__")
        msg = str(err.get("msg", "invalid value")).removeprefix("Value error, ")
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "Invalid arguments: " + "; ".join(parts)


def friendly_error(exc: BaseException) -> tuple[str, ToolSource | None]:
    """Map an exception to (user-safe message, source override or None)."""
    if isinstance(exc, ValidationError):
        return format_validation_error(exc), "validation"
    if isinstance(exc, ArxivNetworkError):
        return "arXiv is not reachable right now. Please try again in a minute.", "arxiv"
    if isinstance(exc, ArxivNotFoundError):
        return "arXiv has no paper with that ID.", "arxiv"
    if isinstance(exc, ArxivHTTPError):
        return "arXiv rejected the request. Try a simpler query.", "arxiv"
    if isinstance(exc, ArxivError):
        return "arXiv returned an unexpected response. Please try again later.", "arxiv"
    if isinstance(exc, SemanticScholarNotFoundError):
        return "Semantic Scholar has no record of this paper.", "semantic_scholar"
    if isinstance(exc, SemanticScholarRateLimitError):
        return (
            "Semantic Scholar is rate-limiting requests. Please try again in a minute.",
            "semantic_scholar",
        )
    if isinstance(exc, SemanticScholarUnavailableError):
        return "Semantic Scholar is not reachable right now. Please try again later.", (
            "semantic_scholar"
        )
    if isinstance(exc, SemanticScholarError):
        return "Semantic Scholar returned an unexpected response.", "semantic_scholar"
    return "The tool failed unexpectedly. Please try again.", None


def run_safely(source: ToolSource, fn: Callable[[], Any]) -> ToolResult:
    """Call ``fn`` and wrap its return value in an ok ToolResult; never raises.

    ``fn`` may also return a ready ToolResult (e.g. for domain-level failures).
    """
    try:
        value = fn()
    except Exception as exc:
        message, src = friendly_error(exc)
        if src is None:
            logger.exception("Tool %s failed", source)
        else:
            logger.warning("Tool %s failed: %s: %s", source, type(exc).__name__, exc)
        return ToolResult(ok=False, error=message, source=src or source)
    if isinstance(value, ToolResult):
        return value
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return ToolResult(ok=True, data=value, source=source)


def fail(source: ToolSource, message: str) -> ToolResult:
    return ToolResult(ok=False, error=message, source=source)
