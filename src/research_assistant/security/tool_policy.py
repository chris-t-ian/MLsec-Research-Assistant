"""Tool policy: allowlist, per-turn call limit, ingest confirmation and an audit log.

Rules enforced for every tool call the agent model requests (:meth:`ToolPolicy.authorize`):

1. The arguments must not contain the session canary (exfiltration through a tool).
2. The tool must be on the allowlist. Ingestion (``confirm_ingest`` / ``arxiv_ingest``) is
   never a model tool; the model can only *propose* an ingest via ``arxiv_ingest_request``.
3. At most ``max_calls`` tool calls per turn (hard cap 6); further calls are denied.
4. ``arxiv_ingest_request`` only accepts IDs with trusted provenance: the ID must appear in
   the user's own messages or in arXiv / Semantic Scholar tool results of this session, never
   (only) in retrieved paper text.

Actual ingestion happens in :func:`confirm_ingest_with_policy`, which runs ``confirm_ingest``
only for a pending request of this session and only when the user explicitly confirmed it.

Every decision (allowed or denied) and every result is written to the :class:`AuditLog`.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from research_assistant.events import EventSink
from research_assistant.models import IngestRequest, ToolResult
from research_assistant.security.canary import value_contains_canary
from research_assistant.tools.arxiv_ids import ArxivIdError, parse_arxiv_id
from research_assistant.tools.arxiv_tools import confirm_ingest
from research_assistant.tools.base import ToolDeps

logger = logging.getLogger(__name__)

HARD_MAX_TOOL_CALLS = 6

ALLOWED_TOOLS: frozenset[str] = frozenset(
    {
        "search_knowledge_base",
        "arxiv_search",
        "arxiv_ingest_request",
        "citation_graph",
        "dp_accountant",
        "bibtex_export",
    }
)
# Side-effect tools that only the UI may trigger, after explicit user confirmation.
CONFIRMATION_ONLY_TOOLS: frozenset[str] = frozenset({"confirm_ingest", "arxiv_ingest"})

MAX_AUDIT_VALUE_CHARS = 500

Decision = Literal["allowed", "denied"]


class PolicyDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    allowed: bool
    reason: str | None = Field(default=None, description="User-safe reason when denied.")
    kind: Literal["ok", "canary", "not_allowed", "limit", "provenance"] = "ok"


class ToolCallRecord(BaseModel):
    """One audit-log entry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: str
    session_id: str
    turn_id: str
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    decision: Decision
    reason: str | None = None
    ok: bool | None = None
    error: str | None = None
    latency_ms: float | None = None


def _truncate(value: Any) -> Any:
    if isinstance(value, str):
        return value if len(value) <= MAX_AUDIT_VALUE_CHARS else value[:MAX_AUDIT_VALUE_CHARS] + "…"
    if isinstance(value, Mapping):
        return {str(k): _truncate(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_truncate(v) for v in value[:50]]
    if value is None or isinstance(value, bool | int | float):
        return value
    return _truncate(str(value))


class AuditLog:
    """In-memory audit trail, mirrored to the log and (optionally) a JSONL file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.records: list[ToolCallRecord] = []
        self._lock = threading.Lock()
        self._file_failed = False

    def record(
        self,
        *,
        session_id: str,
        turn_id: str,
        tool: str,
        args: Mapping[str, Any] | None,
        decision: Decision,
        reason: str | None = None,
        result: ToolResult | None = None,
        latency_ms: float | None = None,
    ) -> ToolCallRecord:
        rec = ToolCallRecord(
            timestamp=datetime.now(UTC).isoformat(timespec="milliseconds"),
            session_id=session_id,
            turn_id=turn_id,
            tool=tool,
            args=_truncate(dict(args or {})),
            decision=decision,
            reason=reason,
            ok=None if result is None else result.ok,
            error=None if result is None else result.error,
            latency_ms=None if latency_ms is None else round(latency_ms, 1),
        )
        with self._lock:
            self.records.append(rec)
            self._write(rec)
        log = logger.info if decision == "allowed" else logger.warning
        log(
            "tool-audit session=%s turn=%s tool=%s decision=%s ok=%s reason=%s",
            session_id,
            turn_id,
            tool,
            decision,
            rec.ok,
            reason,
        )
        return rec

    def _write(self, rec: ToolCallRecord) -> None:
        if self.path is None or self._file_failed:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(rec.model_dump_json() + "\n")
        except OSError as exc:
            self._file_failed = True
            logger.warning("Cannot write tool audit log %s: %s", self.path, exc)


# --------------------------------------------------------------------------- provenance

_NEW_ID = re.compile(r"(?<![\w.])\d{4}\.\d{4,5}(?:v\d+)?(?![\w])")
_OLD_ID = re.compile(r"(?<![\w/])[a-z]+(?:-[a-z]+)?(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?(?![\w])")


def extract_arxiv_ids(text: str) -> set[str]:
    """Versionless arXiv IDs mentioned in ``text`` (only syntactically valid ones)."""
    found: set[str] = set()
    for m in [*_NEW_ID.finditer(text), *_OLD_ID.finditer(text)]:
        try:
            found.add(parse_arxiv_id(m.group(0)).base)
        except ArxivIdError:
            continue
    return found


def ids_from_tool_result(tool: str, result: ToolResult) -> set[str]:
    """arXiv IDs returned by metadata tools (arXiv search, Semantic Scholar citation graph)."""
    if not result.ok or not isinstance(result.data, dict):
        return set()
    if tool == "arxiv_search":
        items = result.data.get("results")
    elif tool == "citation_graph":
        items = result.data.get("papers")
    else:
        return set()
    out: set[str] = set()
    for item in items if isinstance(items, list) else []:
        raw = item.get("arxiv_id") if isinstance(item, dict) else None
        if isinstance(raw, str):
            try:
                out.add(parse_arxiv_id(raw).base)
            except ArxivIdError:
                continue
    return out


# --------------------------------------------------------------------------- policy


class ToolPolicy:
    def __init__(
        self,
        max_calls: int = HARD_MAX_TOOL_CALLS,
        audit: AuditLog | None = None,
        allowlist: Iterable[str] = ALLOWED_TOOLS,
    ) -> None:
        self.max_calls = max(1, min(max_calls, HARD_MAX_TOOL_CALLS))
        self.audit = audit or AuditLog()
        self.allowlist = frozenset(allowlist) - CONFIRMATION_ONLY_TOOLS

    def authorize(
        self,
        name: str,
        args: Mapping[str, Any],
        *,
        calls_made: int,
        canary: str,
        trusted_ids: set[str],
    ) -> PolicyDecision:
        """Decide whether a model-requested tool call may run (does not record it)."""
        if value_contains_canary(dict(args), canary):
            return PolicyDecision(
                allowed=False, kind="canary", reason="Tool call blocked by the security policy."
            )
        if name in CONFIRMATION_ONLY_TOOLS:
            return PolicyDecision(
                allowed=False,
                kind="not_allowed",
                reason=(
                    "Papers can only be added after the user confirms an ingest request; "
                    "use arxiv_ingest_request to propose one."
                ),
            )
        if name not in self.allowlist:
            return PolicyDecision(
                allowed=False, kind="not_allowed", reason=f"Unknown or disallowed tool {name!r}."
            )
        if calls_made >= self.max_calls:
            return PolicyDecision(
                allowed=False,
                kind="limit",
                reason=f"Tool-call limit reached ({self.max_calls} per turn).",
            )
        if name == "arxiv_ingest_request":
            raw = args.get("arxiv_id")
            try:
                base = parse_arxiv_id(str(raw or "")).base
            except ArxivIdError:
                return PolicyDecision(allowed=True)  # the tool reports the invalid ID itself
            if base not in trusted_ids:
                return PolicyDecision(
                    allowed=False,
                    kind="provenance",
                    reason=(
                        f"arXiv:{base} was not given by the user or found by arxiv_search / "
                        "citation_graph in this session, so it cannot be proposed for ingest."
                    ),
                )
        return PolicyDecision(allowed=True)


def confirm_ingest_with_policy(
    arxiv_id: str,
    pending: dict[str, IngestRequest],
    user_confirmed: bool,
    sink: EventSink,
    deps: ToolDeps,
    audit: AuditLog,
    session_id: str,
    turn_id: str = "confirm",
) -> ToolResult:
    """Run ``confirm_ingest`` for a pending request after explicit user confirmation.

    Removes the request from ``pending`` when it was executed or declined.
    """
    args = {"arxiv_id": arxiv_id, "user_confirmed": user_confirmed}
    try:
        base = parse_arxiv_id(arxiv_id).base
    except ArxivIdError as exc:
        result = ToolResult(ok=False, error=str(exc), source="validation")
        audit.record(
            session_id=session_id,
            turn_id=turn_id,
            tool="confirm_ingest",
            args=args,
            decision="denied",
            reason="invalid arXiv ID",
            result=result,
        )
        return result
    request = pending.get(base)
    if request is None:
        reason = f"There is no pending ingest request for arXiv:{base} in this session."
    elif not user_confirmed:
        pending.pop(base, None)
        reason = f"Adding arXiv:{base} was not confirmed, so nothing was ingested."
    else:
        reason = None
    if request is None or reason is not None:
        result = ToolResult(ok=False, error=reason, source="knowledge_base")
        audit.record(
            session_id=session_id,
            turn_id=turn_id,
            tool="confirm_ingest",
            args=args,
            decision="denied",
            reason=reason,
            result=result,
        )
        return result

    start = datetime.now(UTC)
    result = confirm_ingest(request, sink, deps)
    latency = (datetime.now(UTC) - start).total_seconds() * 1000
    pending.pop(base, None)
    audit.record(
        session_id=session_id,
        turn_id=turn_id,
        tool="confirm_ingest",
        args=args,
        decision="allowed",
        reason="explicit user confirmation",
        result=result,
        latency_ms=latency,
    )
    sink.tool_result("confirm_ingest", {"arxiv_id": base}, result)
    return result


__all__ = [
    "ALLOWED_TOOLS",
    "CONFIRMATION_ONLY_TOOLS",
    "HARD_MAX_TOOL_CALLS",
    "AuditLog",
    "PolicyDecision",
    "ToolCallRecord",
    "ToolPolicy",
    "confirm_ingest_with_policy",
    "extract_arxiv_ids",
    "ids_from_tool_result",
]
