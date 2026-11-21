"""Session and turn models of the agent (data crossing the core/UI boundary)."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from research_assistant.models import (
    Citation,
    CitationError,
    IngestRequest,
    SecurityEvent,
    TokenUsage,
    ToolCitation,
    ToolResult,
)
from research_assistant.security.canary import new_canary


class SessionState(BaseModel):
    """Per-user-session state, owned by the UI and passed to every :func:`run_turn`.

    Conversation memory itself lives in the agent's LangGraph checkpointer under
    ``session_id``. The canary is excluded from dumps and repr so it never reaches logs or
    the UI.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    session_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1)
    strategy: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9_]+$",
        description="Retrieval strategy; None uses agent.default_strategy from settings.",
    )
    canary: str = Field(default_factory=new_canary, exclude=True, repr=False)
    pending_ingests: dict[str, IngestRequest] = Field(
        default_factory=dict, description="Ingest requests awaiting user confirmation."
    )
    trusted_ids: set[str] = Field(
        default_factory=set,
        description="arXiv IDs from user messages or arXiv/Semantic Scholar results.",
    )
    turns: int = Field(default=0, ge=0)


class TraceStep(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    step_id: str
    node: str
    latency_ms: float = Field(ge=0)
    output: dict[str, Any] = Field(default_factory=dict)


class ToolCallOutcome(BaseModel):
    """One tool call requested by the agent model (allowed or denied) and its result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    call_id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    allowed: bool
    result: ToolResult


class TurnResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str
    turn_id: str
    question: str
    answer: str = Field(description="Final, user-facing text (citations rendered).")
    raw_answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    tool_citations: list[ToolCitation] = Field(default_factory=list)
    citation_errors: list[CitationError] = Field(default_factory=list)
    tool_results: list[ToolCallOutcome] = Field(default_factory=list)
    trace: list[TraceStep] = Field(default_factory=list)
    pending_ingests: list[IngestRequest] = Field(default_factory=list)
    security_events: list[SecurityEvent] = Field(default_factory=list)
    blocked: bool = False
    model: str | None = None
    token_usage: TokenUsage = TokenUsage()
    latency_ms: float = Field(default=0.0, ge=0)


__all__ = ["SessionState", "ToolCallOutcome", "TraceStep", "TurnResult"]
