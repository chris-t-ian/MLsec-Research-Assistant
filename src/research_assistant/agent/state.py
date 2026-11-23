"""Session and turn models of the agent (data crossing the core/UI boundary)."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from research_assistant.models import (
    Citation,
    CitationError,
    ClaimCheck,
    IngestRequest,
    RetrievalTrace,
    SecurityEvent,
    TokenUsage,
    ToolCitation,
    ToolResult,
    VerificationInfo,
    VerificationPolicy,
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
    top_k: int | None = Field(
        default=None, ge=1, le=50, description="Chunks per search; None uses the strategy's."
    )
    model: str | None = Field(
        default=None,
        min_length=1,
        description="Answering model (one of settings.models.choices()); None uses models.main.",
    )
    verification: VerificationPolicy | None = Field(
        default=None, description="Claim verification policy; None uses verification.policy."
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


class SourceRef(BaseModel):
    """A chunk passed to the answer generator as source ``[number]`` (cited or not)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    number: int = Field(ge=1)
    chunk_id: str = Field(min_length=1)
    arxiv_id: str = Field(min_length=1)
    label: str
    title: str = ""
    section: str | None = None


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
    # RAG trace data (defaults keep turns stored before these fields existed loadable).
    retrievals: list[RetrievalTrace] = Field(
        default_factory=list, description="One trace per knowledge-base search, in order."
    )
    sources: list[SourceRef] = Field(
        default_factory=list, description="Every chunk given to the answer generator."
    )
    paper_labels: dict[str, str] = Field(
        default_factory=dict, description="arXiv ID -> author-year label for traced papers."
    )
    generation_latency_ms: float = Field(default=0.0, ge=0)
    # Claim verification (defaults keep older stored turns loadable).
    claim_checks: list[ClaimCheck] = Field(default_factory=list)
    supported_rate: float | None = None
    verification: VerificationInfo | None = None


__all__ = ["SessionState", "SourceRef", "ToolCallOutcome", "TraceStep", "TurnResult"]
