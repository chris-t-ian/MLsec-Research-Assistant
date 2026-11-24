"""Pydantic models for data crossing module boundaries."""

from __future__ import annotations

from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


Topic = Literal[
    "membership_inference",
    "differential_privacy",
    "model_inversion",
    "data_extraction",
    "prompt_injection",
    "other",
]


class PaperListEntry(_Frozen):
    """One entry of data/papers.yaml."""

    arxiv_id: str = Field(min_length=1)
    topic: Topic


class Paper(_Frozen):
    arxiv_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    authors: list[str] = Field(default_factory=list)
    year: int | None = Field(default=None, ge=1900, le=2100)
    abstract: str | None = None
    url: str | None = None
    categories: list[str] = Field(default_factory=list)
    version: int | None = Field(default=None, ge=1)
    topic: Topic | None = None
    latex_available: bool = False


class Chunk(_Frozen):
    id: str = Field(min_length=1)
    arxiv_id: str = Field(min_length=1)
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = Field(default=None, ge=1900, le=2100)
    section: str | None = None
    text: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    # Security / quality flags set during ingest, e.g. "possible_injection".
    flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_span(self) -> Self:
        if self.char_end < self.char_start:
            raise ValueError("char_end must be >= char_start")
        return self


class Citation(_Frozen):
    """One source cited in an answer; ``number`` is the ``[n]`` marker used in the answer."""

    number: int = Field(ge=1)
    chunk_id: str = Field(min_length=1)
    arxiv_id: str = Field(min_length=1)
    label: str = Field(description="Short author-year label, e.g. 'Shokri et al. 2017'.")
    title: str
    section: str | None = None
    excerpt: str = ""
    url: str


class ScoredChunk(_Frozen):
    """One ranked hit. ``score`` is the score of the stage that produced the ranking."""

    chunk: Chunk
    score: float
    rank: int = Field(ge=1)
    source: str = Field(
        description="Stage that produced this ranking, e.g. 'bm25', 'dense', 'rrf', 'rerank'."
    )
    # Scores from every stage the chunk passed, e.g. {"dense": 0.71, "rrf": 0.032}.
    stage_scores: dict[str, float] = Field(default_factory=dict)


# --------------------------------------------------------------------------- retrieval

SectionType = Literal["method", "results", "related_work", "any"]
QueryTranslation = Literal["none", "multi_query", "hyde", "decompose"]
RetrieverKind = Literal["dense", "bm25", "hybrid"]


class QueryFilters(_Frozen):
    """Validated metadata filters (only values known to the index survive validation)."""

    year_min: int | None = Field(default=None, ge=1900, le=2100)
    year_max: int | None = Field(default=None, ge=1900, le=2100)
    topic: Topic | None = None
    authors: list[str] = Field(default_factory=list)
    arxiv_ids: list[str] = Field(default_factory=list)
    section_type: SectionType = "any"

    def is_empty(self) -> bool:
        return (
            self.year_min is None
            and self.year_max is None
            and self.topic is None
            and not self.authors
            and not self.arxiv_ids
            and self.section_type == "any"
        )


class TokenUsage(_Frozen):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    cache_read_tokens: int = Field(
        default=0, ge=0, description="Part of input_tokens read from the prompt cache."
    )
    cache_write_tokens: int = Field(
        default=0, ge=0, description="Part of input_tokens written to the prompt cache."
    )

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


class CandidateScore(_Frozen):
    """A candidate as seen by one stage (ids and scores only, no text)."""

    chunk_id: str
    arxiv_id: str
    rank: int = Field(ge=1)
    score: float
    list_name: str = Field(description="Ranked list the candidate came from, e.g. 'dense:q0'.")


class StageTrace(_Frozen):
    stage: str
    latency_ms: float = Field(ge=0)
    candidates: list[CandidateScore] = Field(default_factory=list)
    token_usage: TokenUsage = TokenUsage()
    notes: list[str] = Field(default_factory=list)


class RetrievalTrace(_Frozen):
    strategy: str
    question: str
    semantic_query: str = Field(description="Query after self-query cleaning (or the question).")
    query_translation: QueryTranslation | None = None
    retriever: RetrieverKind | None = None
    rerank: bool = False
    rewrites: list[str] = Field(
        default_factory=list, description="Paraphrases, HyDE passages or sub-questions."
    )
    filters: QueryFilters | None = None
    stages: list[StageTrace] = Field(default_factory=list)
    token_usage: TokenUsage = TokenUsage()
    total_latency_ms: float = Field(default=0.0, ge=0)


class RetrievalResult(_Frozen):
    """Outcome of one retrieval run: the selected chunks plus a full trace."""

    question: str
    strategy: str
    chunks: list[ScoredChunk] = Field(default_factory=list)
    trace: RetrievalTrace


# --------------------------------------------------------------------------- generation


class Source(_Frozen):
    """A retrieved chunk as numbered source ``[number]`` in the generation prompt."""

    number: int = Field(ge=1)
    chunk: Chunk
    label: str = Field(description="Short author-year label, unique within one prompt.")


class CitationError(_Frozen):
    """A marker in the model output that refers to no source (removed from the answer)."""

    marker: str
    number: int
    reason: str


class ToolEvidence(_Frozen):
    """A tool result passed to the answer generator as spotlighted block ``[T<number>]``."""

    number: int = Field(ge=1)
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: ToolResult


class ToolCitation(_Frozen):
    """A tool result cited in an answer with the marker ``[T<number>]``."""

    number: int = Field(ge=1)
    name: str
    ok: bool


# --------------------------------------------------------------------------- verification

JudgeLabel = Literal["supported", "partially_supported", "unsupported"]
ClaimStatus = Literal["supported", "partially_supported", "unsupported", "uncited"]
VerificationPolicy = Literal["off", "flag", "remove", "regenerate"]


class ClaimCheck(_Frozen):
    """Verdict on one sentence of an answer (see research_assistant.verification).

    ``start``/``end`` locate the sentence in :attr:`Answer.raw_text`. ``status`` is the final
    label after the evidence check; ``judge_label`` is what the judge said before it.
    """

    index: int = Field(ge=1, description="Claim number, shown as 'C<index>'.")
    sentence: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    cited: list[int] = Field(default_factory=list, description="Source numbers [n] cited.")
    status: ClaimStatus
    judge_label: JudgeLabel | None = None
    evidence_span: str = ""
    evidence_source: int | None = Field(default=None, description="Source number of the span.")
    evidence_chunk_id: str | None = None
    # Chunk text around the span (for highlighting in the UI), whitespace-normalized.
    evidence_before: str = ""
    evidence_after: str = ""
    rationale: str = ""
    note: str = Field(default="", description="e.g. 'fabricated evidence'.")


class VerificationInfo(_Frozen):
    """How the claims of an answer were verified and what the policy did."""

    policy: VerificationPolicy
    status: Literal["ok", "error", "skipped"]
    error: str | None = None
    judge_model: str | None = None
    n_judge_calls: int = Field(default=0, ge=0)
    regenerated: bool = False
    removed: int = Field(default=0, ge=0, description="Sentences removed by the policy.")
    initial_supported_rate: float | None = Field(
        default=None, description="Supported rate before removal / regeneration."
    )
    token_usage: TokenUsage = TokenUsage()
    latency_ms: float = Field(default=0.0, ge=0)


class Answer(_Frozen):
    question: str
    text: str = Field(description="Answer with markers rendered as '(Shokri et al. 2017 [1])'.")
    raw_text: str = Field(
        default="", description="Answer with validated numeric markers only, e.g. '... [1][3].'"
    )
    citations: list[Citation] = Field(default_factory=list)
    citation_errors: list[CitationError] = Field(default_factory=list)
    tool_citations: list[ToolCitation] = Field(default_factory=list)
    strategy: str | None = None
    retrieval: RetrievalTrace | None = None
    model: str | None = None
    token_usage: TokenUsage = Field(
        default=TokenUsage(), description="Retrieval + generation tokens."
    )
    generation_token_usage: TokenUsage = TokenUsage()
    latency_ms: float = Field(default=0.0, ge=0)
    generation_latency_ms: float = Field(default=0.0, ge=0)
    claim_checks: list[ClaimCheck] = Field(default_factory=list)
    supported_rate: float | None = Field(
        default=None,
        description="supported / (cited claims + uncited factual sentences); None if none.",
    )
    verification: VerificationInfo | None = None


# --------------------------------------------------------------------------- ingest

Severity = Literal["low", "medium", "high"]

FindingType = Literal[
    "hidden_color",  # LaTeX \color{white} / \textcolor{white}{...}
    "phantom",  # LaTeX \phantom / \hphantom / \vphantom
    "tiny_font",  # LaTeX \fontsize below threshold
    "zero_scale",  # LaTeX near-zero \scalebox / \resizebox
    "hidden_macro",  # call of a user macro whose definition hides its content
    "comment_instruction",  # AI-directed text inside a LaTeX comment
    "ai_directed_text",  # visible text addressed to AI systems
    "pdf_hidden_color",  # PDF text colored like its background
    "pdf_tiny_font",  # PDF text below the minimum font size
]


class Finding(_Frozen):
    """Hidden or suspicious content found in a paper source before text conversion."""

    type: FindingType
    snippet: str = Field(description="Short excerpt of the raw source (truncated).")
    location: str = Field(description="Human-readable position, e.g. 'main.tex:42' or 'page 3'.")
    severity: Severity
    # Plain-text form of the suspicious content, used to locate it in chunks.
    payload: str = ""
    # Character offset in the scanned (flattened) LaTeX source; None for PDF findings.
    offset: int | None = Field(default=None, ge=0)
    # Index of the parsed section containing the finding, if known.
    section_idx: int | None = Field(default=None, ge=0)


SectionKind = Literal["abstract", "front_matter", "section", "subsection", "appendix", "body"]


class ParsedSection(_Frozen):
    title: str
    kind: SectionKind
    text: str


class ParsedDocument(_Frozen):
    """A paper converted to plain text, split into ordered sections."""

    arxiv_id: str = Field(min_length=1)
    source: Literal["latex", "pdf"]
    sections: list[ParsedSection] = Field(default_factory=list)
    main_file: str | None = None

    SECTION_SEPARATOR: ClassVar[str] = "\n\n"

    def section_offsets(self) -> list[int]:
        """Start offset of each section in :meth:`full_text`."""
        offsets: list[int] = []
        pos = 0
        for s in self.sections:
            offsets.append(pos)
            pos += len(s.text) + len(self.SECTION_SEPARATOR)
        return offsets

    def full_text(self) -> str:
        """All section texts joined; chunk char offsets refer to this string."""
        return self.SECTION_SEPARATOR.join(s.text for s in self.sections)


# --------------------------------------------------------------------------- tools

ToolSource = Literal["arxiv", "semantic_scholar", "dp_accounting", "knowledge_base", "validation"]


class ToolResult(_Frozen):
    """Uniform result of every agent tool. Tools never raise; failures set ``ok=False``."""

    ok: bool
    data: Any = Field(default=None, description="JSON-serializable payload when ok.")
    error: str | None = Field(default=None, description="User-safe error message when not ok.")
    source: ToolSource

    @model_validator(mode="after")
    def _check_error(self) -> Self:
        if self.ok == (self.error is not None):
            raise ValueError("error must be set exactly when ok is False")
        return self


IngestStatus = Literal["pending_confirmation", "completed", "failed"]


class IngestRequest(_Frozen):
    """A proposal to add a paper to the knowledge base; nothing happens until confirmed."""

    arxiv_id: str = Field(min_length=1, description="Versionless arXiv ID, e.g. '2112.03570'.")
    requested_id: str = Field(description="The ID as given, e.g. '2112.03570v2'.")
    status: IngestStatus = "pending_confirmation"
    already_in_kb: bool = False
    message: str = ""


# --------------------------------------------------------------------------- security

SecurityEventKind = Literal[
    "input_rejected",  # input validation failed
    "injection_warn",  # possible prompt injection, turn continued
    "injection_block",  # prompt injection, turn blocked
    "canary_leak",  # the session canary appeared in model output or tool arguments
    "tool_denied",  # the tool policy refused a tool call
    "tool_limit",  # the per-turn tool-call limit was reached
    "output_filtered",  # external links/images to non-allowlisted hosts removed from output
]


class SecurityEvent(_Frozen):
    """A security-relevant decision during one turn (shown in traces, logged)."""

    kind: SecurityEventKind
    detail: str = Field(description="Short, user-safe description (never contains secrets).")
    score: float | None = None
