"""Pydantic models for data crossing module boundaries."""

from __future__ import annotations

from typing import ClassVar, Literal, Self

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
    chunk_id: str = Field(min_length=1)
    arxiv_id: str = Field(min_length=1)
    title: str
    section: str | None = None
    quote: str | None = None


class RetrievalResult(_Frozen):
    chunk: Chunk
    score: float
    rank: int = Field(ge=1)
    source: str = Field(description="Retriever that produced this hit, e.g. 'bm25', 'dense'.")


class Answer(_Frozen):
    question: str
    text: str
    citations: list[Citation] = Field(default_factory=list)
    model: str | None = None


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
