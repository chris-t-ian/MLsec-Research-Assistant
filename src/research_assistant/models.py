"""Pydantic models for data crossing module boundaries."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Paper(_Frozen):
    arxiv_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    authors: list[str] = Field(default_factory=list)
    year: int | None = Field(default=None, ge=1900, le=2100)
    abstract: str | None = None
    url: str | None = None


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
