"""Records of one evaluation run (one line of results.jsonl per strategy x question)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from research_assistant.abtest.gold import QuestionType, Split
from research_assistant.models import TokenUsage


class RetrievedChunk(BaseModel):
    chunk_id: str
    arxiv_id: str
    rank: int = Field(ge=1)


class SearchRecord(BaseModel):
    """The final chunk selection of one ``search_knowledge_base`` call."""

    query: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    latency_ms: float = 0.0


class FactJudgment(BaseModel):
    fact: str
    covered: bool
    rationale: str = ""


class RefusalJudgment(BaseModel):
    correct: bool = Field(description="The answer declines / corrects the premise as expected.")
    rationale: str = ""


class JudgeResult(BaseModel):
    model: str
    facts: list[FactJudgment] = Field(default_factory=list)
    refusal: RefusalJudgment | None = None
    token_usage: TokenUsage = TokenUsage()
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class EvalRecord(BaseModel):
    """Everything the metrics need about one (strategy, question) turn."""

    run_id: str
    strategy: str
    question_id: str
    question_type: QuestionType
    split: Split | None
    question: str
    answer: str
    raw_answer: str = ""
    cited_arxiv_ids: list[str] = Field(
        default_factory=list, description="arXiv ID per citation, in citation-number order."
    )
    searches: list[SearchRecord] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
    blocked: bool = False
    model: str | None = None
    token_usage: TokenUsage = TokenUsage()
    latency_ms: float = 0.0
    event_counts: dict[str, int] = Field(default_factory=dict)
    judge: JudgeResult | None = None
    created_at: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.strategy, self.question_id)

    def retrieved_arxiv_ids(self) -> list[str]:
        """Paper of every retrieved chunk, searches in call order, ranks within a search.

        Chunks repeated by a later search are counted once (first occurrence).
        """
        seen: set[str] = set()
        out: list[str] = []
        for s in self.searches:
            for c in sorted(s.chunks, key=lambda c: c.rank):
                if c.chunk_id not in seen:
                    seen.add(c.chunk_id)
                    out.append(c.arxiv_id)
        return out


__all__ = [
    "EvalRecord",
    "FactJudgment",
    "JudgeResult",
    "RefusalJudgment",
    "RetrievedChunk",
    "SearchRecord",
]
