"""Data for the "How this answer was built" view, derived from one :class:`TurnResult`.

Pure functions on the turn's recorded traces (no I/O, no UI): the UI turns a :class:`RagTrace`
into charts and text. Per knowledge-base search it lists the query rewrites and filters and,
for every candidate chunk, the value it reached in each pipeline stage (dense score, BM25
rank, RRF score, rerank score, final rank). Across the turn it reports the latency per
pipeline stage and which sources the answer cited and which it left unused.

Candidate list names come from :func:`research_assistant.retrieval.strategies.retrieve`:
``dense:q0`` / ``dense_hyde:q0`` / ``bm25:q1`` (prefixed ``g<i>/`` per sub-question group),
``rrf`` / ``rrf:g<i>``, ``rerank`` / ``rerank:g<i>`` and ``final``. With several queries or
groups a chunk keeps its best value per stage (highest score, lowest rank).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from research_assistant.agent.state import TurnResult
from research_assistant.models import QueryFilters, RetrievalTrace

CandidateStatus = Literal["cited", "unused", "not_selected"]
LatencyGroup = Literal["security", "agent", "retrieval", "tools", "generation"]

MAX_CANDIDATES = 30

# Retrieval stages in pipeline order, with display labels.
RETRIEVAL_STAGE_LABELS: dict[str, str] = {
    "self_query": "Self-query filters",
    "query_translation": "Query rewrites",
    "retrieve": "Dense + BM25 search",
    "fuse": "RRF fusion",
    "rerank": "Cross-encoder rerank",
    "select": "Select top chunks",
}


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CandidateRow(_Frozen):
    """One candidate chunk of one search and its value in each stage (None = not reached)."""

    chunk_id: str
    arxiv_id: str
    paper_label: str
    source_number: int | None = Field(default=None, description="[n] in the answer prompt.")
    status: CandidateStatus
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    final_rank: int | None = None


class SearchView(_Frozen):
    number: int = Field(ge=1)
    question: str
    strategy: str
    semantic_query: str
    query_translation: str | None = None
    rewrites: list[str] = Field(default_factory=list)
    filters: QueryFilters | None = None
    stages: list[str] = Field(default_factory=list, description="Stages that ran, in order.")
    notes: list[str] = Field(default_factory=list)
    candidates: list[CandidateRow] = Field(default_factory=list)
    n_candidates_total: int = 0
    latency_ms: float = 0.0


class StageLatency(_Frozen):
    stage: str
    label: str
    group: LatencyGroup
    latency_ms: float = Field(ge=0)


class SourceUse(_Frozen):
    number: int
    label: str
    title: str
    section: str | None = None
    arxiv_id: str
    chunk_id: str
    cited: bool


class RagTrace(_Frozen):
    searches: list[SearchView] = Field(default_factory=list)
    latencies: list[StageLatency] = Field(default_factory=list)
    sources: list[SourceUse] = Field(default_factory=list)
    total_latency_ms: float = 0.0

    @property
    def cited(self) -> list[SourceUse]:
        return [s for s in self.sources if s.cited]

    @property
    def unused(self) -> list[SourceUse]:
        return [s for s in self.sources if not s.cited]


# --------------------------------------------------------------------------- candidates


def _list_kind(list_name: str) -> str:
    """``g1/dense:q0`` and ``dense_hyde:q0`` -> ``dense``; ``rrf:g0`` -> ``rrf``."""
    kind = list_name.rsplit("/", 1)[-1].split(":", 1)[0]
    return "dense" if kind.startswith("dense") else kind


def _max(a: float | None, b: float) -> float:
    return b if a is None else max(a, b)


class _Acc:
    def __init__(self, chunk_id: str, arxiv_id: str) -> None:
        self.chunk_id = chunk_id
        self.arxiv_id = arxiv_id
        self.dense: float | None = None
        self.bm25_rank: int | None = None
        self.bm25_score: float | None = None
        self.rrf: float | None = None
        self.rerank: float | None = None
        self.final: int | None = None
        self.best_rank = 10**9  # best rank in any list, for ordering


def _collect(trace: RetrievalTrace) -> dict[str, _Acc]:
    acc: dict[str, _Acc] = {}
    for stage in trace.stages:
        for c in stage.candidates:
            a = acc.get(c.chunk_id)
            if a is None:
                a = acc[c.chunk_id] = _Acc(c.chunk_id, c.arxiv_id)
            kind = _list_kind(c.list_name)
            if stage.stage == "retrieve" and kind == "dense":
                a.dense = _max(a.dense, c.score)
            elif stage.stage == "retrieve" and kind == "bm25":
                if a.bm25_rank is None or c.rank < a.bm25_rank:
                    a.bm25_rank, a.bm25_score = c.rank, c.score
            elif stage.stage == "fuse":
                a.rrf = _max(a.rrf, c.score)
            elif stage.stage == "rerank":
                a.rerank = _max(a.rerank, c.score)
            elif stage.stage == "select":
                a.final = c.rank if a.final is None else min(a.final, c.rank)
            a.best_rank = min(a.best_rank, c.rank)
    return acc


def _order_key(a: _Acc) -> tuple[float, float, float, int, str]:
    # Selected chunks first (by final rank), then by the latest stage they reached.
    return (
        a.final if a.final is not None else 10**9,
        -(a.rerank if a.rerank is not None else float("-inf")),
        -(a.rrf if a.rrf is not None else float("-inf")),
        a.best_rank,
        a.chunk_id,
    )


def search_view(
    number: int,
    trace: RetrievalTrace,
    sources_by_chunk: Mapping[str, SourceUse],
    paper_labels: Mapping[str, str],
    max_candidates: int = MAX_CANDIDATES,
) -> SearchView:
    acc = _collect(trace)
    ordered = sorted(acc.values(), key=_order_key)
    rows = []
    for a in ordered[:max_candidates]:
        src = sources_by_chunk.get(a.chunk_id)
        status: CandidateStatus = (
            "not_selected" if src is None else ("cited" if src.cited else "unused")
        )
        label = src.label if src else paper_labels.get(a.arxiv_id, f"arXiv:{a.arxiv_id}")
        rows.append(
            CandidateRow(
                chunk_id=a.chunk_id,
                arxiv_id=a.arxiv_id,
                paper_label=label,
                source_number=src.number if src else None,
                status=status,
                dense_score=a.dense,
                bm25_rank=a.bm25_rank,
                bm25_score=a.bm25_score,
                rrf_score=a.rrf,
                rerank_score=a.rerank,
                final_rank=a.final,
            )
        )
    return SearchView(
        number=number,
        question=trace.question,
        strategy=trace.strategy,
        semantic_query=trace.semantic_query,
        query_translation=trace.query_translation,
        rewrites=list(trace.rewrites),
        filters=trace.filters,
        stages=[s.stage for s in trace.stages],
        notes=[n for s in trace.stages for n in s.notes],
        candidates=rows,
        n_candidates_total=len(acc),
        latency_ms=trace.total_latency_ms,
    )


# --------------------------------------------------------------------------- latency


def _sum(values: Iterable[float]) -> float:
    return float(sum(values))


def stage_latencies(turn: TurnResult) -> list[StageLatency]:
    """Wall time per pipeline stage, in pipeline order; stages that did not run are omitted.

    Retrieval stages are summed over all searches of the turn. "Other tools and overhead" is
    what the agent's tool and finalize nodes spent beyond retrieval and generation (domain
    tools such as the DP accountant, citation parsing, bookkeeping).
    """
    node_ms: dict[str, float] = {}
    for step in turn.trace:
        node_ms[step.node] = node_ms.get(step.node, 0.0) + step.latency_ms

    out: list[StageLatency] = []

    def add(stage: str, label: str, group: LatencyGroup, ms: float) -> None:
        if ms > 0:
            out.append(StageLatency(stage=stage, label=label, group=group, latency_ms=ms))

    add(
        "input_checks",
        "Input validation + injection check",
        "security",
        node_ms.get("validate_input", 0.0) + node_ms.get("detect_injection", 0.0),
    )
    add("agent", "Planning (agent model)", "agent", node_ms.get("agent", 0.0))

    retrieval_total = 0.0
    for stage, label in RETRIEVAL_STAGE_LABELS.items():
        ms = _sum(s.latency_ms for r in turn.retrievals for s in r.stages if s.stage == stage)
        retrieval_total += ms
        add(stage, label, "retrieval", ms)

    other = (
        node_ms.get("tools", 0.0)
        + node_ms.get("finalize", 0.0)
        - retrieval_total
        - turn.generation_latency_ms
    )
    add("other_tools", "Other tools + overhead", "tools", max(0.0, other))
    add("generate", "Answer generation (LLM)", "generation", turn.generation_latency_ms)
    add(
        "verify",
        "Claim verification (judge, regeneration)",
        "generation",
        node_ms.get("verify", 0.0),
    )
    return out


# --------------------------------------------------------------------------- turn


def build_rag_trace(turn: TurnResult, max_candidates: int = MAX_CANDIDATES) -> RagTrace:
    """Everything the "How this answer was built" view shows for one turn."""
    cited = {c.chunk_id for c in turn.citations}
    sources = [
        SourceUse(
            number=s.number,
            label=s.label,
            title=s.title,
            section=s.section,
            arxiv_id=s.arxiv_id,
            chunk_id=s.chunk_id,
            cited=s.chunk_id in cited,
        )
        for s in turn.sources
    ]
    by_chunk = {s.chunk_id: s for s in sources}
    labels = dict(turn.paper_labels)
    for s in sources:  # source labels carry the a/b suffixes used in the answer
        labels.setdefault(s.arxiv_id, s.label)
    searches = [
        search_view(i, r, by_chunk, labels, max_candidates)
        for i, r in enumerate(turn.retrievals, 1)
    ]
    return RagTrace(
        searches=searches,
        latencies=stage_latencies(turn),
        sources=sources,
        total_latency_ms=turn.latency_ms,
    )


__all__ = [
    "MAX_CANDIDATES",
    "RETRIEVAL_STAGE_LABELS",
    "CandidateRow",
    "RagTrace",
    "SearchView",
    "SourceUse",
    "StageLatency",
    "build_rag_trace",
    "search_view",
    "stage_latencies",
]
