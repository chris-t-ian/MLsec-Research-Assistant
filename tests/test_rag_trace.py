"""RAG trace data for the "How this answer was built" view (pure functions on a TurnResult)."""

from __future__ import annotations

from typing import Any

import pytest

from research_assistant.agent.state import SourceRef, TraceStep, TurnResult
from research_assistant.models import (
    CandidateScore,
    Citation,
    QueryFilters,
    RetrievalTrace,
    StageTrace,
)
from research_assistant.rag_trace import build_rag_trace, stage_latencies


def cand(chunk_id: str, rank: int, score: float, list_name: str) -> CandidateScore:
    return CandidateScore(
        chunk_id=chunk_id,
        arxiv_id=chunk_id.split(":")[0],
        rank=rank,
        score=score,
        list_name=list_name,
    )


A, B, C = "1610.05820:1:0", "2112.03570:2:1", "1912.00001:0:0"


def hybrid_trace(**kw: Any) -> RetrievalTrace:
    stages = [
        StageTrace(
            stage="query_translation",
            latency_ms=300,
        ),
        StageTrace(
            stage="retrieve",
            latency_ms=100,
            candidates=[
                cand(A, 1, 0.81, "dense:q0"),
                cand(B, 2, 0.70, "dense:q0"),
                cand(A, 3, 0.79, "dense:q1"),  # second query: lower score, kept max
                cand(B, 1, 12.0, "bm25:q0"),
                cand(C, 2, 9.0, "bm25:q0"),
                cand(C, 1, 8.0, "bm25:q1"),  # better rank in the second list
            ],
        ),
        StageTrace(
            stage="fuse",
            latency_ms=1,
            candidates=[
                cand(B, 1, 0.032, "rrf"),
                cand(A, 2, 0.031, "rrf"),
                cand(C, 3, 0.016, "rrf"),
            ],
        ),
        StageTrace(
            stage="rerank",
            latency_ms=2000,
            candidates=[cand(A, 1, 4.2, "rerank"), cand(B, 2, 1.0, "rerank")],
        ),
        StageTrace(
            stage="select",
            latency_ms=1,
            candidates=[cand(A, 1, 4.2, "final"), cand(B, 2, 1.0, "final")],
        ),
    ]
    base: dict[str, Any] = {
        "strategy": "full",
        "question": "How does LiRA work?",
        "semantic_query": "LiRA likelihood ratio attack",
        "query_translation": "multi_query",
        "retriever": "hybrid",
        "rerank": True,
        "rewrites": ["likelihood ratio membership attack"],
        "filters": QueryFilters(year_min=2020),
        "stages": stages,
        "total_latency_ms": 2402,
    }
    return RetrievalTrace(**{**base, **kw})


def citation(n: int, chunk_id: str) -> Citation:
    return Citation(
        number=n,
        chunk_id=chunk_id,
        arxiv_id=chunk_id.split(":")[0],
        label="Shokri et al. 2017",
        title="t",
        url="https://arxiv.org/abs/x",
    )


def turn(**kw: Any) -> TurnResult:
    base: dict[str, Any] = {
        "session_id": "s",
        "turn_id": "t1",
        "question": "q",
        "answer": "a",
        "retrievals": [hybrid_trace()],
        "sources": [
            SourceRef(number=1, chunk_id=A, arxiv_id="1610.05820", label="Shokri et al. 2017"),
            SourceRef(number=2, chunk_id=B, arxiv_id="2112.03570", label="Carlini et al. 2021"),
        ],
        "citations": [citation(1, A)],
        "paper_labels": {"1912.00001": "Doe et al. 2019"},
        "trace": [
            TraceStep(step_id="agent:validate_input", node="validate_input", latency_ms=1),
            TraceStep(step_id="agent:detect_injection", node="detect_injection", latency_ms=2),
            TraceStep(step_id="agent:agent", node="agent", latency_ms=800),
            TraceStep(step_id="agent:tools", node="tools", latency_ms=2600),
            TraceStep(step_id="agent:finalize", node="finalize", latency_ms=1600),
        ],
        "generation_latency_ms": 1500,
        "latency_ms": 5100,
    }
    return TurnResult(**{**base, **kw})


def test_candidates_aggregate_best_value_per_stage() -> None:
    [search] = build_rag_trace(turn()).searches
    rows = {r.chunk_id: r for r in search.candidates}
    assert rows[A].dense_score == pytest.approx(0.81)
    assert rows[A].bm25_rank is None
    assert rows[C].bm25_rank == 1 and rows[C].bm25_score == 8.0
    assert rows[B].rrf_score == pytest.approx(0.032)
    assert rows[A].rerank_score == 4.2 and rows[C].rerank_score is None
    assert (rows[A].final_rank, rows[B].final_rank, rows[C].final_rank) == (1, 2, None)
    # Selected chunks first, by final rank.
    assert [r.chunk_id for r in search.candidates] == [A, B, C]
    assert search.n_candidates_total == 3


def test_candidate_status_and_labels() -> None:
    [search] = build_rag_trace(turn()).searches
    rows = {r.chunk_id: r for r in search.candidates}
    assert (rows[A].status, rows[A].source_number) == ("cited", 1)
    assert (rows[B].status, rows[B].paper_label) == ("unused", "Carlini et al. 2021")
    assert (rows[C].status, rows[C].paper_label) == ("not_selected", "Doe et al. 2019")


def test_unknown_paper_falls_back_to_arxiv_id() -> None:
    [search] = build_rag_trace(turn(paper_labels={})).searches
    assert search.candidates[-1].paper_label == "arXiv:1912.00001"


def test_rewrites_filters_and_stages() -> None:
    [search] = build_rag_trace(turn()).searches
    assert search.query_translation == "multi_query"
    assert search.rewrites == ["likelihood ratio membership attack"]
    assert search.filters is not None and search.filters.year_min == 2020
    assert search.stages == ["query_translation", "retrieve", "fuse", "rerank", "select"]


def test_hyde_and_group_lists_count_as_dense() -> None:
    trace = hybrid_trace(
        stages=[
            StageTrace(
                stage="retrieve",
                latency_ms=1,
                candidates=[cand(A, 1, 0.6, "dense_hyde:q0"), cand(B, 1, 0.5, "g1/dense:q0")],
            )
        ]
    )
    [search] = build_rag_trace(turn(retrievals=[trace])).searches
    assert {r.chunk_id: r.dense_score for r in search.candidates} == {A: 0.6, B: 0.5}


def test_max_candidates_caps_rows() -> None:
    [search] = build_rag_trace(turn(), max_candidates=2).searches
    assert len(search.candidates) == 2 and search.n_candidates_total == 3


def test_cited_and_unused_sources() -> None:
    trace = build_rag_trace(turn())
    assert [s.number for s in trace.cited] == [1]
    assert [s.number for s in trace.unused] == [2]


def test_stage_latencies_in_pipeline_order() -> None:
    lat = {s.stage: s.latency_ms for s in stage_latencies(turn())}
    assert list(lat) == [
        "input_checks",
        "agent",
        "query_translation",
        "retrieve",
        "fuse",
        "rerank",
        "select",
        "other_tools",
        "generate",
    ]
    assert lat["input_checks"] == 3
    assert lat["rerank"] == 2000
    # tools + finalize - retrieval (2402) - generation (1500)
    assert lat["other_tools"] == pytest.approx(2600 + 1600 - 2402 - 1500)


def test_retrieval_latency_summed_over_searches() -> None:
    lat = {s.stage: s.latency_ms for s in stage_latencies(turn(retrievals=[hybrid_trace()] * 2))}
    assert lat["rerank"] == 4000
    assert "other_tools" not in lat  # negative remainder is dropped


def test_turn_without_search() -> None:
    trace = build_rag_trace(turn(retrievals=[], sources=[], citations=[]))
    assert trace.searches == [] and trace.sources == []
    assert any(s.stage == "generate" for s in trace.latencies)
