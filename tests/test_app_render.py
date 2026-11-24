"""Chainlit rendering helpers in app/render.py (pure formatting, no UI session needed)."""

from __future__ import annotations

from typing import Any

import pytest
import render
from chainlit.context import init_http_context

from research_assistant.agent.state import TurnResult
from research_assistant.app_guide import GuideAnswer, HelpSection
from research_assistant.models import (
    Answer,
    Citation,
    ClaimCheck,
    SecurityEvent,
    ToolResult,
    VerificationInfo,
)
from research_assistant.rag_trace import (
    CandidateRow,
    RagTrace,
    SearchView,
    SourceUse,
    StageLatency,
)


@pytest.fixture
async def chainlit_context() -> None:
    """Elements (cl.Text, cl.File) need a Chainlit context; an HTTP context needs no server."""
    init_http_context()


def cite(n: int = 1, label: str = "Shokri et al. 2017") -> Citation:
    return Citation(
        number=n,
        chunk_id="1610.05820:2:0",
        arxiv_id="1610.05820",
        label=label,
        title="Membership Inference Attacks against Machine Learning Models",
        section="Shadow training",
        excerpt="We use shadow models.",
        url="https://arxiv.org/abs/1610.05820",
    )


def turn(**kw: Any) -> TurnResult:
    base: dict[str, Any] = {"session_id": "s", "turn_id": "t1", "question": "q", "answer": "a"}
    return TurnResult(**{**base, **kw})


def test_md_escapes_untrusted_markdown() -> None:
    out = render.md("[click](http://evil) | *x*\n<img src=x>")
    assert "](" not in out.replace("\\](", "")
    assert "\\[click\\]" in out and "\\|" in out and "\\<img" in out
    assert "\n" not in out


def test_links_only_from_valid_ids() -> None:
    assert render.arxiv_url("1610.05820") == "https://arxiv.org/abs/1610.05820"
    assert render.arxiv_url("cs/0112017v2") == "https://arxiv.org/abs/cs/0112017v2"
    assert render.arxiv_url("javascript:alert(1)") is None
    assert render.paper_link("T", None, "not a doi") == "T"


def test_step_types() -> None:
    assert render.step_type("agent:t1:agent:1", "agent") == "llm"
    assert render.step_type("agent:t1:tools:1", "tools") == "tool"
    assert render.step_type("agent:t1:finalize:1", "finalize") == "run"
    assert render.step_type("retrieval:hybrid:select", "select") == "retrieval"
    assert render.step_type("retrieval:full:query_translation", "query_translation") == "llm"
    assert render.step_type("generation:hybrid:generate", "generate") == "llm"
    assert render.step_type("ingest:2112.03570:fetch", "fetch") == "tool"


def test_step_outputs_are_readable() -> None:
    out = render.format_step_output(
        "query_translation", {"rewrites": ["LiRA likelihood ratio", "shadow models"]}
    )
    assert "1. LiRA likelihood ratio" in out and "2. shadow models" in out
    out = render.format_step_output(
        "self_query",
        {"semantic_query": "MIA", "filters": {"year_min": 2020, "section_type": "any"}},
    )
    assert "year ≥ 2020" in out
    out = render.format_step_output(
        "select", {"chunks": [{"chunk_id": "a:1:0", "score": 0.5}]}, {"latency_ms": 12.0}
    )
    assert "| 1 | a:1:0 | 0.5 |" in out and "_12 ms_" in out


def test_injection_step_hides_scores_and_patterns() -> None:
    out = render.format_step_output(
        "detect_injection", {"verdict": "warn", "score": 0.55, "matches": ["ignore_previous"]}
    )
    assert "warn" in out and "0.55" not in out and "ignore_previous" not in out


def test_dp_accountant_table() -> None:
    data = {
        "mode": "epsilon",
        "accountant": "rdp",
        "epsilon": 1.234567,
        "delta": 1e-5,
        "noise_multiplier": 1.1,
        "sample_rate": 0.01,
        "steps": 10000,
        "target_epsilon": None,
        "sampling": "poisson",
        "neighboring_relation": "add_or_remove_one",
        "warnings": ["delta too large"],
    }
    out = render.format_tool_result(
        "dp_accountant", ToolResult(ok=True, data=data, source="dp_accounting")
    )
    assert "| ε (epsilon) | 1.235 |" in out
    assert "| T (steps) | 10000 |" in out
    assert "⚠️ delta too large" in out


def test_citation_graph_links_and_badge() -> None:
    data = {
        "arxiv_id": "1610.05820",
        "direction": "citations",
        "papers": [
            {
                "title": "LiRA",
                "year": 2022,
                "authors": ["N. Carlini"],
                "arxiv_id": "2112.03570",
                "doi": None,
                "citation_count": 900,
                "in_kb": True,
            },
            {
                "title": "Other",
                "year": 2023,
                "authors": [],
                "arxiv_id": None,
                "doi": "10.1145/123",
                "citation_count": 1,
                "in_kb": False,
            },
        ],
    }
    out = render.format_tool_result(
        "citation_graph", ToolResult(ok=True, data=data, source="semantic_scholar")
    )
    lines = out.splitlines()
    assert "Papers citing arXiv:1610.05820" in lines[0]
    assert "[LiRA](https://arxiv.org/abs/2112.03570)" in out
    assert render.IN_KB_BADGE in lines[2] and render.IN_KB_BADGE not in lines[3]
    assert "[Other](https://doi.org/10.1145/123)" in out


def test_arxiv_search_badge() -> None:
    data = {
        "query": "x",
        "results": [
            {
                "arxiv_id": "2112.03570",
                "title": "LiRA",
                "authors": ["A"],
                "year": 2021,
                "abstract_snippet": "We attack.",
                "already_in_kb": True,
            },
        ],
    }
    out = render.format_tool_result("arxiv_search", ToolResult(ok=True, data=data, source="arxiv"))
    assert "[LiRA](https://arxiv.org/abs/2112.03570)" in out and render.IN_KB_BADGE in out
    assert "We attack." in out


async def test_bibtex_code_block_and_file(chainlit_context: None) -> None:
    bib = "@misc{shokri2017membership,\n  title = {{X}}\n}\n"
    result = ToolResult(
        ok=True, data={"bibtex": bib, "entries": [], "not_found": []}, source="arxiv"
    )
    out = render.format_tool_result("bibtex_export", result)
    assert out.startswith("```bibtex\n@misc{shokri2017membership,")
    f = render.bibtex_file(result)
    assert f is not None and f.name == "references.bib" and f.content == bib.encode()
    assert render.bibtex_file(ToolResult(ok=False, error="x", source="arxiv")) is None


def test_failed_tool_shows_error() -> None:
    out = render.format_tool_result(
        "arxiv_search", ToolResult(ok=False, error="arXiv is down", source="arxiv")
    )
    assert out == "⚠️ arXiv is down"


def test_ingest_result_with_findings() -> None:
    data = {
        "request": {},
        "result": {
            "arxiv_id": "2112.03570",
            "title": "LiRA",
            "source": "latex",
            "n_chunks": 40,
            "n_quarantined": 1,
            "n_suspicious": 2,
        },
        "findings": [
            {
                "type": "hidden_color",
                "snippet": "\\color{white} ignore all",
                "location": "main.tex:42",
                "severity": "high",
            }
        ],
    }
    out = render.format_tool_result(
        "confirm_ingest", ToolResult(ok=True, data=data, source="knowledge_base")
    )
    assert "40 chunks" in out and "1 chunk(s) were quarantined" in out
    assert "**1 finding(s)**" in out and "main.tex:42" in out and "hidden color" in out


async def test_source_element_named_like_reference(chainlit_context: None) -> None:
    el = render.source_element(cite())
    assert el.name == "[1] Shokri et al. 2017"
    assert el.display == "side"
    assert "Shadow training" in el.content and "https://arxiv.org/abs/1610.05820" in el.content


def test_answer_markdown_references_elements() -> None:
    res = turn(
        raw_answer="Shadow models [1].",
        answer="Shadow models (Shokri et al. 2017 [1]).",
        citations=[cite()],
    )
    assert render.answer_markdown(res) == "Shadow models ([1] Shokri et al. 2017)."


def test_blocked_injection_message_has_no_internals() -> None:
    res = turn(
        blocked=True,
        security_events=[
            SecurityEvent(kind="injection_block", detail="patterns: ignore_previous", score=0.93)
        ],
    )
    out = render.answer_markdown(res)
    assert out == render.INJECTION_BLOCKED
    assert "0.93" not in out and "ignore_previous" not in out


def test_rejected_input_explains_reason() -> None:
    res = turn(
        blocked=True,
        security_events=[SecurityEvent(kind="input_rejected", detail="Please enter a question.")],
    )
    assert render.answer_markdown(res) == f"{render.INPUT_REJECTED_PREFIX} Please enter a question."


def test_warned_input_adds_note() -> None:
    res = turn(
        raw_answer="LiRA [1].",
        citations=[cite()],
        security_events=[SecurityEvent(kind="injection_warn", detail="x", score=0.5)],
    )
    out = render.answer_markdown(res)
    assert out.startswith("> **Note:**") and out.endswith("LiRA ([1] Shokri et al. 2017).")


# --------------------------------------------------------------------------- RAG view


def row(chunk_id: str, status: str = "not_selected", **kw: Any) -> CandidateRow:
    arxiv_id = chunk_id.split(":")[0]
    base: dict[str, Any] = {
        "chunk_id": chunk_id,
        "arxiv_id": arxiv_id,
        "paper_label": f"Paper {arxiv_id}",
        "status": status,
    }
    return CandidateRow(**{**base, **kw})


def search_view(rows: list[CandidateRow], **kw: Any) -> SearchView:
    base: dict[str, Any] = {
        "number": 1,
        "question": "How does LiRA work?",
        "strategy": "hybrid",
        "semantic_query": "How does LiRA work?",
        "candidates": rows,
        "n_candidates_total": len(rows),
        "latency_ms": 1234,
        "stages": ["retrieve", "fuse", "select"],
    }
    return SearchView(**{**base, **kw})


def rag(searches: list[SearchView], **kw: Any) -> RagTrace:
    base: dict[str, Any] = {
        "searches": searches,
        "latencies": [
            StageLatency(
                stage="retrieve", label="Dense + BM25 search", group="retrieval", latency_ms=120
            ),
            StageLatency(
                stage="generate",
                label="Answer generation (LLM)",
                group="generation",
                latency_ms=2500,
            ),
        ],
        "sources": [
            SourceUse(
                number=1,
                label="Shokri et al. 2017",
                title="MIA",
                arxiv_id="1610.05820",
                chunk_id="1610.05820:1:0",
                cited=True,
            ),
            SourceUse(
                number=2,
                label="Carlini et al. 2021",
                title="LiRA",
                arxiv_id="2112.03570",
                chunk_id="2112.03570:1:0",
                cited=False,
            ),
        ],
        "total_latency_ms": 4000,
    }
    return RagTrace(**{**base, **kw})


def hybrid_rows() -> list[CandidateRow]:
    return [
        row(
            "1610.05820:1:0",
            "cited",
            source_number=1,
            dense_score=0.8,
            bm25_rank=2,
            rrf_score=0.03,
            final_rank=1,
        ),
        row(
            "2112.03570:1:0",
            "unused",
            source_number=2,
            dense_score=0.7,
            rrf_score=0.02,
            final_rank=2,
        ),
        row("1912.00001:0:3", bm25_rank=1, rrf_score=0.016),
    ]


def test_candidates_figure_has_one_panel_per_stage_that_ran() -> None:
    fig = render.candidates_figure(search_view(hybrid_rows()))
    assert fig is not None
    titles = [a.text for a in fig.layout.annotations]
    assert titles == ["Dense score (cosine)", "BM25 rank", "RRF score"]  # no rerank stage
    assert fig.layout.xaxis2.autorange == "reversed"  # BM25 rank: better to the right
    labels = list(fig.layout.yaxis.categoryarray)
    assert labels == ["[1] Paper 1610.05820", "[2] Paper 2112.03570", "Paper 1912.00001 · s0·c3"]


def test_candidates_figure_colors_by_paper_and_marks_status() -> None:
    fig = render.candidates_figure(search_view(hybrid_rows()))
    assert fig is not None
    data = [t for t in fig.data if t.x and t.x[0] is not None]
    by_name = {t.name: t for t in data if t.xaxis == "x"}  # dense panel
    assert by_name["Paper 1610.05820"].marker.color == render.PAPER_COLORS[0]
    assert by_name["Paper 2112.03570"].marker.color == render.PAPER_COLORS[1]
    assert list(by_name["Paper 1610.05820"].marker.symbol) == ["circle"]
    assert list(by_name["Paper 2112.03570"].marker.symbol) == ["circle-open"]
    legend = [t.name for t in fig.data if t.showlegend is not False]
    assert "cited in the answer" in legend and "Paper 1912.00001" in legend


def test_candidates_figure_folds_extra_papers_into_other() -> None:
    rows = [row(f"2001.{i:05d}:0:0", dense_score=1 - i / 10) for i in range(9)]
    fig = render.candidates_figure(search_view(rows))
    assert fig is not None
    names = {t.name for t in fig.data}
    assert render.OTHER_PAPERS in names
    assert len(names - {render.OTHER_PAPERS} - {s[3] for s in render.STATUS_STYLE.values()}) == 7


def test_candidates_figure_escapes_untrusted_labels() -> None:
    bad = row("1610.05820:1:0", dense_score=0.5, paper_label='<a href="http://evil">x</a>')
    fig = render.candidates_figure(search_view([bad]))
    assert fig is not None
    dumped = fig.to_json()
    assert "<a href" not in dumped and "&lt;a href" in dumped


def test_candidates_figure_none_without_candidates() -> None:
    assert render.candidates_figure(search_view([])) is None


def test_latency_figure_bars_in_pipeline_order() -> None:
    fig = render.latency_figure(rag([]))
    assert fig is not None
    assert list(fig.layout.yaxis.categoryarray) == [
        "Dense + BM25 search",
        "Answer generation (LLM)",
    ]
    assert {t.name for t in fig.data} == {"Retrieval", "Generation"}
    assert render.latency_figure(rag([], latencies=[])) is None


def test_rag_markdown_lists_rewrites_filters_and_sources() -> None:
    s = search_view(
        hybrid_rows(),
        query_translation="hyde",
        rewrites=["A *hypothetical* passage"],
        stages=["self_query", "query_translation", "retrieve"],
    )
    text = render.rag_markdown(rag([s]))
    assert "Hypothetical passage (HyDE)" in text and "A \\*hypothetical\\* passage" in text
    assert "Extracted filters: none" in text
    assert "1 cited, 1 unused" in text
    assert "[1] Shokri et al. 2017" in text and "Retrieved but not cited" in text


def test_rag_markdown_without_search_explains_missing_chart() -> None:
    text = render.rag_markdown(rag([], sources=[]))
    assert "No knowledge-base search" in text


async def test_rag_elements_are_inline_plotly(chainlit_context: None) -> None:
    elements = render.rag_elements(rag([search_view(hybrid_rows())]))
    assert [e.name for e in elements] == ["Candidates per stage", "Latency per stage"]
    assert all(e.display == "inline" for e in elements)


async def test_help_element_is_named_like_the_reference(chainlit_context: None) -> None:
    sec = HelpSection(title="Export", text="JSON, CSV, PDF")
    ans = GuideAnswer(question="q", text="Use export (Help: Export).", cited=[sec])
    el = render.help_element(ans.cited[0])
    assert el.name == "Help: Export" and el.name in ans.text


# --------------------------------------------------------------------------- claim badges


def _claims_turn() -> TurnResult:
    raw = (
        "## LiRA\n- Shadow models are used [1].\n- The *attack* fails [1].\n\n"
        "Uncited fact here today."
    )
    s1 = raw.index("Shadow")
    s2 = raw.index("The *attack*")
    s3 = raw.index("Uncited")
    checks = [
        ClaimCheck(
            index=1,
            sentence="Shadow models are used [1].",
            start=s1,
            end=s1 + 27,
            cited=[1],
            status="supported",
            judge_label="supported",
            evidence_span="We use shadow models.",
            evidence_source=1,
            evidence_before="In this work ",
            evidence_after=" They mimic *the* target.",
            rationale="Stated directly.",
        ),
        ClaimCheck(
            index=2,
            sentence="The *attack* fails [1].",
            start=s2,
            end=s2 + 23,
            cited=[1],
            status="unsupported",
            judge_label="supported",
            evidence_span="It fails <img src=x>",
            rationale="Quote not in source.",
            note="fabricated evidence",
        ),
        ClaimCheck(
            index=3,
            sentence="Uncited fact here today.",
            start=s3,
            end=s3 + 24,
            status="uncited",
        ),
    ]
    info = VerificationInfo(policy="flag", status="ok", n_judge_calls=1)
    return turn(
        raw_answer=raw,
        citations=[cite()],
        claim_checks=checks,
        supported_rate=1 / 3,
        verification=info,
    )


def test_answer_markdown_adds_claim_badges_and_summary() -> None:
    out = render.answer_markdown(_claims_turn())
    assert out == (
        "## LiRA\n"
        "- Shadow models are used ([1] Shokri et al. 2017). ✓ C1\n"
        "- The *attack* fails ([1] Shokri et al. 2017). ✗ C2\n\n"
        "Uncited fact here today. ? C3\n\n"
        "_Claim check: 1 of 3 claims supported (1 unsupported, 1 without citation)_"
    )


def test_stale_claim_span_gets_no_badge() -> None:
    res = _claims_turn()
    bad = res.claim_checks[0].model_copy(update={"start": 0, "end": 5})
    res = res.model_copy(update={"claim_checks": [bad, *res.claim_checks[1:]]})
    out = render.answer_markdown(res)
    assert "✓ C1" not in out and "✗ C2" in out


def test_verification_error_line() -> None:
    info = VerificationInfo(policy="flag", status="error", error="x")
    res = turn(raw_answer="A [1].", citations=[cite()], verification=info)
    assert render.answer_markdown(res).endswith("_Claim check unavailable for this answer._")
    off = turn(raw_answer="A [1].", citations=[cite()])
    assert "Claim check" not in render.answer_markdown(off)


async def test_claim_elements_highlight_evidence_and_escape(chainlit_context: None) -> None:
    res = _claims_turn()
    supported, fabricated, uncited = (
        render.claim_element(c, res.citations) for c in res.claim_checks
    )
    assert supported.name == "✓ C1" and supported.display == "side"
    assert "**C1 · Supported**" in supported.content
    assert "In this work **We use shadow models.** They mimic \\*the\\* target." in (
        supported.content
    )
    assert "[1] Shokri et al. 2017" in supported.content.replace("\\[", "[").replace("\\]", "]")
    assert fabricated.name == "✗ C2" and "Fabricated evidence" in fabricated.content
    assert "<img" not in fabricated.content.replace("\\<img", "")
    assert uncited.name == "? C3" and "cites a source, so it was not checked" in uncited.content


def test_verify_step_output() -> None:
    out = render.format_step_output(
        "verify_claims",
        {"counts": {"supported": 2, "unsupported": 1}, "fabricated": 1},
        {"judge_calls": 1, "latency_ms": 900},
    )
    assert "supported: 2" in out and "1 verdict(s) quoted text" in out
    assert render.step_label("verify_claims") == "Verifying claims"
    out = render.format_step_output(
        "verify", {"policy": "regenerate", "status": "ok", "summary": "1 of 1", "regenerated": True}
    )
    assert "rewritten once" in out


async def test_borrowed_and_uncited_badges_are_grouped(chainlit_context: None) -> None:
    raw = "Shadow models mimic the target. They use similar data [1]. One query is enough."
    s2 = raw.index("They")
    s3 = raw.index("One")
    checks = [
        ClaimCheck(index=1, sentence="Shadow models mimic the target.", start=0, end=31,
                   cited=[1], status="supported"),
        ClaimCheck(index=2, sentence="They use similar data [1].", start=s2, end=s2 + 26,
                   cited=[1], status="partially_supported"),
        ClaimCheck(index=3, sentence="One query is enough.", start=s3, end=s3 + 20,
                   status="uncited"),
    ]  # fmt: skip
    info = VerificationInfo(policy="flag", status="ok", n_judge_calls=1)
    res = turn(raw_answer=raw, citations=[cite()], claim_checks=checks, verification=info)
    out = render.answer_markdown(res).split("\n\n")[0]
    assert out == (
        "Shadow models mimic the target. They use similar data ([1] Shokri et al. 2017). "
        "✓ C1 ◐ C2 One query is enough. ? C3"
    )
    panel = render.claim_element(checks[0], [cite()]).content
    assert "Cites (via the next cited sentence of the paragraph)" in panel


def test_progress_labels_for_the_research_step() -> None:
    assert render.progress_label("agent:t:validate_input:1", "validate_input") == (
        "Validating input…"
    )
    assert render.progress_label("verify:t", "verify_claims") == "Verifying claims…"
    assert render.progress_label("generation:hybrid:revision:generate", "generate") == (
        "Revising answer…"
    )
    assert render.progress_label("retrieval:x:fuse", "fuse") is None


def test_draft_markdown_links_sources_and_says_claims_are_checked() -> None:
    ans = Answer(question="q", text="x", raw_text="Shadow models [1].", citations=[cite()])
    out = render.draft_markdown(ans, verifying=True)
    assert out.startswith("Shadow models ([1] Shokri et al. 2017).")
    assert out.endswith(render.CHECKING_CLAIMS)
    assert render.draft_markdown(ans, verifying=False) == "Shadow models ([1] Shokri et al. 2017)."


def test_security_notice_mentions_removed_links() -> None:
    filtered = SecurityEvent(kind="output_filtered", detail="removed")
    warned = SecurityEvent(kind="injection_warn", detail="warn", score=0.5)

    text, replaces = render.security_notice([filtered], blocked=False)
    assert text == render.OUTPUT_FILTERED and not replaces

    both, replaces = render.security_notice([warned, filtered], blocked=False)
    assert both is not None and render.INJECTION_WARNED in both
    assert render.OUTPUT_FILTERED in both and not replaces
