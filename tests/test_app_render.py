"""Chainlit rendering helpers in app/render.py (pure formatting, no UI session needed)."""

from __future__ import annotations

from typing import Any

import pytest
import render
from chainlit.context import init_http_context

from research_assistant.agent.state import TurnResult
from research_assistant.models import Citation, SecurityEvent, ToolResult


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
