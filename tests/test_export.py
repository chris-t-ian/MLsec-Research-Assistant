"""Exports: JSON round trip, CSV columns, PDF with Unicode, thread -> Conversation."""

from __future__ import annotations

import csv
import io
import json
from datetime import UTC, datetime
from typing import Any

import pdfplumber
import pytest

from research_assistant.agent.state import ToolCallOutcome, TraceStep, TurnResult
from research_assistant.export import (
    CITATION_COLUMNS,
    MESSAGE_COLUMNS,
    METADATA_KEY,
    Conversation,
    ConversationMessage,
    conversation_from_thread,
    export_citations_csv,
    export_json,
    export_messages_csv,
    export_pdf,
    load_json,
    parse_turn_metadata,
    turn_metadata,
)
from research_assistant.export.csv_export import safe_cell
from research_assistant.models import Citation, ToolResult

T0 = datetime(2026, 11, 22, 10, 0, tzinfo=UTC)
T1 = datetime(2026, 11, 22, 10, 1, tzinfo=UTC)

ANSWER = (
    "DP-SGD reaches (ε, δ)-differential privacy with ε ≈ 1.2 and δ = 10⁻⁵ "
    "(Abadi et al. 2016 [1]); LiRA reports TPR at low FPR (Carlini et al. 2021 [2])."
)


def citation(n: int, arxiv_id: str, label: str, excerpt: str = "We clip gradients.") -> Citation:
    return Citation(
        number=n,
        chunk_id=f"{arxiv_id}:3:0",
        arxiv_id=arxiv_id,
        label=label,
        title=f"Paper {arxiv_id}",
        section="Method",
        excerpt=excerpt,
        url=f"https://arxiv.org/abs/{arxiv_id}",
    )


def turn() -> TurnResult:
    return TurnResult(
        session_id="s1",
        turn_id="t1",
        question="What ε does DP-SGD reach?",
        answer=ANSWER,
        raw_answer="... [1] ... [2].",
        citations=[
            citation(1, "1607.00133", "Abadi et al. 2016", "Noise is added; δ ≪ 1/n."),
            citation(2, "2112.03570", "Carlini et al. 2021", "=HYPERLINK(evil)"),
            citation(3, "1607.00133", "Abadi et al. 2016"),
        ],
        tool_results=[
            ToolCallOutcome(
                call_id="c1",
                name="dp_epsilon",
                args={"noise_multiplier": 1.1, "steps": 10000},
                allowed=True,
                result=ToolResult(ok=True, data={"epsilon": 1.23}, source="dp_accounting"),
            )
        ],
        trace=[TraceStep(step_id="n1", node="agent", latency_ms=12.5, output={"calls": []})],
        model="test/model",
        latency_ms=1234.0,
    )


def conversation() -> Conversation:
    return Conversation(
        id="thread-1",
        title="DP-SGD ε question",
        user="demo",
        created_at=T0,
        exported_at=T1,
        messages=[
            ConversationMessage(id="m1", role="user", content=turn().question, created_at=T0),
            ConversationMessage(
                id="m2",
                role="assistant",
                content=ANSWER,
                created_at=T1,
                strategy="hybrid",
                top_k=8,
                turn=turn(),
            ),
        ],
    )


def read_csv(data: bytes) -> list[dict[str, str]]:
    assert data.startswith(b"\xef\xbb\xbf"), "UTF-8 BOM for spreadsheet apps"
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline="")))


# --------------------------------------------------------------------------- JSON


def test_json_round_trip_keeps_everything() -> None:
    conv = conversation()
    data = export_json(conv)
    assert load_json(data) == conv
    raw = json.loads(data)
    answer = raw["messages"][1]
    assert answer["strategy"] == "hybrid"
    assert answer["created_at"].startswith("2026-11-22T10:01")
    assert [c["arxiv_id"] for c in answer["turn"]["citations"]][:2] == ["1607.00133", "2112.03570"]
    assert answer["turn"]["tool_results"][0]["result"]["data"] == {"epsilon": 1.23}
    assert answer["turn"]["trace"][0]["node"] == "agent"
    assert "ε" in data.decode("utf-8")


# --------------------------------------------------------------------------- CSV


def test_messages_csv_columns_and_rows() -> None:
    rows = read_csv(export_messages_csv(conversation()))
    assert tuple(rows[0].keys()) == MESSAGE_COLUMNS
    assert [r["role"] for r in rows] == ["user", "assistant"]
    assert rows[0]["cited_arxiv_ids"] == "" and rows[0]["strategy"] == ""
    assert rows[1]["strategy"] == "hybrid"
    assert rows[1]["cited_arxiv_ids"] == "1607.00133;2112.03570"  # de-duplicated, in order
    assert rows[1]["content"] == ANSWER  # embedded quotes/commas/Unicode survive
    assert rows[1]["timestamp"] == T1.isoformat()


def test_citations_csv_one_row_per_citation() -> None:
    rows = read_csv(export_citations_csv(conversation()))
    assert tuple(rows[0].keys()) == CITATION_COLUMNS
    assert [(r["message_id"], r["marker"], r["arxiv_id"]) for r in rows] == [
        ("m2", "[1]", "1607.00133"),
        ("m2", "[2]", "2112.03570"),
        ("m2", "[3]", "1607.00133"),
    ]
    assert rows[0]["section"] == "Method"
    assert rows[0]["excerpt"] == "Noise is added; δ ≪ 1/n."
    assert rows[1]["excerpt"] == "'=HYPERLINK(evil)"  # no formula injection from paper text


@pytest.mark.parametrize("value", ["=1+1", "+1", "-1", "@SUM(A1)", "\tx"])
def test_safe_cell_neutralizes_formulas(value: str) -> None:
    assert safe_cell(value) == "'" + value
    assert safe_cell("ε = 1") == "ε = 1"


def test_empty_conversation_csv_has_header_only() -> None:
    empty = Conversation(id="x")
    assert read_csv(export_messages_csv(empty)) == []
    assert read_csv(export_citations_csv(empty)) == []


# --------------------------------------------------------------------------- PDF


def pdf_text(data: bytes) -> str:
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def test_pdf_contains_questions_answers_references_and_unicode() -> None:
    data = export_pdf(conversation())
    assert data.startswith(b"%PDF-") and len(data) > 1000
    assert b"DejaVu" in data  # font embedded
    text = pdf_text(data)
    assert "DP-SGD ε question" in text
    assert "What ε does DP-SGD reach?" in text
    assert "ε" in text and "δ" in text and "≈" in text
    assert "References" in text
    assert "https://arxiv.org/abs/1607.00133" in text
    assert "https://arxiv.org/abs/2112.03570" in text
    assert b"/URI (https://arxiv.org/abs/2112.03570)" in data  # clickable link


def test_pdf_handles_long_unbroken_text_and_many_messages() -> None:
    long = "x" * 2000 + " https://arxiv.org/abs/" + "9" * 300
    msgs = [
        ConversationMessage(id=f"m{i}", role="user" if i % 2 == 0 else "assistant", content=long)
        for i in range(30)
    ]
    data = export_pdf(Conversation(id="long", messages=msgs))
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        assert len(pdf.pages) > 3


def test_pdf_of_empty_conversation() -> None:
    assert "no messages yet" in pdf_text(export_pdf(Conversation(id="x")))


# --------------------------------------------------------------------------- metadata / thread


def test_turn_metadata_round_trip_through_json_text() -> None:
    meta = turn_metadata(turn(), "hybrid", 8)
    as_stored = json.dumps(meta)  # SQLite stores metadata as JSON text
    parsed, strategy, top_k = parse_turn_metadata(as_stored)
    assert parsed == turn() and strategy == "hybrid" and top_k == 8


@pytest.mark.parametrize(
    "metadata",
    [None, "", "not json", {}, {"favorite": True}, {METADATA_KEY: {"turn_result": {"bad": 1}}}],
)
def test_parse_turn_metadata_tolerates_garbage(metadata: Any) -> None:
    parsed, _, _ = parse_turn_metadata(metadata)
    assert parsed is None


def test_conversation_from_thread() -> None:
    thread = {
        "id": "thread-1",
        "name": "DP-SGD ε question",
        "createdAt": "2026-11-22T10:00:00.000000Z",
        "userIdentifier": "demo",
        "metadata": "{}",
        "steps": [
            # Out of order on purpose; non-message steps and empty placeholders are skipped.
            {
                "id": "m2",
                "type": "assistant_message",
                "output": "rendered answer",
                "createdAt": "2026-11-22T10:01:00Z",
                "metadata": json.dumps(turn_metadata(turn(), "hybrid", None)),
            },
            {"id": "r1", "type": "run", "output": "Research", "createdAt": "2026-11-22T10:00:30Z"},
            {
                "id": "m1",
                "type": "user_message",
                "output": "What ε does DP-SGD reach?",
                "createdAt": "2026-11-22T10:00:10Z",
                "metadata": "{}",
            },
            {"id": "m0", "type": "assistant_message", "output": "", "createdAt": None},
        ],
    }
    conv = conversation_from_thread(thread)
    assert conv.id == "thread-1" and conv.user == "demo" and conv.title == "DP-SGD ε question"
    assert [(m.id, m.role) for m in conv.messages] == [("m1", "user"), ("m2", "assistant")]
    answer = conv.messages[1]
    assert answer.turn == turn() and answer.strategy == "hybrid"
    assert answer.cited_arxiv_ids() == ["1607.00133", "2112.03570"]
    assert conv.messages[0].created_at == datetime(2026, 11, 22, 10, 0, 10, tzinfo=UTC)


def test_markdown_lines_for_pdf() -> None:
    from research_assistant.export.pdf_export import markdown_lines

    md = (
        "## Summary\n**What (ε, δ)-DP guarantees**\n- A **bold** claim [1]\n\n"
        "  * nested `code`\nDP--SGD"
    )
    assert markdown_lines(md) == [
        ("B", "Summary"),
        ("B", "What (ε, δ)-DP guarantees"),
        ("", "• A bold claim [1]"),
        ("", ""),
        ("", "  • nested code"),
        ("", "DP--SGD"),
    ]


def test_pdf_renders_answer_without_markdown_markers() -> None:
    conv = conversation()
    msg = conv.messages[1]
    md_turn = msg.turn.model_copy(update={"answer": "**Bold heading**\n- item with ε"})
    conv = conv.model_copy(
        update={"messages": [conv.messages[0], msg.model_copy(update={"turn": md_turn})]}
    )
    text = pdf_text(export_pdf(conv))
    assert "Bold heading" in text and "**" not in text
    assert "• item with ε" in text
