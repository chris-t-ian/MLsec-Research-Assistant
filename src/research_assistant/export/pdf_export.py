"""PDF export with fpdf2: title, date, every question and answer, references per answer.

DejaVu Sans is embedded (``fonts/``) so Greek letters and math symbols such as epsilon and
delta render; the built-in PDF core fonts only cover Latin-1.
"""

from __future__ import annotations

import re
from datetime import UTC
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from research_assistant.export.models import Conversation, ConversationMessage
from research_assistant.models import Citation

FONT_DIR = Path(__file__).resolve().parent / "fonts"
FONT_FAMILY = "DejaVu"
DEFAULT_TITLE = "Research Assistant conversation"

_TEXT = (33, 37, 41)
_MUTED = (108, 117, 125)
_LINK = (13, 110, 253)


def arxiv_abs_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}"


class _ConversationPDF(FPDF):
    def footer(self) -> None:
        self.set_y(-12)
        self.set_font(FONT_FAMILY, "", 8)
        self.set_text_color(*_MUTED)
        self.cell(0, 8, f"Page {self.page_no()}/{{nb}}", align="C")


def _new_pdf(title: str) -> _ConversationPDF:
    pdf = _ConversationPDF(format="A4")
    pdf.add_font(FONT_FAMILY, "", FONT_DIR / "DejaVuSans.ttf")
    pdf.add_font(FONT_FAMILY, "B", FONT_DIR / "DejaVuSans-Bold.ttf")
    pdf.set_title(title)
    pdf.set_creator("research-assistant")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 18, 18)
    pdf.add_page()
    return pdf


_HEADING = re.compile(r"^\s*#{1,6}\s+")
_BULLET = re.compile(r"^(\s*)[-*+]\s+")
_INLINE = re.compile(r"\*\*|__|`")


def markdown_lines(text: str) -> list[tuple[str, str]]:
    """Answer Markdown as ``(style, line)`` pairs for plain PDF text.

    Headings and lines that are entirely bold become bold lines, bullets become "•", other
    inline markers are dropped. (fpdf2's own markdown mode would misread "--" in paper text.)
    """
    lines: list[tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        style = ""
        if _HEADING.match(line):
            line, style = _HEADING.sub("", line), "B"
        stripped = line.strip()
        if (
            len(stripped) > 4
            and stripped.startswith("**")
            and stripped.endswith("**")
            and "**" not in stripped[2:-2]
        ):
            style = "B"
        line = _BULLET.sub(lambda m: m.group(1) + "• ", line)
        lines.append((style, _INLINE.sub("", line)))
    return lines


def _para(pdf: FPDF, text: str, size: float = 10, style: str = "", color=_TEXT) -> None:
    pdf.set_font(FONT_FAMILY, style, size)
    pdf.set_text_color(*color)
    pdf.multi_cell(0, size * 0.5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _reference(pdf: FPDF, c: Citation) -> None:
    section = f", {c.section}" if c.section else ""
    _para(pdf, f"[{c.number}] {c.label}: {c.title}{section}", size=9)
    url = arxiv_abs_url(c.arxiv_id)
    pdf.set_font(FONT_FAMILY, "", 9)
    pdf.set_text_color(*_LINK)
    pdf.set_x(pdf.l_margin + 6)
    pdf.multi_cell(0, 4.5, url, link=url, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)


def _message(pdf: FPDF, m: ConversationMessage) -> None:
    when = m.created_at.strftime("%Y-%m-%d %H:%M UTC") if m.created_at else ""
    if m.role == "user":
        pdf.ln(3)
        _para(pdf, f"Question  {when}".rstrip(), size=9, style="B", color=_MUTED)
        _para(pdf, m.content, size=11, style="B")
        pdf.ln(2)
        return

    _para(pdf, "Answer", size=9, style="B", color=_MUTED)
    for style, line in markdown_lines(m.turn.answer if m.turn else m.content):
        if line.strip():
            _para(pdf, line, style=style)
        else:
            pdf.ln(2)
    details = [f"strategy: {m.strategy}" if m.strategy else ""]
    if m.turn and m.turn.model:
        details.append(f"model: {m.turn.model}")
    if any(details):
        _para(pdf, " · ".join(d for d in details if d), size=8, color=_MUTED)
    if m.citations:
        pdf.ln(1)
        _para(pdf, "References", size=9, style="B")
        for c in m.citations:
            _reference(pdf, c)
    pdf.ln(2)
    y = pdf.get_y()
    pdf.set_draw_color(*_MUTED)
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
    pdf.ln(2)


def export_pdf(conversation: Conversation) -> bytes:
    title = conversation.title or DEFAULT_TITLE
    pdf = _new_pdf(title)
    _para(pdf, title, size=16, style="B")
    date = (conversation.created_at or conversation.exported_at).astimezone(UTC)
    exported = conversation.exported_at.astimezone(UTC)
    _para(
        pdf,
        f"Conversation of {date:%Y-%m-%d} · exported {exported:%Y-%m-%d %H:%M} UTC",
        size=9,
        color=_MUTED,
    )
    pdf.ln(2)
    if not conversation.messages:
        _para(pdf, "This conversation has no messages yet.")
    for m in conversation.messages:
        _message(pdf, m)
    return bytes(pdf.output())
