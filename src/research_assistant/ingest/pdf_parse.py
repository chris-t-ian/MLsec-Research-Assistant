"""PDF fallback parser (pdfplumber / pdfminer.six) for papers without usable LaTeX source.

pdfminer's layout analysis groups characters into text boxes (≈ paragraphs, kept apart
per column) and lines. Headings are detected by font size: a short line set noticeably
larger than the body font, or a bold line at body size that starts with a section number
("3 Method", "4.2 Results"). If no headings are found, the whole text is one section
titled "body". Content under a "References"/"Bibliography" heading is dropped. Rotated
text (e.g. the arXiv side stamp) is skipped.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
from pdfminer.layout import LTChar, LTTextBoxHorizontal, LTTextLine

from research_assistant.models import ParsedDocument, ParsedSection, SectionKind

logger = logging.getLogger(__name__)

HEADING_SIZE_RATIO = 1.15
MAX_HEADING_CHARS = 100
_NUMBERED_RE = re.compile(r"^(?:[A-Z]|\d+)(?:\.\d+)*\.?\s+[A-Z]")
_REFERENCES_RE = re.compile(r"^\s*(?:\d+\.?\s*)?(references?|bibliography)\s*$", re.I)
_ABSTRACT_RE = re.compile(r"^\s*abstract\s*[.:—-]?\s*$", re.I)
_BOLD_RE = re.compile(r"bold|black|heavy|semibold|\.b$|-b$|cmbx", re.I)


class PdfParseError(RuntimeError):
    """The PDF could not be opened or contains no text."""


@dataclass
class _Line:
    text: str
    size: float
    bold: bool
    block: int


@dataclass
class _Section:
    title: str
    kind: SectionKind
    paragraphs: list[str] = field(default_factory=list)


def normalize(text: str) -> str:
    """NFKC (expands ligatures like 'ﬁ'), collapse whitespace."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def _line_chars(line: LTTextLine) -> list[LTChar]:
    return [c for c in line if isinstance(c, LTChar) and c.get_text().strip()]


def _iter_lines(pdf: pdfplumber.PDF) -> Iterator[_Line]:
    block_no = 0
    for page in pdf.pages:
        for element in page.layout:
            if not isinstance(element, LTTextBoxHorizontal):
                continue
            block_no += 1
            for line in element:
                if not isinstance(line, LTTextLine):
                    continue
                chars = _line_chars(line)
                if not chars or sum(c.upright for c in chars) < len(chars) / 2:
                    continue  # empty, or rotated text such as the arXiv side stamp
                text = normalize(line.get_text())
                if not text:
                    continue
                yield _Line(
                    text=text,
                    size=max(c.size for c in chars),
                    bold=all(_BOLD_RE.search(c.fontname) for c in chars),
                    block=block_no,
                )


def _body_size(lines: list[_Line]) -> float:
    counts: Counter[float] = Counter()
    for ln in lines:
        counts[round(ln.size * 2) / 2] += len(ln.text)
    return counts.most_common(1)[0][0] if counts else 10.0


def _is_heading(ln: _Line, body: float) -> bool:
    text = ln.text
    if not (2 <= len(text) <= MAX_HEADING_CHARS) or not re.search(r"[A-Za-z]{2,}", text):
        return False
    if text.endswith((".", ",", ";")) and not _NUMBERED_RE.match(text):
        return False
    if ln.size >= body * HEADING_SIZE_RATIO:
        return True
    return ln.bold and ln.size >= body - 0.5 and bool(_NUMBERED_RE.match(text))


def _join(lines: list[str]) -> str:
    out = ""
    for text in lines:
        if out.endswith("-") and text[:1].islower():
            out = out[:-1] + text  # de-hyphenate line breaks
        else:
            out = f"{out} {text}" if out else text
    return out


def _read_lines(pdf_path: Path) -> list[_Line]:
    try:
        with pdfplumber.open(pdf_path, laparams={}) as pdf:
            return list(_iter_lines(pdf))
    except Exception as exc:  # pdfminer raises many unrelated types for broken files
        raise PdfParseError(f"Cannot read PDF {pdf_path.name} ({type(exc).__name__})") from exc


def parse_pdf(pdf_path: Path, arxiv_id: str) -> ParsedDocument:
    lines = _read_lines(pdf_path)
    if not lines:
        raise PdfParseError(f"No extractable text in {pdf_path.name}")

    body = _body_size(lines)
    sections: list[_Section] = [_Section("front matter", "front_matter")]
    skipping = False
    para: list[str] = []
    prev_block = -1
    prev_heading = False

    def end_para() -> None:
        nonlocal para
        if para and not skipping:
            sections[-1].paragraphs.append(_join(para))
        para = []

    for ln in lines:
        if _is_heading(ln, body):
            end_para()
            if (
                prev_heading
                and not skipping
                and not sections[-1].paragraphs
                and not _NUMBERED_RE.match(ln.text)
            ):
                sections[-1].title += " " + ln.text  # multi-line heading
            elif _REFERENCES_RE.match(ln.text):
                skipping = True
            else:
                skipping = False
                kind: SectionKind = "abstract" if _ABSTRACT_RE.match(ln.text) else "section"
                sections.append(_Section(ln.text, kind))
            prev_heading = True
            prev_block = ln.block
            continue
        prev_heading = False
        if ln.block != prev_block:
            end_para()
            prev_block = ln.block
        para.append(ln.text)
    end_para()

    if len(sections) == 1:
        sections[0].title, sections[0].kind = "body", "body"
    parsed = [
        ParsedSection(title=s.title, kind=s.kind, text="\n\n".join(s.paragraphs))
        for s in sections
        if s.paragraphs
    ]
    if not parsed:
        raise PdfParseError(f"No text sections in {pdf_path.name}")
    return ParsedDocument(arxiv_id=arxiv_id, source="pdf", sections=parsed)
