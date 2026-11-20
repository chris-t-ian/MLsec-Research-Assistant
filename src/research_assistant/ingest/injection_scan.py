"""Detect hidden or AI-directed content in paper sources BEFORE conversion to plain text.

Once LaTeX is converted to text, colour, size and visibility are gone, so hidden prompt
injections ("white text") become indistinguishable from real content. This module scans:

* LaTeX (flattened, comments masked): white text (``\\color``/``\\textcolor``, incl. colours
  defined as white via ``\\definecolor``), ``\\phantom``, tiny ``\\fontsize``, near-zero
  ``\\scalebox``/``\\resizebox``, and calls of user macros whose definition hides content.
* LaTeX comments: instruction-like text aimed at AI systems.
* PDF (pdfplumber): characters coloured like the background underneath, or tiny.
* Visible text in both: phrases aimed at AI systems and chat/system-prompt markup.

Severity: hidden content that addresses an AI is ``high`` (the chunk is quarantined);
visible AI-directed text, instruction-like comments and longer hidden text are ``medium``;
short hidden fragments (e.g. ``\\phantom{0}`` table alignment) are ``low``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pdfplumber
import pdfplumber.page

from research_assistant.ingest.latex_parse import FlatSource, latex_to_text
from research_assistant.ingest.latex_utils import (
    comment_spans,
    enclosing_group_end,
    mask_comments,
    read_group,
    read_optional,
)
from research_assistant.ingest.pdf_parse import normalize
from research_assistant.models import Finding, FindingType, Severity

logger = logging.getLogger(__name__)

SNIPPET_CHARS = 160
MIN_LATEX_FONT_PT = 4.0
MIN_PDF_FONT_PT = 2.0
MIN_SCALE = 0.1
MIN_ABS_DIM_PT = 1.0
MIN_REL_DIM = 0.02
WHITE_TOLERANCE = 0.06  # per-channel distance in [0, 1] counted as "same colour"
MEDIUM_HIDDEN_WORDS = 8

# Phrases addressed to AI systems (reviewers' LLMs, RAG assistants, ...).
AI_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.I)
    for p in (
        r"\b(?:ignore|disregard|forget|override)\s+(?:all\s+|any\s+)?(?:the\s+|your\s+)?"
        r"(?:previous|prior|above|earlier|preceding|former)\s+"
        r"(?:instructions?|prompts?|directions?|context|rules)",
        r"\byou\s+are\s+(?:now\s+)?(?:an?\s+)?(?:AI|artificial\s+intelligence|"
        r"(?:large\s+)?language\s+model|LLM|AI\s+assistant|chatbot)\b",
        r"\bas\s+an?\s+(?:AI|(?:large\s+)?language\s+model|LLM)\b",
        r"\bdo\s+not\s+(?:mention|disclose\s+this|tell\s+the\s+user)\b",
        r"\bgive\s+(?:this\s+paper\s+|it\s+)?(?:a\s+)?(?:very\s+|highly\s+)?positive\s+review",
        r"\b(?:recommend|rate)\s+(?:this\s+(?:paper|submission)\s+)?(?:for\s+)?(?:strong\s+)?accept",
        r"\bpraise\s+(?:this|the)\s+(?:paper|work|submission|authors)",
        r"\b(?:LLM|AI)\s+reviewers?\b",
        r"<\|(?:im_start|im_end|system|user|assistant|endoftext)\|>",
        r"\[/?INST\]|<<\s*/?SYS\s*>>",
        r"</?\s*system\s*>",
        r"(?m)^\s*#{2,}\s*(?:system|instructions?)\b",
        r"\bsystem\s+prompt\s*:",
    )
)

_COLOR_NAMES_WHITE = {"white"}
_DEFINECOLOR_RE = re.compile(r"\\(?:definecolor|colorlet)\s*(?=\{)")
_TEXTCOLOR_RE = re.compile(r"\\textcolor(?![a-zA-Z])")
_COLORBOX_RE = re.compile(r"\\(f?)colorbox(?![a-zA-Z])")
_COLOR_RE = re.compile(r"\\color(?![a-zA-Z])")
_PHANTOM_RE = re.compile(r"\\[hv]?phantom(?![a-zA-Z])")
_FONTSIZE_RE = re.compile(r"\\fontsize(?![a-zA-Z])")
_SCALEBOX_RE = re.compile(r"\\scalebox(?![a-zA-Z])")
_RESIZEBOX_RE = re.compile(r"\\resizebox(?![a-zA-Z])\*?")
_DEF_RE = re.compile(
    r"\\(?:re|provide)?newcommand\*?\s*\{?\s*\\([a-zA-Z]+)\s*\}?\s*(?:\[(\d)\])?\s*(?:\[[^\]]*\])?\s*(?=\{)"
    r"|\\DeclareRobustCommand\*?\s*\{?\s*\\([a-zA-Z]+)\s*\}?\s*(?:\[(\d)\])?\s*(?=\{)"
    r"|\\[gex]?def\s*\\([a-zA-Z]+)\s*((?:#\d)*)\s*(?=\{)"
)
_DIM_RE = re.compile(
    r"^\s*([-+]?\d*\.?\d+)\s*(pt|bp|mm|cm|in|ex|em|sp|pc|px|dd|cc)?\s*(\\[a-zA-Z]+)?\s*$"
)
_PT_PER_UNIT = {
    "pt": 1.0,
    "bp": 1.00375,
    "mm": 2.84528,
    "cm": 28.4528,
    "in": 72.27,
    "ex": 4.3,
    "em": 10.0,
    "sp": 1 / 65536,
    "pc": 12.0,
    "px": 1.0,
    "dd": 1.07,
    "cc": 12.84,
}


def ai_pattern(text: str) -> re.Match[str] | None:
    """First AI-directed phrase in ``text`` (whitespace-normalised), if any."""
    normalized = re.sub(r"[ \t\r\n~]+", " ", text)
    for pattern in AI_PATTERNS:
        m = pattern.search(normalized)
        if m:
            return m
    return None


def severity_for(payload: str, hidden: bool, raw: str = "") -> Severity:
    """Severity of ``payload`` (plain text); ``raw`` source is also checked for AI phrases."""
    if ai_pattern(payload) or (raw and ai_pattern(raw)):
        return "high" if hidden else "medium"
    if hidden and len(re.findall(r"[A-Za-z]{2,}", payload)) >= MEDIUM_HIDDEN_WORDS:
        return "medium"
    return "low"


def _snippet(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= SNIPPET_CHARS else text[: SNIPPET_CHARS - 1] + "…"


# --------------------------------------------------------------------------- colours


def _channels(model: str, spec: str) -> tuple[float, ...] | None:
    # xcolor models are case-sensitive: "RGB" is 0-255, "rgb" is 0-1.
    model = model.strip()
    spec = spec.strip()
    try:
        if model == "RGB":
            return tuple(float(v) / 255 for v in re.split(r"[,\s]+", spec) if v)
        model = model.lower()
        if model in {"rgb", "cmy"}:
            vals = tuple(float(v) for v in re.split(r"[,\s]+", spec) if v)
            return tuple(1 - v for v in vals) if model == "cmy" else vals
        if model == "html":
            spec = spec.lstrip("#")
            return tuple(int(spec[i : i + 2], 16) / 255 for i in (0, 2, 4))
        if model in {"gray", "grey"}:
            return (float(spec),)
        if model == "cmyk":
            c, m, y, k = (float(v) for v in re.split(r"[,\s]+", spec) if v)
            return ((1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k))
    except (ValueError, IndexError):
        return None
    return None


def _is_white(model: str | None, spec: str, white_names: set[str]) -> bool:
    if model:
        ch = _channels(model, spec)
        return ch is not None and len(ch) > 0 and all(v >= 1 - WHITE_TOLERANCE for v in ch)
    # xcolor expressions such as "white!95!black" (95 % white).
    parts = [p.strip() for p in spec.split("!")]
    if parts[0].lower() not in white_names and parts[0] not in white_names:
        return False
    if len(parts) >= 2:
        try:
            pct = float(parts[1])
        except ValueError:
            return False
        other = parts[2].lower() if len(parts) >= 3 else "white"
        return pct >= 95 or other in white_names
    return True


def _white_color_names(masked: str) -> set[str]:
    names = set(_COLOR_NAMES_WHITE)
    for m in _DEFINECOLOR_RE.finditer(masked):
        is_colorlet = masked.startswith("\\colorlet", m.start())
        g1 = read_group(masked, m.end())
        if not g1:
            continue
        name = masked[g1[0] : g1[1]].strip()
        if is_colorlet:
            g2 = read_group(masked, g1[2])
            if g2 and _is_white(None, masked[g2[0] : g2[1]], names):
                names.add(name)
            continue
        g2 = read_group(masked, g1[2])
        g3 = read_group(masked, g2[2]) if g2 else None
        if g2 and g3 and _is_white(masked[g2[0] : g2[1]], masked[g3[0] : g3[1]], names):
            names.add(name)
    return names


def _read_color(masked: str, i: int, white: set[str]) -> tuple[bool, int] | None:
    """Parse ``[model]{spec}`` at ``i``; return (is_white, next_index)."""
    opt = read_optional(masked, i)
    model = masked[opt[0] : opt[1]] if opt else None
    grp = read_group(masked, opt[2] if opt else i)
    if grp is None:
        return None
    return _is_white(model, masked[grp[0] : grp[1]], white), grp[2]


# --------------------------------------------------------------------------- dimensions


def _near_zero_dim(spec: str) -> bool:
    spec = spec.strip()
    if spec == "!" or not spec:
        return False
    m = _DIM_RE.match(spec)
    if not m:
        return False
    value = abs(float(m.group(1)))
    unit, rel = m.group(2), m.group(3)
    if rel:  # e.g. 0.001\textwidth
        return value < MIN_REL_DIM
    if unit is None:
        return value == 0
    return value * _PT_PER_UNIT[unit] < MIN_ABS_DIM_PT


def _to_pt(spec: str) -> float | None:
    m = _DIM_RE.match(spec)
    if not m or m.group(3):
        return None
    return float(m.group(1)) * _PT_PER_UNIT.get(m.group(2) or "pt", 1.0)


# --------------------------------------------------------------------------- LaTeX


@dataclass
class _Hit:
    type: FindingType
    start: int  # command start
    end: int  # end of hidden region
    payload_start: int
    payload_end: int


def _colored_backgrounds(masked: str, white: set[str]) -> list[tuple[int, int]]:
    """Bodies of ``\\colorbox``/``\\fcolorbox`` with a non-white background.

    White text there is visible (e.g. white-on-black prompt boxes), so not hidden.
    """
    spans: list[tuple[int, int]] = []
    for m in _COLORBOX_RE.finditer(masked):
        i = m.end()
        if m.group(1):  # \fcolorbox[model]{frame}{background}{body}
            opt = read_optional(masked, i)
            frame = read_group(masked, opt[2] if opt else i)
            if not frame:
                continue
            bg_opt, bg = opt, read_group(masked, frame[2])
        else:  # \colorbox[model]{background}{body}
            bg_opt = read_optional(masked, i)
            bg = read_group(masked, bg_opt[2] if bg_opt else i)
        body = read_group(masked, bg[2]) if bg else None
        if not (bg and body):
            continue
        model = masked[bg_opt[0] : bg_opt[1]] if bg_opt else None
        if not _is_white(model, masked[bg[0] : bg[1]], white):
            spans.append((body[0], body[1]))
    return spans


def _hidden_hits(masked: str, white: set[str]) -> list[_Hit]:
    hits = _raw_hidden_hits(masked, white)
    backgrounds = _colored_backgrounds(masked, white)
    return [h for h in hits if not (h.type == "hidden_color" and _inside(h.start, backgrounds))]


def _raw_hidden_hits(masked: str, white: set[str]) -> list[_Hit]:
    hits: list[_Hit] = []

    for m in _TEXTCOLOR_RE.finditer(masked):
        color = _read_color(masked, m.end(), white)
        if not color or not color[0]:
            continue
        grp = read_group(masked, color[1])
        if grp:
            hits.append(_Hit("hidden_color", m.start(), grp[2], grp[0], grp[1]))

    for m in _COLOR_RE.finditer(masked):
        color = _read_color(masked, m.end(), white)
        if not color or not color[0]:
            continue
        end = enclosing_group_end(masked, color[1])
        hits.append(_Hit("hidden_color", m.start(), end, color[1], end))

    for m in _PHANTOM_RE.finditer(masked):
        grp = read_group(masked, m.end())
        if grp:
            hits.append(_Hit("phantom", m.start(), grp[2], grp[0], grp[1]))

    for m in _FONTSIZE_RE.finditer(masked):
        size = read_group(masked, m.end())
        skip = read_group(masked, size[2]) if size else None
        if not size or not skip:
            continue
        pt = _to_pt(masked[size[0] : size[1]])
        if pt is not None and pt < MIN_LATEX_FONT_PT:
            end = enclosing_group_end(masked, skip[2])
            hits.append(_Hit("tiny_font", m.start(), end, skip[2], end))

    for m in _SCALEBOX_RE.finditer(masked):
        h = read_group(masked, m.end())
        if not h:
            continue
        v = read_optional(masked, h[2])
        body = read_group(masked, v[2] if v else h[2])
        if not body:
            continue
        factors = [masked[h[0] : h[1]]] + ([masked[v[0] : v[1]]] if v else [])
        try:
            tiny = any(abs(float(f)) < MIN_SCALE for f in factors)
        except ValueError:
            tiny = False
        if tiny:
            hits.append(_Hit("zero_scale", m.start(), body[2], body[0], body[1]))

    for m in _RESIZEBOX_RE.finditer(masked):
        w = read_group(masked, m.end())
        h = read_group(masked, w[2]) if w else None
        body = read_group(masked, h[2]) if h else None
        if not (w and h and body):
            continue
        if _near_zero_dim(masked[w[0] : w[1]]) or _near_zero_dim(masked[h[0] : h[1]]):
            hits.append(_Hit("zero_scale", m.start(), body[2], body[0], body[1]))

    return hits


def _hiding_macros(masked: str, white: set[str]) -> tuple[dict[str, int], list[tuple[int, int]]]:
    """User macros whose body hides content -> number of args; plus definition spans."""
    hiding: dict[str, int] = {}
    def_spans: list[tuple[int, int]] = []
    for m in _DEF_RE.finditer(masked):
        name = m.group(1) or m.group(3) or m.group(5)
        nargs_s = m.group(2) or m.group(4)
        nargs = int(nargs_s) if nargs_s else len(re.findall(r"#\d", m.group(6) or ""))
        body = read_group(masked, m.end())
        if not name or not body:
            continue
        def_spans.append((m.start(), body[2]))
        if _hidden_hits(masked[body[0] : body[1]], white):
            hiding[name] = nargs
    return hiding, def_spans


def _inside(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(s <= pos < e for s, e in spans)


def scan_latex(flat: FlatSource, to_text: Callable[[str], str] = latex_to_text) -> list[Finding]:
    """Scan flattened LaTeX source. Findings carry ``offset`` into ``flat.text``."""
    raw = flat.text
    masked = mask_comments(raw)
    white = _white_color_names(masked)
    hiding, def_spans = _hiding_macros(masked, white)

    hits = [h for h in _hidden_hits(masked, white) if not _inside(h.start, def_spans)]
    if hiding:
        call_re = re.compile(r"\\(" + "|".join(map(re.escape, hiding)) + r")(?![a-zA-Z])")
        for m in call_re.finditer(masked):
            if _inside(m.start(), def_spans):
                continue
            if hiding[m.group(1)] >= 1:
                grp = read_group(masked, m.end())
                if grp:
                    hits.append(_Hit("hidden_macro", m.start(), grp[2], grp[0], grp[1]))
            else:
                end = enclosing_group_end(masked, m.end())
                hits.append(_Hit("hidden_macro", m.start(), end, m.end(), end))

    findings: list[Finding] = []
    hidden_spans: list[tuple[int, int]] = []
    for h in sorted(hits, key=lambda h: h.start):
        payload_latex = masked[h.payload_start : h.payload_end]
        payload = to_text(payload_latex)
        if not payload.strip() and not ai_pattern(payload_latex):
            continue  # e.g. \color{white} with nothing after it in the group
        hidden_spans.append((h.start, h.end))
        findings.append(
            Finding(
                type=h.type,
                snippet=_snippet(raw[h.start : h.end]),
                location=flat.location(h.start),
                severity=severity_for(payload, hidden=True, raw=payload_latex),
                payload=payload,
                offset=h.start,
            )
        )

    # Instruction-like LaTeX comments (stripped later, but worth reporting).
    for start, end in comment_spans(raw):
        comment = raw[start + 1 : end]
        if ai_pattern(comment):
            findings.append(
                Finding(
                    type="comment_instruction",
                    snippet=_snippet(raw[start:end]),
                    location=flat.location(start),
                    severity="medium",
                    payload=comment.strip(),
                    offset=start,
                )
            )

    # Visible AI-directed text (comments masked; hidden regions already reported).
    normalized = re.sub(r"[\r\n~]", " ", masked)
    for pattern in AI_PATTERNS:
        for m in pattern.finditer(normalized):
            if _inside(m.start(), hidden_spans) or _inside(m.start(), def_spans):
                continue
            findings.append(
                Finding(
                    type="ai_directed_text",
                    snippet=_snippet(raw[max(0, m.start() - 40) : m.end() + 40]),
                    location=flat.location(m.start()),
                    severity="medium",
                    payload=m.group(0),
                    offset=m.start(),
                )
            )
    return sorted(findings, key=lambda f: f.offset or 0)


# --------------------------------------------------------------------------- PDF

RGB = tuple[float, float, float]


class PdfScanError(RuntimeError):
    """The PDF could not be scanned."""


def _to_rgb(color: object) -> RGB | None:
    """pdfminer colour (gray / RGB / CMYK tuple) -> RGB in [0, 1]; None if unknown."""
    if isinstance(color, int | float):
        color = (color,)
    if not isinstance(color, tuple | list) or not all(isinstance(v, int | float) for v in color):
        return None  # pattern / unsupported colour space
    vals = [float(v) for v in color]
    if len(vals) == 1:
        return (vals[0], vals[0], vals[0])
    if len(vals) == 3:
        return (vals[0], vals[1], vals[2])
    if len(vals) == 4:
        c, m, y, k = vals
        return ((1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k))
    return None


def _same_color(a: RGB, b: RGB) -> bool:
    return all(abs(x - y) <= WHITE_TOLERANCE for x, y in zip(a, b, strict=True))


@dataclass(frozen=True)
class _Fill:
    x0: float
    top: float
    x1: float
    bottom: float
    rgb: RGB

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.top <= y <= self.bottom


def _page_fills(page: pdfplumber.page.Page) -> list[_Fill]:
    fills: list[_Fill] = []
    for obj in [*page.rects, *page.curves]:
        if not obj.get("fill"):
            continue
        rgb = _to_rgb(obj.get("non_stroking_color"))
        if rgb is not None:
            fills.append(_Fill(obj["x0"], obj["top"], obj["x1"], obj["bottom"], rgb))
    return fills


def _hidden_reason(char: dict[str, Any], fills: list[_Fill]) -> FindingType | None:
    if char.get("size", 10.0) < MIN_PDF_FONT_PT:
        return "pdf_tiny_font"
    color = char.get("non_stroking_color")
    rgb = (0.0, 0.0, 0.0) if color is None else _to_rgb(color)  # PDF default fill is black
    if rgb is None:
        return None
    cx, cy = (char["x0"] + char["x1"]) / 2, (char["top"] + char["bottom"]) / 2
    background: RGB = (1.0, 1.0, 1.0)  # assume a white page
    for fill in fills:  # later fills are drawn on top
        if fill.contains(cx, cy):
            background = fill.rgb
    return "pdf_hidden_color" if _same_color(rgb, background) else None


def _hidden_runs(page: pdfplumber.page.Page) -> list[tuple[FindingType, str]]:
    """Consecutive hidden characters (content-stream order) grouped into text runs."""
    fills = _page_fills(page)
    runs: list[tuple[FindingType, str]] = []
    reason: FindingType | None = None
    buf: list[str] = []
    prev: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal reason, buf
        text = normalize("".join(buf))
        if reason and text:
            runs.append((reason, text))
        reason, buf = None, []

    for char in page.chars:
        text = char.get("text", "")
        if not text.strip():
            if reason:
                buf.append(" ")
            continue
        r = _hidden_reason(char, fills)
        if r != reason:
            flush()
            reason = r
            prev = None
        if r is None:
            continue
        if prev is not None:
            size = char.get("size", 10.0) or 10.0
            if char["x0"] - prev["x1"] > 0.2 * size or abs(char["top"] - prev["top"]) > 0.5 * size:
                buf.append(" ")  # PDFs often position words instead of emitting spaces
        buf.append(text)
        prev = char
    flush()
    return runs


def scan_pdf(pdf_path: Path) -> list[Finding]:
    """Scan a PDF for hidden characters and visible AI-directed text.

    Hidden: drawn in the colour of the filled shape underneath (white page by default) or
    below ``MIN_PDF_FONT_PT``. Not detectable with pdfminer: text render mode 3
    (invisible) and transparency, since neither is exposed per character.
    """
    findings: list[Finding] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_no, page in enumerate(pdf.pages, start=1):
                location = f"page {page_no}"
                hidden: list[str] = []
                for reason, text in _hidden_runs(page):
                    hidden.append(text.lower())
                    findings.append(
                        Finding(
                            type=reason,
                            snippet=_snippet(text),
                            location=location,
                            severity=severity_for(text, hidden=True),
                            payload=text,
                        )
                    )
                page_text = normalize(page.extract_text() or "")
                for pattern in AI_PATTERNS:
                    for hit in pattern.finditer(page_text):
                        if any(hit.group(0).lower() in h for h in hidden):
                            continue  # already reported as hidden
                        findings.append(
                            Finding(
                                type="ai_directed_text",
                                snippet=_snippet(
                                    page_text[max(0, hit.start() - 40) : hit.end() + 40]
                                ),
                                location=location,
                                severity="medium",
                                payload=hit.group(0),
                            )
                        )
    except Exception as exc:  # pdfminer raises many unrelated types for broken files
        raise PdfScanError(f"Cannot scan PDF {pdf_path.name} ({type(exc).__name__})") from exc
    return findings
