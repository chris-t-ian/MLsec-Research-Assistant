"""Split a parsed document into overlapping, token-bounded chunks.

Each section is split on its own (chunks never cross section boundaries), recursively on
paragraph, line, sentence and word boundaries until every piece fits, then pieces are
merged greedily up to ``max_tokens`` with ~``overlap_tokens`` of overlap. Token counts use
tiktoken (``cl100k_base`` by default) as an approximation for all models.

Chunk ids are ``<arxiv_id>:<section_idx>:<chunk_idx>``. ``char_start``/``char_end`` refer
to :meth:`ParsedDocument.full_text`. Injection findings are mapped onto chunks: chunks
containing high-severity content get ``flags=["quarantined"]``, medium ones ``["suspicious"]``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import tiktoken

from research_assistant.config import ChunkingConfig
from research_assistant.models import Chunk, Finding, Paper, ParsedDocument

logger = logging.getLogger(__name__)

TokenCounter = Callable[[str], int]

SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "; ", ", ", " ")
QUARANTINED = "quarantined"
SUSPICIOUS = "suspicious"
# Findings that never reach chunk text (comments are stripped during conversion).
_REPORT_ONLY_TYPES = {"comment_instruction"}
_PAYLOAD_MATCH_WORDS = 12


@lru_cache(maxsize=4)
def tiktoken_counter(encoding: str = "cl100k_base") -> TokenCounter:
    enc = tiktoken.get_encoding(encoding)
    return lambda text: len(enc.encode(text, disallowed_special=()))


# --------------------------------------------------------------------------- splitting


def _split_spans(
    text: str, start: int, end: int, max_tokens: int, count: TokenCounter, seps: Sequence[str]
) -> list[tuple[int, int]]:
    """Recursively split ``text[start:end]`` into contiguous spans of <= max_tokens."""
    if count(text[start:end]) <= max_tokens:
        return [(start, end)]
    for i, sep in enumerate(seps):
        if sep not in text[start:end]:
            continue
        pieces: list[tuple[int, int]] = []
        pos = start
        while pos < end:
            hit = text.find(sep, pos, end)
            cut = end if hit == -1 else hit + len(sep)  # separator stays with the left piece
            pieces.append((pos, cut))
            pos = cut
        out: list[tuple[int, int]] = []
        for s, e in pieces:
            out.extend(_split_spans(text, s, e, max_tokens, count, seps[i + 1 :]))
        return out
    # No separator left: hard split by characters, proportional to the token count.
    n = -(-count(text[start:end]) // max_tokens)
    step = max(1, -(-(end - start) // n))
    return [(s, min(s + step, end)) for s in range(start, end, step)]


def split_text(
    text: str, max_tokens: int, overlap_tokens: int, count: TokenCounter
) -> list[tuple[int, int]]:
    """Return ``(start, end)`` offsets of overlapping chunks covering ``text``."""
    pieces = _split_spans(text, 0, len(text), max_tokens, count, SEPARATORS)
    sizes = [count(text[s:e]) for s, e in pieces]
    chunks: list[tuple[int, int]] = []
    window: list[int] = []  # indices into pieces
    total = 0
    for i, size in enumerate(sizes):
        if window and total + size > max_tokens:
            chunks.append((pieces[window[0]][0], pieces[window[-1]][1]))
            # Keep a tail of the window as overlap, as long as it leaves room for piece i.
            while window and (total > overlap_tokens or total + size > max_tokens):
                total -= sizes[window.pop(0)]
        window.append(i)
        total += size
    if window:
        chunks.append((pieces[window[0]][0], pieces[window[-1]][1]))

    # Trim surrounding whitespace so offsets point at real text.
    trimmed: list[tuple[int, int]] = []
    for s, e in chunks:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            trimmed.append((s, e))
    return trimmed


# --------------------------------------------------------------------------- flags


def _payload_regex(payload: str) -> re.Pattern[str] | None:
    words = re.findall(r"\w+", payload)[:_PAYLOAD_MATCH_WORDS]
    if not words:
        return None
    return re.compile(r"\W+".join(map(re.escape, words)), re.I)


def _finding_flag(f: Finding) -> str | None:
    if f.type in _REPORT_ONLY_TYPES:
        return None
    return {"high": QUARANTINED, "medium": SUSPICIOUS}.get(f.severity)


@dataclass(frozen=True)
class _Region:
    """Part of the full text affected by ``finding``; ``section`` set = whole section."""

    start: int
    end: int
    section: int | None
    flag: str
    finding: Finding

    def hits(self, section_idx: int, start: int, end: int) -> bool:
        if self.section is not None:
            return self.section == section_idx
        return self.start < end and self.end > start


def _flag_regions(doc: ParsedDocument, findings: Sequence[Finding]) -> list[_Region]:
    """Map findings to the regions of :meth:`ParsedDocument.full_text` they affect."""
    text = doc.full_text()
    offsets = doc.section_offsets()
    regions: list[_Region] = []
    for f in findings:
        flag = _finding_flag(f)
        if flag is None:
            continue
        if f.section_idx is not None and f.section_idx < len(doc.sections):
            lo = offsets[f.section_idx]
            hi = lo + len(doc.sections[f.section_idx].text)
        else:
            lo, hi = 0, len(text)
        pattern = _payload_regex(f.payload)
        matches = list(pattern.finditer(text, lo, hi)) if pattern else []
        if matches:
            regions.extend(_Region(m.start(), m.end(), None, flag, f) for m in matches)
        elif f.section_idx is not None:
            # Content was transformed beyond recognition: flag the whole section.
            regions.append(_Region(lo, hi, f.section_idx, flag, f))
        else:
            logger.info("Finding %s at %s not located in text", f.type, f.location)
    return regions


# --------------------------------------------------------------------------- chunks


def chunk_document(
    doc: ParsedDocument,
    paper: Paper | None = None,
    findings: Sequence[Finding] = (),
    config: ChunkingConfig | None = None,
    count_tokens: TokenCounter | None = None,
) -> list[Chunk]:
    cfg = config or ChunkingConfig()
    count = count_tokens or tiktoken_counter(cfg.encoding)
    regions = _flag_regions(doc, findings)
    chunks: list[Chunk] = []
    for s_idx, (section, base) in enumerate(zip(doc.sections, doc.section_offsets(), strict=True)):
        spans = split_text(section.text, cfg.max_tokens, cfg.overlap_tokens, count)
        for c_idx, (s, e) in enumerate(spans):
            start, end = base + s, base + e
            flags = {r.flag for r in regions if r.hits(s_idx, start, end)}
            chunks.append(
                Chunk(
                    id=f"{doc.arxiv_id}:{s_idx}:{c_idx}",
                    arxiv_id=doc.arxiv_id,
                    title=paper.title if paper else "",
                    authors=paper.authors if paper else [],
                    year=paper.year if paper else None,
                    section=section.title,
                    text=section.text[s:e],
                    char_start=start,
                    char_end=end,
                    flags=[QUARANTINED] if QUARANTINED in flags else sorted(flags),
                )
            )
    return chunks


def chunk_findings(
    doc: ParsedDocument, chunks: Sequence[Chunk], findings: Sequence[Finding]
) -> dict[str, list[Finding]]:
    """For each flagged chunk id, the findings that caused its flags."""
    regions = _flag_regions(doc, findings)
    out: dict[str, list[Finding]] = {}
    for c in chunks:
        section_idx = int(c.id.rsplit(":", 2)[1])
        hits: list[Finding] = []
        for r in regions:
            if r.hits(section_idx, c.char_start, c.char_end) and r.finding not in hits:
                hits.append(r.finding)
        if hits:
            out[c.id] = hits
    return out
