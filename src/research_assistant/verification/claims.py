"""Split a cited answer into sentences and turn them into checkable claims.

The splitter is written for answers about scientific papers, where a period often does not
end a sentence: "Carlini et al. show", "e.g. LiRA", "Eq. (3)", "Fig. 2", "0.5", "1e-5",
"N. Carlini", inline math ``$\\epsilon = 1.$`` or code. A sentence ends at ``.``, ``!``,
``?`` or ``…`` that is followed by whitespace and a sentence start (uppercase letter, digit,
quote, bracket, math, emphasis, Greek letter) or by the end of its block. Citation markers
right after the period (``... membership. [1][2] Next``) stay with their sentence. Markdown
structure is respected: list items, headings, table rows and blank lines always start a new
block, and fenced code blocks are skipped.

Each sentence becomes a :class:`Claim`:

- ``cited``: the sentence carries ``[n]`` source markers, or it has none and a later sentence
  of the same paragraph (or list item) does: answers often cite once at the end of a run of
  sentences, so such a sentence is checked against the sources of the next cited sentence;
- ``uncited``: no markers in the sentence or after it in its paragraph, but it reads like a
  factual statement;
- sentences citing only tool results (``[Tn]``) or leading up to one, headings, table rows,
  questions, list intros ending in ":", very short fragments and meta sentences ("the
  sources do not cover ...", "I could not find ...") are not claims and are left out.

Spans (``start``/``end``) are offsets into the analysed text (``Answer.raw_text``).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from research_assistant.models import ClaimCheck, ClaimStatus

BlockKind = Literal["paragraph", "list", "heading", "table", "quote"]
ClaimKind = Literal["cited", "uncited"]

# Tokens (lowercased, without the final period) after which a period does not end a sentence.
ABBREVIATIONS = frozenset(
    {
        "al",  # et al.
        "e.g",
        "i.e",
        "cf",
        "vs",
        "viz",
        "approx",
        "resp",
        "incl",
        "eq",
        "eqs",
        "eqn",
        "eqns",
        "fig",
        "figs",
        "sec",
        "secs",
        "sect",
        "tab",
        "tabs",
        "ref",
        "refs",
        "thm",
        "lem",
        "prop",
        "cor",
        "def",
        "defn",
        "alg",
        "algo",
        "app",
        "appx",
        "ch",
        "chap",
        "no",
        "nos",
        "vol",
        "pp",
        "p",
        "ed",
        "eds",
        "dr",
        "prof",
        "mr",
        "ms",
        "st",
        "jr",
        "sr",
        "et",
        "u.s",
        "w.r.t",
        "a.k.a",
        "w.l.o.g",
        "s.t",
        "ca",
    }
)

_TERMINATORS = ".!?…"
# Characters that may close a sentence after its terminator: brackets, quotes, emphasis.
_CLOSERS = ")]\"'\u201d\u2019*_"
# Characters that may open a sentence besides letters and digits.
_OPENERS = "\"'\u201c\u2018([*_$`\\"

# Inline math and code: never split inside.
_PROTECTED = re.compile(
    r"\$\$.+?\$\$|(?<!\\)\$[^$\n]+?(?<!\\)\$|\\\(.+?\\\)|\\\[.+?\\\]|`[^`\n]+`", re.DOTALL
)
# A run of citation markers after a terminator, e.g. " [1][2]" or "[T1]".
_MARKER_RUN = re.compile(r"(?:[ \t]*\[(?:T?[1-9]\d*)(?:\s*,\s*T?[1-9]\d*)*\])+")
_SOURCE_GROUP = re.compile(r"\[\s*([1-9]\d*(?:\s*,\s*[1-9]\d*)*)\s*\]")
_TOOL_MARKER = re.compile(r"\[T[1-9]\d*\]")
_ANY_MARKER = re.compile(r"\s*\[(?:T?[1-9]\d*)(?:\s*,\s*T?[1-9]\d*)*\]")
_TOKEN_BEFORE = re.compile(r"([A-Za-z][A-Za-z.]*)$")

# Block prefixes at the start of a line.
_HEADING = re.compile(r"[ \t]{0,3}#{1,6}[ \t]+")
_LIST = re.compile(r"[ \t]*(?:[-*+•]|\d{1,3}[.)])[ \t]+")
_QUOTE = re.compile(r"[ \t]*>[ \t]?")
_TABLE = re.compile(r"[ \t]*\|")
_FENCE = re.compile(r"[ \t]*(```|~~~)")

_WORD = re.compile(r"[A-Za-zÀ-ɏͰ-Ͽ]+(?:[-'][A-Za-z]+)*|\d+(?:\.\d+)?")
MIN_CLAIM_WORDS = 4
# A sentence that is bold as a whole works as a heading, not as a claim.
_BOLD_LINE = re.compile(r"(\*\*|__)(?:(?!\1).)+\1[.:]?")

# Sentences about the answer or the sources rather than about the research itself.
_META_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:the|these|those|provided|retrieved|available|indexed|given|cited)\s+"
        r"(?:sources?|papers?|passages?|excerpts?|documents?|knowledge base)\b"
        r".*\b(?:do|does|did)\s*(?:not|n't)\b",
        r"\bnot\s+(?:covered|addressed|discussed|mentioned|included|specified|reported|stated|"
        r"provided|described|available)\b.*\b(?:sources?|papers?|passages?|excerpts?|"
        r"knowledge base|corpus)\b",
        r"\b(?:sources?|knowledge base|corpus)\s+(?:do|does)\s*(?:not|n't)\s+"
        r"(?:cover|contain|mention|address|discuss|include|say|provide|describe|report)",
        r"\bI(?:'m|'ve| am| have| can| could| cannot|can't| did| do| don't| found| would| "
        r"will| suggest| recommend)\b",
        r"\barxiv search\b|\buse the arxiv\b|\badd (?:relevant |more )?papers\b",
        r"\b(?:let me know|feel free|if you (?:want|would like|like|need))\b",
        r"\bwere removed because\b|\bwas removed because\b",
    )
]


class Sentence(BaseModel):
    model_config = ConfigDict(frozen=True)

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str
    block: BlockKind = "paragraph"
    block_start: int = Field(default=0, ge=0, description="Start of the enclosing block.")


class Claim(BaseModel):
    """A sentence to verify: ``index`` is its claim number (C1, C2, ...)."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=1)
    sentence: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    kind: ClaimKind
    cited: list[int] = Field(default_factory=list)

    @property
    def text_without_markers(self) -> str:
        return strip_markers(self.sentence)


# --------------------------------------------------------------------------- blocks


def _line_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of every line, end excluding the newline."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for line in text.split("\n"):
        spans.append((pos, pos + len(line)))
        pos += len(line) + 1
    return spans


def _blocks(text: str) -> list[tuple[int, int, BlockKind]]:
    """Markdown blocks as (content start, end, kind); fenced code is dropped.

    A block is one heading, table row or list item, or a run of non-blank paragraph lines
    (soft-wrapped lines are joined). The content start skips the line prefix ("- ", "1. ",
    "## ", "> ").
    """
    blocks: list[tuple[int, int, BlockKind]] = []
    open_block: list[int | BlockKind] | None = None  # [start, end, kind]
    in_fence = False

    def close() -> None:
        nonlocal open_block
        if open_block is not None:
            start, end, kind = open_block
            blocks.append((int(start), int(end), kind))  # type: ignore[arg-type]
            open_block = None

    for ls, le in _line_spans(text):
        line = text[ls:le]
        if _FENCE.match(line):
            close()
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not line.strip():
            close()
            continue
        if m := _HEADING.match(line):
            close()
            blocks.append((ls + m.end(), le, "heading"))
            continue
        if _TABLE.match(line):
            close()
            blocks.append((ls, le, "table"))
            continue
        if m := _LIST.match(line):
            close()
            open_block = [ls + m.end(), le, "list"]
            continue
        if m := _QUOTE.match(line):
            if open_block is None or open_block[2] != "quote":
                close()
                open_block = [ls + m.end(), le, "quote"]
            else:
                open_block[1] = le
            continue
        if open_block is not None and open_block[2] in ("paragraph", "list"):
            open_block[1] = le  # soft-wrapped continuation line
            continue
        close()
        open_block = [ls, le, "paragraph"]
    close()
    return blocks


# --------------------------------------------------------------------------- sentences


def _protected_mask(text: str) -> list[bool]:
    mask = [False] * len(text)
    for m in _PROTECTED.finditer(text):
        for i in range(m.start(), m.end()):
            mask[i] = True
    return mask


def _bracket_depth(text: str, start: int, end: int, mask: Sequence[bool]) -> list[int]:
    """Nesting depth of matched ()/[] pairs per position; unmatched brackets are ignored."""
    depth = [0] * (end - start)
    stack: list[tuple[str, int]] = []
    pairs: list[tuple[int, int]] = []
    for i in range(start, end):
        if mask[i]:
            continue
        ch = text[i]
        if ch in "([":
            stack.append((ch, i))
        elif ch in ")]":
            want = "(" if ch == ")" else "["
            for k in range(len(stack) - 1, -1, -1):
                if stack[k][0] == want:
                    pairs.append((stack[k][1], i))
                    del stack[k:]
                    break
    for a, b in pairs:
        for i in range(a + 1, b):
            depth[i - start] += 1
    return depth


def _is_abbreviation(text: str, block_start: int, i: int) -> bool:
    """The period at ``i`` ends an abbreviation or an initial ("et al.", "Eq.", "N.")."""
    m = _TOKEN_BEFORE.search(text, block_start, i)
    if m is None:
        return False
    token = m.group(1).lower()
    if token in ABBREVIATIONS:
        return True
    letters = token.replace(".", "")
    # Single-letter initials ("N. Carlini") and dotted acronyms ("U.S.").
    return len(letters) == 1 or (len(letters) <= 4 and "." in token)


def _starts_sentence(ch: str) -> bool:
    if ch.isdigit() or ch in _OPENERS:
        return True
    if "Ͱ" <= ch <= "Ͽ":  # Greek letters (epsilon, delta, ...)
        return True
    return ch.isalpha() and not ch.islower()


def _sentence_end(text: str, i: int, end: int, block_start: int) -> int | None:
    """If the terminator at ``i`` ends a sentence, the offset right after the sentence
    (terminators, closers and trailing citation markers included); else None."""
    ch = text[i]
    j = i + 1
    if ch == "." and j < end and text[j] not in _TERMINATORS:
        nxt = text[j]
        if not (nxt.isspace() or nxt in _CLOSERS or nxt == "["):
            return None  # "0.5", "arXiv.org", "e.g", ...
    if ch == "." and _is_abbreviation(text, block_start, i):
        return None
    while j < end and text[j] in _TERMINATORS:
        j += 1
    while j < end and text[j] in _CLOSERS:
        j += 1
    if m := _MARKER_RUN.match(text, j, end):
        j = m.end()
        while j < end and text[j] in _TERMINATORS + _CLOSERS:
            j += 1
    k = j
    while k < end and text[k].isspace():
        k += 1
    if k >= end:
        return j
    if k == j:
        return None  # no whitespace after the terminator
    return j if _starts_sentence(text[k]) else None


def split_sentences(text: str) -> list[Sentence]:
    """Sentences of ``text`` (markdown answer with citation markers), in order."""
    mask = _protected_mask(text)
    out: list[Sentence] = []

    def emit(a: int, b: int, kind: BlockKind, block_start: int) -> None:
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if a < b:
            out.append(
                Sentence(start=a, end=b, text=text[a:b], block=kind, block_start=block_start)
            )

    for bstart, bend, kind in _blocks(text):
        if kind in ("heading", "table"):
            emit(bstart, bend, kind, bstart)
            continue
        depth = _bracket_depth(text, bstart, bend, mask)
        sent_start = bstart
        i = bstart
        while i < bend:
            if mask[i] or depth[i - bstart] > 0:
                i += 1
                continue
            if text[i] in _TERMINATORS:
                stop = _sentence_end(text, i, bend, bstart)
            elif text[i] in ")]" and i > bstart and text[i - 1] in _TERMINATORS:
                # A parenthetical sentence: "(This holds for all x.) Next ..."
                stop = _sentence_end(text, i - 1, bend, bstart)
            else:
                i += 1
                continue
            if stop is None:
                i += 1
                continue
            emit(sent_start, stop, kind, bstart)
            sent_start = i = stop
        emit(sent_start, bend, kind, bstart)
    return out


# --------------------------------------------------------------------------- claims


def strip_markers(sentence: str) -> str:
    """The sentence without ``[n]`` / ``[Tn]`` markers (whitespace before them removed)."""
    return _ANY_MARKER.sub("", sentence).strip()


def cited_numbers(sentence: str) -> list[int]:
    """Source numbers cited in ``sentence``, in order of first appearance."""
    seen: list[int] = []
    for m in _SOURCE_GROUP.finditer(sentence):
        for n in (int(x) for x in m.group(1).split(",")):
            if n not in seen:
                seen.append(n)
    return seen


def is_factual(sentence: Sentence) -> bool:
    """Heuristic: does an unmarked sentence state something that would need a citation?"""
    if sentence.block in ("heading", "table"):
        return False
    if _BOLD_LINE.fullmatch(sentence.text.strip()):
        return False  # a bold run-in heading such as "**Formal link to DP.**"
    plain = strip_markers(sentence.text).strip().rstrip(_CLOSERS).strip()
    if not plain or plain.endswith(("?", ":")):
        return False
    if len(_WORD.findall(plain)) < MIN_CLAIM_WORDS:
        return False
    return not any(p.search(plain) for p in _META_PATTERNS)


def _has_marker(sentence: Sentence) -> bool:
    return _ANY_MARKER.search(sentence.text) is not None


def _next_marked(sentences: Sequence[Sentence], i: int) -> Sentence | None:
    """The first sentence after ``sentences[i]`` in the same block that carries a marker."""
    s = sentences[i]
    for t in sentences[i + 1 :]:
        if t.block_start != s.block_start:
            return None
        if _has_marker(t):
            return t
    return None


def extract_claims(text: str) -> list[Claim]:
    """Claims of an answer: cited sentences and uncited factual sentences, numbered 1..n.

    A sentence without markers takes the sources of the next cited sentence in its block.
    Sentences that cite only tool results, or lead up to one, are not claims about papers and
    are skipped.
    """
    claims: list[Claim] = []
    sentences = split_sentences(text)
    for i, s in enumerate(sentences):
        numbers = cited_numbers(s.text)
        if numbers:
            kind: ClaimKind = "cited"
        elif _TOOL_MARKER.search(s.text) or not is_factual(s):
            continue
        elif (nxt := _next_marked(sentences, i)) is not None:
            numbers = cited_numbers(nxt.text)
            if not numbers:
                continue  # leads up to a tool result
            kind = "cited"
        else:
            kind = "uncited"
        claims.append(
            Claim(
                index=len(claims) + 1,
                sentence=s.text,
                start=s.start,
                end=s.end,
                kind=kind,
                cited=numbers,
            )
        )
    return claims


def badge_positions(text: str, checks: Sequence[ClaimCheck]) -> dict[int, int]:
    """Where to show each claim's badge in ``text`` (claim index -> offset).

    A sentence with its own markers gets its badge right after it. A sentence that borrows
    the sources of a later sentence gets it after that sentence (next to the citation), and
    an uncited sentence after the last sentence of its paragraph. Checks whose span no longer
    matches ``text`` are left out.
    """
    sentences = split_sentences(text)
    by_start = {s.start: i for i, s in enumerate(sentences)}
    out: dict[int, int] = {}
    for c in checks:
        if c.end > len(text) or text[c.start : c.end] != c.sentence:
            continue
        i = by_start.get(c.start)
        if i is None or sentences[i].end != c.end or _has_marker(sentences[i]):
            out[c.index] = c.end
            continue
        nxt = _next_marked(sentences, i)
        if nxt is not None:
            out[c.index] = nxt.end
            continue
        last = sentences[i]
        for t in sentences[i + 1 :]:
            if t.block_start != last.block_start:
                break
            last = t
        out[c.index] = last.end
    return out


def borrows_citation(check: ClaimCheck) -> bool:
    """True if the claim was checked against the sources of a later sentence."""
    return bool(check.cited) and not cited_numbers(check.sentence)


# --------------------------------------------------------------------------- summaries


def supported_rate(checks: Sequence[ClaimCheck]) -> float | None:
    """Supported claims / all claims (cited + uncited); partial counts as not supported."""
    if not checks:
        return None
    return sum(1 for c in checks if c.status == "supported") / len(checks)


def status_counts(checks: Sequence[ClaimCheck]) -> dict[ClaimStatus, int]:
    counts: dict[ClaimStatus, int] = {
        "supported": 0,
        "partially_supported": 0,
        "unsupported": 0,
        "uncited": 0,
    }
    for c in checks:
        counts[c.status] += 1
    return counts


def summary_line(checks: Sequence[ClaimCheck]) -> str:
    """e.g. ``12 of 13 claims supported (1 partially)``."""
    if not checks:
        return "No claims to check."
    counts = status_counts(checks)
    n = len(checks)
    head = f"{counts['supported']} of {n} claim{'s' if n != 1 else ''} supported"
    extra = []
    if counts["partially_supported"]:
        extra.append(f"{counts['partially_supported']} partially")
    if counts["unsupported"]:
        extra.append(f"{counts['unsupported']} unsupported")
    if counts["uncited"]:
        extra.append(f"{counts['uncited']} without citation")
    return head + (f" ({', '.join(extra)})" if extra else "")


__all__ = [
    "ABBREVIATIONS",
    "Claim",
    "Sentence",
    "badge_positions",
    "borrows_citation",
    "cited_numbers",
    "extract_claims",
    "is_factual",
    "split_sentences",
    "status_counts",
    "strip_markers",
    "summary_line",
    "supported_rate",
]
