"""Small, dependency-free helpers for scanning raw LaTeX source.

These work on character offsets so that callers (parser, injection scanner) can map
positions back to the original files. They are deliberately tolerant: arXiv sources
are frequently unbalanced or use odd constructs, so helpers return ``None`` instead of raising.
"""

from __future__ import annotations

import re

# A '%' starts a comment unless escaped by an odd number of backslashes (\% is a literal).
_COMMENT_RE = re.compile(r"(?<!\\)((?:\\\\)*)%[^\n]*")

# \begin{comment}...\end{comment} (comment/verbatim packages) is not typeset either.
COMMENT_ENV_RE = re.compile(r"\\begin\s*\{comment\}.*?(?:\\end\s*\{comment\}|\Z)", re.S)

# Upper bound for brace matching so a single unbalanced '{' cannot make us scan megabytes.
MAX_GROUP_CHARS = 200_000


def _percent_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start() + len(m.group(1)), m.end()) for m in _COMMENT_RE.finditer(text)]


def _blank(text: str, spans: list[tuple[int, int]]) -> str:
    """Replace ``spans`` with spaces, keeping newlines so offsets and lines stay valid."""
    if not spans:
        return text
    parts: list[str] = []
    last = 0
    for start, end in spans:
        if start < last:
            continue
        parts.append(text[last:start])
        parts.append(re.sub(r"[^\n]", " ", text[start:end]))
        last = end
    parts.append(text[last:])
    return "".join(parts)


def comment_spans(text: str) -> list[tuple[int, int]]:
    """Spans of ``%`` comments (start at the '%') and ``comment`` environments."""
    spans = _percent_spans(text)
    without_percent = _blank(text, spans)
    spans += [(m.start(), m.end()) for m in COMMENT_ENV_RE.finditer(without_percent)]
    return sorted(spans)


def mask_comments(text: str) -> str:
    """Blank out comments with spaces, keeping all offsets and newlines intact."""
    return _blank(text, comment_spans(text))


def match_brace(text: str, open_idx: int, limit: int = MAX_GROUP_CHARS) -> int | None:
    """Given ``text[open_idx] == '{'``, return the index of the matching '}' (or None)."""
    if open_idx >= len(text) or text[open_idx] != "{":
        return None
    depth = 0
    i = open_idx
    end = min(len(text), open_idx + limit)
    while i < end:
        c = text[i]
        if c == "\\":
            i += 2  # skip escaped char, e.g. \{ \} \\
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def skip_ws(text: str, i: int) -> int:
    while i < len(text) and text[i] in " \t\r\n":
        i += 1
    return i


def read_group(text: str, i: int) -> tuple[int, int, int] | None:
    """Read one mandatory argument starting at ``i`` (whitespace allowed before it).

    Returns ``(content_start, content_end, next_index)``. A braced group yields its inner
    content; an unbraced argument is a single control sequence or character.
    """
    i = skip_ws(text, i)
    if i >= len(text):
        return None
    if text[i] == "{":
        close = match_brace(text, i)
        if close is None:
            return None
        return i + 1, close, close + 1
    if text[i] == "\\":
        m = re.compile(r"\\(?:[a-zA-Z@]+|.)").match(text, i)
        if m:
            return m.start(), m.end(), m.end()
    if text[i] in "}%":
        return None
    return i, i + 1, i + 1


def read_optional(text: str, i: int) -> tuple[int, int, int] | None:
    """Read an optional ``[...]`` argument at ``i``; returns (start, end, next) or None."""
    j = skip_ws(text, i)
    if j >= len(text) or text[j] != "[":
        return None
    depth = 0
    k = j
    end = min(len(text), j + 2000)
    while k < end:
        c = text[k]
        if c == "\\":
            k += 2
            continue
        if c in "[{":
            depth += 1
        elif c in "]}":
            depth -= 1
            if depth == 0:
                return j + 1, k, k + 1
        k += 1
    return None


def enclosing_group_end(text: str, i: int, limit: int = 5000) -> int:
    """Index where the group enclosing position ``i`` closes.

    Stops at the first unmatched '}', at an ``\\end{`` that closes the surrounding
    environment, at a paragraph-level sectioning command, or after ``limit`` characters.
    Used for declarations like ``\\color{white}`` whose scope is the rest of the group.
    """
    depth = 0
    k = i
    end = min(len(text), i + limit)
    while k < end:
        c = text[k]
        if c == "\\":
            if depth == 0 and (
                text.startswith("\\end{", k)
                or text.startswith("\\section", k)
                or text.startswith("\\subsection", k)
            ):
                return k
            k += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            if depth == 0:
                return k
            depth -= 1
        k += 1
    return end


def line_of(text: str, offset: int) -> int:
    """1-based line number of ``offset`` in ``text``."""
    return text.count("\n", 0, max(0, offset)) + 1
