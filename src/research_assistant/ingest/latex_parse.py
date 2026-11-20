"""Convert an arXiv LaTeX source tree into a structured plain-text document.

Steps:
1. :func:`find_main_tex` picks the file with ``\\documentclass``.
2. :func:`flatten` resolves ``\\input``/``\\include`` recursively into one string and keeps a
   source map so offsets can be reported as ``file:line`` (used by the injection scanner,
   which must run on this raw text BEFORE conversion).
3. :func:`parse_latex` splits the flattened text into abstract / sections / subsections /
   appendix, drops the bibliography and converts each part with pylatexenc, keeping math
   as inline LaTeX.

Source files are untrusted: includes are confined to the source directory, recursion
depth and file sizes are capped.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from pydantic import BaseModel, Field
from pylatexenc.latex2text import LatexNodes2Text

from research_assistant.ingest.latex_utils import (
    COMMENT_ENV_RE,
    line_of,
    mask_comments,
    read_group,
    read_optional,
)
from research_assistant.models import ParsedDocument, ParsedSection, SectionKind

logger = logging.getLogger(__name__)

MAX_INCLUDE_DEPTH = 20
MAX_FILE_BYTES = 5 * 1024 * 1024
TEX_SUFFIXES = (".tex", ".ltx")

_DOCCLASS_RE = re.compile(r"\\document(?:class|style)(?![a-zA-Z])")
_BEGIN_DOC_RE = re.compile(r"\\begin\s*\{document\}")
_END_DOC_RE = re.compile(r"\\end\s*\{document\}")
_INPUT_RE = re.compile(
    r"\\(?:input|include|subfile)(?![a-zA-Z])\s*(?:\{\s*([^{}]*?)\s*\}|([^\s{}\\%]+))"
)
_HEADING_RE = re.compile(r"\\(section|subsection)(?![a-zA-Z])\s*\*?\s*")
# Template-specific heading commands, e.g. \def\iccvsection{\@startsection{section}...}
# or \newcommand{\mysec}[1]{\section{#1}}.
_HEADING_ALIAS_RE = re.compile(
    r"\\(?:[gex]?def\s*\\([a-zA-Z]+)|(?:re)?newcommand\*?\s*\{?\s*\\([a-zA-Z]+)\s*\}?\s*(?:\[\d\])?)"
    r"\s*\{\s*\\(?:@startsection\s*\{\s*(section|subsection)\s*\}|(section|subsection)\*?\s*\{\s*#1)"
)
ALIAS_SUFFIXES = (".tex", ".ltx", ".sty", ".cls")
_APPENDIX_RE = re.compile(r"\\appendix(?![a-zA-Z])|\\begin\s*\{appendi(?:x|ces)\}")
_ABSTRACT_ENV_RE = re.compile(r"\\begin\s*\{abstract\}(.*?)\\end\s*\{abstract\}", re.DOTALL)
_ABSTRACT_MACRO_RE = re.compile(r"\\abstract\s*(?=\{)")
_BIB_ENV_RE = re.compile(r"\\begin\s*\{thebibliography\}.*?(?:\\end\s*\{thebibliography\}|$)", re.S)
_BIB_CMD_RE = re.compile(
    r"\\(?:bibliography|bibliographystyle|nobibliography|addbibresource)\s*\{[^{}]*\}"
    r"|\\printbibliography(?![a-zA-Z])(?:\s*\[[^\]]*\])?"
)
_REFERENCES_TITLE_RE = re.compile(r"^\s*(references?|bibliography|literature cited)\s*$", re.I)
_SIMPLE_DEF_RE = re.compile(
    r"\\(?:re)?newcommand\*?\s*\{?\s*\\([a-zA-Z]+)\s*\}?\s*(?=\{)|\\def\s*\\([a-zA-Z]+)\s*(?=\{)"
)
_HREF_RE = re.compile(r"\\href(?![a-zA-Z])")
_CONTROL_WORD_RE = re.compile(r"\\([a-zA-Z]+)(?![a-zA-Z])")

# Front-matter macros whose arguments are metadata (title, authors, ...), not body text.
_FRONT_MATTER_MACROS = (
    "title",
    "author",
    "affiliation",
    "affil",
    "institute",
    "institution",
    "address",
    "email",
    "thanks",
    "date",
    "keywords",
    "icmltitle",
    "icmlauthor",
    "icmlaffiliation",
    "icmlcorrespondingauthor",
    "icmlkeywords",
    "ccsdesc",
    "orcid",
    "authornote",
    "IEEEauthorblockN",
    "IEEEauthorblockA",
)
_FRONT_MATTER_ENVS = ("CCSXML", "icmlauthorlist", "IEEEkeywords", "keywords")
_MIN_FRONT_MATTER_CHARS = 200


class LatexParseError(RuntimeError):
    """The LaTeX source could not be parsed into a document."""


# --------------------------------------------------------------------------- source map


class SourceSegment(BaseModel):
    """``flat[start:end]`` was copied verbatim from ``file`` starting at ``file_offset``."""

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    file: str
    file_offset: int = Field(ge=0)


class FlatSource(BaseModel):
    """The main file with all includes inlined, plus a map back to the original files."""

    text: str
    main_file: str
    segments: list[SourceSegment] = Field(default_factory=list)
    files: dict[str, str] = Field(default_factory=dict, repr=False)

    def location(self, offset: int) -> str:
        """``file:line`` for a character offset in :attr:`text`."""
        for seg in self.segments:
            if seg.start <= offset < seg.end or (offset == seg.end == len(self.text)):
                file_text = self.files.get(seg.file, "")
                return f"{seg.file}:{line_of(file_text, seg.file_offset + offset - seg.start)}"
        return f"{self.main_file}:?"


class SectionSpan(BaseModel):
    """Which part of the flattened source produced section ``idx``."""

    idx: int = Field(ge=0)
    start: int = Field(ge=0)
    end: int = Field(ge=0)


class LatexParse(BaseModel):
    document: ParsedDocument
    flat: FlatSource
    spans: list[SectionSpan] = Field(default_factory=list)

    def section_at(self, offset: int) -> int | None:
        for span in self.spans:
            if span.start <= offset < span.end:
                return span.idx
        return None


# --------------------------------------------------------------------------- files


def read_tex(path: Path) -> str:
    data = path.read_bytes()[:MAX_FILE_BYTES]
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _tex_candidates(source_dir: Path) -> list[Path]:
    return sorted(
        p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in TEX_SUFFIXES
    )


def find_main_tex(source_dir: Path) -> Path | None:
    """Return the main LaTeX file (the one containing ``\\documentclass``), or None.

    Preference among several candidates: has ``\\begin{document}``, conventional name
    (main/ms/paper), closer to the root, larger file.
    """
    if not source_dir.is_dir():
        return None
    scored: list[tuple[tuple[int, int, int, int], Path]] = []
    for path in _tex_candidates(source_dir):
        try:
            text = mask_comments(read_tex(path))
        except OSError:
            continue
        if not _DOCCLASS_RE.search(text):
            continue
        has_begin = 1 if _BEGIN_DOC_RE.search(text) else 0
        conventional = 1 if path.stem.lower() in {"main", "ms", "paper", "article"} else 0
        depth = len(path.relative_to(source_dir).parts)
        scored.append(((has_begin, conventional, -depth, len(text)), path))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _resolve_include(name: str, current: Path, root: Path) -> Path | None:
    name = name.strip().strip('"')
    if not name:
        return None
    candidates: list[Path] = []
    for base in (root, current.parent):
        p = base / name
        candidates.append(p)
        if p.suffix.lower() not in TEX_SUFFIXES:
            candidates.append(p.with_name(p.name + ".tex"))
    root_resolved = root.resolve()
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except OSError:
            continue
        if not resolved.is_relative_to(root_resolved):
            logger.debug("Ignoring include candidate outside source dir: %s", cand)
            continue
        if resolved.is_file():
            return resolved
    return None


def flatten(main: Path, root: Path | None = None) -> FlatSource:
    """Inline ``\\input``/``\\include``/``\\subfile`` (ignoring commented-out ones)."""
    root = (root or main.parent).resolve()
    parts: list[str] = []
    segments: list[SourceSegment] = []
    files: dict[str, str] = {}
    pos = 0

    def rel(path: Path) -> str:
        return path.resolve().relative_to(root).as_posix()

    def emit(text: str, file: str, file_offset: int) -> None:
        nonlocal pos
        if not text:
            return
        parts.append(text)
        segments.append(
            SourceSegment(start=pos, end=pos + len(text), file=file, file_offset=file_offset)
        )
        pos += len(text)

    def visit(path: Path, depth: int, stack: tuple[Path, ...]) -> None:
        name = rel(path)
        text = files.setdefault(name, read_tex(path))
        masked = mask_comments(text)
        last, stop = 0, len(text)
        if depth > 0:
            # An included standalone document (e.g. a TikZ figure) contributes only its body,
            # as with the standalone package; its \end{document} must not end the paper.
            begin = _BEGIN_DOC_RE.search(masked)
            if begin:
                last = begin.end()
                end = _END_DOC_RE.search(masked, last)
                stop = end.start() if end else len(text)
        for m in _INPUT_RE.finditer(masked, last, stop):
            emit(text[last : m.start()], name, last)
            last = m.end()
            target = _resolve_include(m.group(1) or m.group(2) or "", path, root)
            if target is None:
                logger.debug("Unresolved include %r in %s", m.group(0), name)
                continue
            if target.suffix.lower() == ".bbl":
                continue  # bibliography is dropped
            if target in stack or depth >= MAX_INCLUDE_DEPTH:
                logger.warning("Skipping recursive/deep include %s in %s", target.name, name)
                continue
            visit(target, depth + 1, (*stack, target))
        emit(text[last:stop], name, last)

    main_resolved = main.resolve()
    visit(main_resolved, 0, (main_resolved,))
    return FlatSource(
        text="".join(parts), main_file=rel(main_resolved), segments=segments, files=files
    )


# --------------------------------------------------------------------------- conversion

_CONVERTER = LatexNodes2Text(math_mode="verbatim")
_POST_SUBS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"<\s*g\s*r\s*a\s*p\s*h\s*i\s*c\s*s\s*>"), ""),
    (re.compile(r"<cit\.>"), "[cite]"),
    (re.compile(r"<ref>"), "[ref]"),
    (re.compile(r"^[ \t]*§(?:\.§)*[ \t]*", re.M), ""),
    # Unwrap source line breaks inside paragraphs (keep list items on their own line).
    (re.compile(r"(?<!\n)[ \t]*\n(?![ \t]*(?:\n|\* ))"), " "),
    (re.compile(r"[ \t\u00a0]+"), " "),
    (re.compile(r" *\n *"), "\n"),
    (re.compile(r"\n{3,}"), "\n\n"),
)


def collect_simple_macros(flat_masked: str) -> dict[str, str]:
    """Zero-argument ``\\newcommand``/``\\def`` macros with short, plain bodies."""
    macros: dict[str, str] = {}
    for m in _SIMPLE_DEF_RE.finditer(flat_masked):
        name = m.group(1) or m.group(2)
        group = read_group(flat_masked, m.end())
        if not name or group is None:
            continue
        body = flat_masked[group[0] : group[1]]
        if "#" in body or len(body) > 200 or "\n\n" in body:
            continue
        macros[name] = re.sub(r"\\xspace(?![a-zA-Z])", "", body).strip()
    return macros


def expand_macros(latex: str, macros: dict[str, str], passes: int = 2) -> str:
    if not macros:
        return latex
    for _ in range(passes):
        changed = False

        def repl(m: re.Match[str]) -> str:
            nonlocal changed
            body = macros.get(m.group(1))
            if body is None:
                return m.group(0)
            changed = True
            return body

        latex = _CONTROL_WORD_RE.sub(repl, latex)
        if not changed:
            break
    return latex


def _remove_bibliography(latex: str) -> str:
    return _BIB_CMD_RE.sub("", _BIB_ENV_RE.sub("", latex))


def _remove_macro_with_args(latex: str, names: tuple[str, ...]) -> str:
    pattern = re.compile(r"\\(" + "|".join(names) + r")(?![a-zA-Z])\*?")
    out: list[str] = []
    last = 0
    for m in pattern.finditer(latex):
        if m.start() < last:
            continue
        i = m.end()
        opt = read_optional(latex, i)
        if opt:
            i = opt[2]
        grp = read_group(latex, i)
        if grp is None or latex[grp[0] - 1 : grp[0]] != "{":
            continue
        out.append(latex[last : m.start()])
        last = grp[2]
    out.append(latex[last:])
    return "".join(out)


def _remove_envs(latex: str, envs: tuple[str, ...]) -> str:
    for env in envs:
        latex = re.sub(
            r"\\begin\s*\{" + re.escape(env) + r"\}.*?\\end\s*\{" + re.escape(env) + r"\}",
            "",
            latex,
            flags=re.S,
        )
    return latex


def _crude_strip(latex: str) -> str:
    text = re.sub(r"(?<!\\)%[^\n]*", "", latex)
    text = re.sub(r"\\[a-zA-Z@]+\*?", " ", text)
    return re.sub(r"[{}]", "", text)


def _rewrite_href(latex: str) -> str:
    """``\\href{url}{text}`` -> ``text`` (pylatexenc 2.11 crashes on \\href)."""
    out: list[str] = []
    last = 0
    for m in _HREF_RE.finditer(latex):
        if m.start() < last:
            continue
        url = read_group(latex, m.end())
        label = read_group(latex, url[2]) if url else None
        if not (url and label):
            continue
        out.append(latex[last : m.start()])
        out.append("{" + latex[label[0] : label[1]] + "}")
        last = label[2]
    out.append(latex[last:])
    return "".join(out)


def _convert(latex: str) -> str:
    try:
        return _CONVERTER.latex_to_text(latex)
    except Exception as exc:  # pylatexenc can fail on exotic input (incl. RecursionError)
        logger.debug("pylatexenc failed (%s)", type(exc).__name__)
    # Retry paragraph by paragraph so one bad construct only degrades its own paragraph.
    parts: list[str] = []
    for para in re.split(r"\n\s*\n", latex):
        try:
            parts.append(_CONVERTER.latex_to_text(para))
        except Exception as exc:
            logger.warning("pylatexenc failed (%s); crude stripping", type(exc).__name__)
            parts.append(_crude_strip(para))
    return "\n\n".join(parts)


def latex_to_text(latex: str, macros: dict[str, str] | None = None) -> str:
    """Convert a LaTeX fragment to plain text; math stays as inline LaTeX."""
    latex = COMMENT_ENV_RE.sub("", latex)
    latex = _rewrite_href(_remove_bibliography(expand_macros(latex, macros or {})))
    text = _convert(latex)
    for pattern, repl in _POST_SUBS:
        text = pattern.sub(repl, text)
    return text.strip()


# --------------------------------------------------------------------------- structure


class _Heading(BaseModel):
    kind: str  # "section" | "subsection"
    start: int  # offset of the backslash
    body_start: int  # first char after the title group
    title: str
    appendix: bool


def find_heading_aliases(source_dir: Path) -> dict[str, str]:
    """Map template-defined heading commands (from .tex/.sty/.cls) to section/subsection."""
    aliases: dict[str, str] = {}
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in ALIAS_SUFFIXES:
            continue
        try:
            text = mask_comments(read_tex(path))
        except OSError:
            continue
        for m in _HEADING_ALIAS_RE.finditer(text):
            name = m.group(1) or m.group(2)
            kind = m.group(3) or m.group(4)
            if name and kind and name not in {"section", "subsection", "subsubsection"}:
                aliases.setdefault(name, kind)
    return aliases


def _heading_regex(aliases: dict[str, str]) -> re.Pattern[str]:
    if not aliases:
        return _HEADING_RE
    names = "|".join(sorted(map(re.escape, {"section", "subsection", *aliases}), key=len)[::-1])
    return re.compile(r"\\(" + names + r")(?![a-zA-Z])\s*\*?\s*")


def _find_headings(
    masked: str,
    start: int,
    end: int,
    macros: dict[str, str],
    aliases: dict[str, str] | None = None,
) -> list[_Heading]:
    aliases = aliases or {}
    appendix_at = [m.start() for m in _APPENDIX_RE.finditer(masked, start, end)]
    first_appendix = appendix_at[0] if appendix_at else None
    headings: list[_Heading] = []
    for m in _heading_regex(aliases).finditer(masked, start, end):
        i = m.end()
        opt = read_optional(masked, i)
        if opt:
            i = opt[2]
        grp = read_group(masked, i)
        if grp is None or masked[grp[0] - 1 : grp[0]] != "{":
            continue
        title = latex_to_text(masked[grp[0] : grp[1]], macros).replace("\n", " ").strip()
        headings.append(
            _Heading(
                kind=aliases.get(m.group(1), m.group(1)),
                start=m.start(),
                body_start=grp[2],
                title=title or "(untitled)",
                appendix=first_appendix is not None and m.start() > first_appendix,
            )
        )
    return headings


def _find_abstract(masked: str) -> tuple[int, int, int, int] | None:
    """Return (span_start, span_end, content_start, content_end) of the abstract."""
    m = _ABSTRACT_ENV_RE.search(masked)
    if m:
        return m.start(), m.end(), m.start(1), m.end(1)
    m = _ABSTRACT_MACRO_RE.search(masked)
    if m:
        grp = read_group(masked, m.end())
        if grp:
            return m.start(), grp[2], grp[0], grp[1]
    return None


def parse_latex(source_dir: Path, arxiv_id: str) -> LatexParse:
    """Parse the LaTeX source in ``source_dir`` into a :class:`LatexParse`."""
    main = find_main_tex(source_dir)
    if main is None:
        raise LatexParseError(f"No .tex file with \\documentclass in {source_dir}")
    flat = flatten(main, source_dir)
    return parse_flat(flat, arxiv_id, find_heading_aliases(source_dir))


def parse_flat(
    flat: FlatSource, arxiv_id: str, heading_aliases: dict[str, str] | None = None
) -> LatexParse:
    raw = flat.text
    masked = mask_comments(raw)
    macros = collect_simple_macros(masked)

    begin = _BEGIN_DOC_RE.search(masked)
    body_start = begin.end() if begin else 0
    end_m = _END_DOC_RE.search(masked, body_start)
    body_end = end_m.start() if end_m else len(masked)

    sections: list[ParsedSection] = []
    spans: list[SectionSpan] = []

    def add(title: str, kind: SectionKind, start: int, end: int, text: str) -> None:
        if not text.strip():
            return
        idx = len(sections)
        sections.append(ParsedSection(title=title, kind=kind, text=text))
        spans.append(SectionSpan(idx=idx, start=start, end=end))

    abstract = _find_abstract(masked)
    if abstract:
        add(
            "Abstract",
            "abstract",
            abstract[0],
            abstract[1],
            latex_to_text(raw[abstract[2] : abstract[3]], macros),
        )

    headings = _find_headings(masked, body_start, body_end, macros, heading_aliases)
    first_heading = headings[0].start if headings else body_end

    # Front matter: between \begin{document} and the first heading, minus the abstract.
    front = raw[body_start:first_heading]
    if abstract and body_start <= abstract[0] < first_heading:
        front = raw[body_start : abstract[0]] + raw[abstract[1] : first_heading]
    front = _remove_envs(_remove_macro_with_args(front, _FRONT_MATTER_MACROS), _FRONT_MATTER_ENVS)
    front_text = latex_to_text(front, macros)
    if len(front_text) >= _MIN_FRONT_MATTER_CHARS:
        add("Front matter", "front_matter", body_start, first_heading, front_text)

    parent = ""
    for n, h in enumerate(headings):
        end = headings[n + 1].start if n + 1 < len(headings) else body_end
        if h.kind == "section":
            parent = h.title
            title = h.title
        else:
            title = f"{parent} > {h.title}" if parent else h.title
        if _REFERENCES_TITLE_RE.match(h.title):
            continue
        kind: SectionKind = "appendix" if h.appendix else h.kind  # type: ignore[assignment]
        if h.appendix:
            title = f"Appendix: {title}"
        content = raw[h.body_start : end]
        if abstract and h.body_start <= abstract[0] < end:
            content = raw[h.body_start : abstract[0]] + raw[abstract[1] : end]
        add(title, kind, h.start, end, latex_to_text(content, macros))

    if not sections:
        raise LatexParseError("LaTeX source produced no text")
    doc = ParsedDocument(
        arxiv_id=arxiv_id, source="latex", sections=sections, main_file=flat.main_file
    )
    return LatexParse(document=doc, flat=flat, spans=spans)
