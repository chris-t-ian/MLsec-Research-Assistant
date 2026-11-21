"""Cited answer generation: retrieve, build a spotlighted prompt, stream, parse citations.

Pipeline of :func:`answer` (events on the sink, step ids ``generation:<strategy>:<stage>``):

    retrieval stages   see research_assistant.retrieval.strategies
    generate           streams raw model tokens (with ``[n]`` markers) to ``sink.token``
    cite               validates markers, builds Citation objects, renders author-year text

The model cites sources as ``[n]`` markers. After streaming, markers that point to no
source are removed and logged as citation errors; valid markers are rendered as
``(Shokri et al. 2017 [1])`` in :attr:`Answer.text`, keeping the number for the UI.
:attr:`Answer.raw_text` keeps the validated numeric markers only.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Sequence
from functools import lru_cache
from itertools import groupby
from typing import Any

from pydantic import BaseModel, ConfigDict

from research_assistant.config import GenerationConfig, Settings, get_settings
from research_assistant.events import EventSink, NullSink
from research_assistant.generation.prompts import build_messages, new_boundary
from research_assistant.llm import (
    LLMCallError,
    LLMConfigError,
    StreamingLLM,
    get_streaming_llm,
)
from research_assistant.models import (
    Answer,
    Chunk,
    Citation,
    CitationError,
    RetrievalTrace,
    ScoredChunk,
    Source,
    TokenUsage,
    ToolCitation,
    ToolEvidence,
)
from research_assistant.retrieval.strategies import (
    RetrievalDeps,
    StrategyConfig,
    default_deps,
    retrieve,
)

logger = logging.getLogger(__name__)

ARXIV_ABS_URL = "https://arxiv.org/abs/{arxiv_id}"

NO_SOURCES_MESSAGE = (
    "I could not find any passages in the indexed papers that match this question, so I "
    "cannot answer it with citations. You can use the arXiv search tool to find and add "
    "relevant papers."
)
LLM_FAILED_MESSAGE = (
    "The language model could not produce an answer right now. Please try again in a moment."
)


class GenerationError(RuntimeError):
    """Answer generation failed. The message is safe to show to users."""


# --------------------------------------------------------------------------- labels

_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def surname(name: str) -> str:
    """Family name of ``"First Last"`` or ``"Last, First"`` (suffixes like Jr. dropped)."""
    name = " ".join(name.split())
    if "," in name:
        head = name.split(",", 1)[0].strip()
        if head:
            return head
    tokens = [t for t in name.split() if t.lower().strip(".") not in _NAME_SUFFIXES]
    return tokens[-1] if tokens else name


def author_year_label(authors: Sequence[str], year: int | None, fallback: str) -> str:
    """``Shokri 2017``, ``Shokri and Stronati 2017`` or ``Shokri et al. 2017``."""
    names = [surname(a) for a in authors if a.strip()]
    if not names:
        who = fallback
    elif len(names) == 1:
        who = names[0]
    elif len(names) == 2:
        who = f"{names[0]} and {names[1]}"
    else:
        who = f"{names[0]} et al."
    return f"{who} {year if year is not None else 'n.d.'}"


def make_sources(chunks: Sequence[ScoredChunk | Chunk]) -> list[Source]:
    """Number chunks 1..n and give them author-year labels.

    Different papers with the same label get letter suffixes (``Carlini et al. 2021a``/``b``)
    in order of first appearance; chunks of the same paper share one label.
    """
    plain = [c.chunk if isinstance(c, ScoredChunk) else c for c in chunks]
    base: dict[str, str] = {}
    for c in plain:
        base.setdefault(c.arxiv_id, author_year_label(c.authors, c.year, f"arXiv:{c.arxiv_id}"))
    papers_per_label: dict[str, list[str]] = {}
    for arxiv_id, label in base.items():
        papers_per_label.setdefault(label, []).append(arxiv_id)
    final: dict[str, str] = {}
    for label, ids in papers_per_label.items():
        if len(ids) == 1:
            final[ids[0]] = label
            continue
        sep = "-" if label.endswith("n.d.") else ""
        for i, arxiv_id in enumerate(ids):
            final[arxiv_id] = f"{label}{sep}{chr(ord('a') + i % 26)}"
    return [Source(number=i, chunk=c, label=final[c.arxiv_id]) for i, c in enumerate(plain, 1)]


# --------------------------------------------------------------------------- citations

# One bracket group: "[3]" or "[2, 4]" (positive integers without leading zeros, so math
# like "[0, 1]" is not taken for a citation). A run is several adjacent groups: "[2][4]".
_GROUP = r"\[\s*[1-9]\d*(?:\s*,\s*[1-9]\d*)*\s*\]"
_RUN = re.compile(rf"(?P<pre>[ \t]*)(?P<run>{_GROUP}(?:[ \t]*{_GROUP})*)")
_NUM = re.compile(r"\d+")
_WS = re.compile(r"\s+")


_TOOL_MARKER = re.compile(r"(?P<pre>[ \t]*)\[T(?P<n>[1-9]\d*)\]")
# A run of adjacent tool markers, e.g. "[T1][T2]" or "[T1] [T2]" (rendered as one group).
_TOOL_RUN = re.compile(r"(?P<pre>[ \t]*)(?P<run>\[T[1-9]\d*\](?:[ \t]*\[T[1-9]\d*\])*)")
_TOOL_NUM = re.compile(r"\[T(\d+)\]")


class ParsedCitations(BaseModel):
    model_config = ConfigDict(frozen=True)

    raw_text: str
    text: str
    citations: list[Citation]
    errors: list[CitationError]
    tool_citations: list[ToolCitation] = []


def make_excerpt(text: str, max_chars: int) -> str:
    """Whitespace-normalized start of ``text``, cut at a word boundary."""
    flat = _WS.sub(" ", text).strip()
    if len(flat) <= max_chars:
        return flat
    cut = flat[:max_chars].rsplit(" ", 1)[0] or flat[:max_chars]
    return cut.rstrip(" ,;:") + "…"


def _render_run(numbers: Sequence[int], by_number: dict[int, Source]) -> str:
    parts = []
    for label, group in groupby(numbers, key=lambda n: by_number[n].label):
        parts.append(f"{label} " + "".join(f"[{n}]" for n in group))
    return "(" + "; ".join(parts) + ")"


def _parse_tool_markers(
    raw: str, text: str, tools: Sequence[ToolEvidence]
) -> tuple[str, str, list[ToolCitation], list[CitationError]]:
    """Validate ``[Tn]`` markers: keep valid ones in ``raw``, render ``(name [Tn])`` in ``text``.

    Invalid markers are removed from both and reported once.
    """
    by_number = {t.number: t for t in tools}
    errors: list[CitationError] = []
    cited: list[int] = []

    def _raw(m: re.Match[str]) -> str:
        n = int(m["n"])
        if n not in by_number:
            errors.append(
                CitationError(
                    marker=f"[T{n}]",
                    number=n,
                    reason=f"no tool result with this number ({len(tools)} tool results)",
                )
            )
            logger.warning("Citation error: marker [T%d] refers to no tool result", n)
            return ""
        if n not in cited:
            cited.append(n)
        return m.group(0)

    def _text(m: re.Match[str]) -> str:
        numbers: list[int] = []
        for n in (int(x) for x in _TOOL_NUM.findall(m["run"])):
            if n in by_number and n not in numbers:
                numbers.append(n)
        if not numbers:
            return ""
        pre = m["pre"]
        if not pre and m.start() > 0 and not m.string[m.start() - 1].isspace():
            pre = " "
        parts = [
            f"{name} " + "".join(f"[T{n}]" for n in group)
            for name, group in groupby(numbers, key=lambda n: by_number[n].name)
        ]
        return pre + "(" + "; ".join(parts) + ")"

    new_raw = _TOOL_MARKER.sub(_raw, raw)
    new_text = _TOOL_RUN.sub(_text, text)
    citations = [
        ToolCitation(number=n, name=by_number[n].name, ok=by_number[n].result.ok) for n in cited
    ]
    return new_raw, new_text, citations, errors


def parse_citations(
    text: str,
    sources: Sequence[Source],
    excerpt_chars: int = 300,
    tools: Sequence[ToolEvidence] = (),
) -> ParsedCitations:
    """Validate ``[n]`` and ``[Tn]`` markers in model output and turn them into citations.

    Markers whose number matches no source (or tool result) are removed (with the whitespace
    before them) and reported as :class:`CitationError`. Citations are ordered by first use.
    """
    by_number = {s.number: s for s in sources}
    errors: list[CitationError] = []
    cited: list[int] = []
    raw_parts: list[str] = []
    out_parts: list[str] = []
    pos = 0
    for m in _RUN.finditer(text):
        raw_parts.append(text[pos : m.start()])
        out_parts.append(text[pos : m.start()])
        pos = m.end()
        numbers: list[int] = []
        for n in (int(x) for x in _NUM.findall(m.group("run"))):
            if n not in by_number:
                errors.append(
                    CitationError(
                        marker=f"[{n}]",
                        number=n,
                        reason=f"no source with this number ({len(sources)} sources)",
                    )
                )
                logger.warning(
                    "Citation error: marker [%d] refers to no source (%d sources)",
                    n,
                    len(sources),
                )
            elif n not in numbers:
                numbers.append(n)
        if not numbers:
            continue
        for n in numbers:
            if n not in cited:
                cited.append(n)
        pre = m.group("pre")
        raw_parts.append(pre + "".join(f"[{n}]" for n in numbers))
        if not pre and m.start() > 0 and not text[m.start() - 1].isspace():
            pre = " "
        out_parts.append(pre + _render_run(numbers, by_number))
    raw_parts.append(text[pos:])
    out_parts.append(text[pos:])

    citations = []
    for n in cited:
        s = by_number[n]
        c = s.chunk
        citations.append(
            Citation(
                number=n,
                chunk_id=c.id,
                arxiv_id=c.arxiv_id,
                label=s.label,
                title=c.title,
                section=c.section,
                excerpt=make_excerpt(c.text, excerpt_chars),
                url=ARXIV_ABS_URL.format(arxiv_id=c.arxiv_id),
            )
        )
    raw_text, out_text, tool_citations, tool_errors = _parse_tool_markers(
        "".join(raw_parts), "".join(out_parts), tools
    )
    return ParsedCitations(
        raw_text=raw_text.strip(),
        text=out_text.strip(),
        citations=citations,
        errors=errors + tool_errors,
        tool_citations=tool_citations,
    )


# --------------------------------------------------------------------------- dependencies


class AnswerDeps:
    """Retrieval dependencies, the streaming LLM and generation settings."""

    def __init__(
        self,
        retrieval: RetrievalDeps,
        llm: StreamingLLM,
        config: GenerationConfig | None = None,
        model_name: str | None = None,
    ) -> None:
        self.retrieval = retrieval
        self.llm = llm
        self.config = config or GenerationConfig()
        self.model_name = model_name

    @classmethod
    def from_settings(
        cls, settings: Settings | None = None, retrieval: RetrievalDeps | None = None
    ) -> AnswerDeps:
        settings = settings or get_settings()
        try:
            llm = get_streaming_llm("main", settings)
        except LLMConfigError as exc:
            raise GenerationError(
                "No language model is configured. Set OPENROUTER_API_KEY in your .env file."
            ) from exc
        return cls(
            retrieval=retrieval or default_deps(),
            llm=llm,
            config=settings.generation,
            model_name=settings.models.main,
        )


@lru_cache(maxsize=1)
def default_answer_deps() -> AnswerDeps:
    return AnswerDeps.from_settings()


# --------------------------------------------------------------------------- pipeline


def generate_answer(
    question: str,
    chunks: Sequence[ScoredChunk | Chunk],
    sink: EventSink | None = None,
    deps: AnswerDeps | None = None,
    *,
    tool_results: Sequence[ToolEvidence] = (),
    history: Sequence[tuple[str, str]] = (),
    extra_system: str = "",
    retrieval: RetrievalTrace | None = None,
    strategy: str | None = None,
    max_tool_chars: int = 6000,
) -> Answer:
    """Write a cited answer from already collected ``chunks`` and ``tool_results``.

    Chunks become numbered sources ``[n]``, tool results spotlighted blocks ``[Tn]``.
    Without either, the no-sources message is returned and the LLM is not called. Raises
    :class:`GenerationError` (user-safe message) if the LLM fails.
    """
    t0 = time.perf_counter()
    sink = sink or NullSink()
    deps = deps or default_answer_deps()
    retrieval_usage = retrieval.token_usage if retrieval is not None else TokenUsage()
    name = strategy or "direct"

    if not chunks and not tool_results:
        sink.token(NO_SOURCES_MESSAGE)
        return Answer(
            question=question,
            text=NO_SOURCES_MESSAGE,
            raw_text=NO_SOURCES_MESSAGE,
            strategy=strategy,
            retrieval=retrieval,
            token_usage=retrieval_usage,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    cfg = deps.config
    sources = make_sources(chunks)
    boundary = new_boundary()
    system, user = build_messages(
        question,
        sources,
        boundary,
        cfg.datamarking,
        cfg.datamark_char,
        tool_results=tool_results,
        history=history,
        extra_system=extra_system,
        max_tool_chars=max_tool_chars,
    )

    step_id = f"generation:{name}:generate"
    sink.step_started(
        step_id,
        "generate",
        {
            "n_sources": len(sources),
            "n_tool_results": len(tool_results),
            "model": deps.model_name,
            "datamarking": cfg.datamarking,
        },
    )
    g0 = time.perf_counter()
    try:
        streamed = deps.llm.stream_text(system, user, sink.token)
    except LLMCallError as exc:
        gen_ms = (time.perf_counter() - g0) * 1000
        logger.warning("Answer generation failed: %s", exc)
        sink.step_finished(step_id, {"error": LLM_FAILED_MESSAGE}, {"latency_ms": round(gen_ms, 1)})
        raise GenerationError(LLM_FAILED_MESSAGE) from exc
    gen_ms = (time.perf_counter() - g0) * 1000
    sink.step_finished(
        step_id,
        {"chars": len(streamed.text)},
        {
            "latency_ms": round(gen_ms, 1),
            "token_usage": streamed.usage.model_dump(),
            "model": streamed.model,
        },
    )

    cite_id = f"generation:{name}:cite"
    sink.step_started(cite_id, "cite", {"n_sources": len(sources)})
    parsed = parse_citations(streamed.text, sources, cfg.excerpt_chars, tool_results)
    cite_out: dict[str, Any] = {
        "citations": [c.number for c in parsed.citations],
        "errors": [e.marker for e in parsed.errors],
    }
    if tool_results:
        cite_out["tool_citations"] = [f"T{c.number}" for c in parsed.tool_citations]
    sink.step_finished(
        cite_id,
        cite_out,
        {"n_citations": len(parsed.citations), "n_errors": len(parsed.errors)},
    )

    return Answer(
        question=question,
        text=parsed.text,
        raw_text=parsed.raw_text,
        citations=parsed.citations,
        citation_errors=parsed.errors,
        tool_citations=parsed.tool_citations,
        strategy=strategy,
        retrieval=retrieval,
        model=streamed.model or deps.model_name,
        token_usage=retrieval_usage + streamed.usage,
        generation_token_usage=streamed.usage,
        latency_ms=(time.perf_counter() - t0) * 1000,
        generation_latency_ms=gen_ms,
    )


def answer(
    question: str,
    strategy: StrategyConfig | str,
    sink: EventSink | None = None,
    deps: AnswerDeps | None = None,
) -> Answer:
    """Answer ``question`` from the indexed papers with citations.

    Raises :class:`~research_assistant.retrieval.strategies.RetrievalError` for invalid
    questions and :class:`GenerationError` if the LLM fails; both messages are user-safe.
    """
    t0 = time.perf_counter()
    sink = sink or NullSink()
    deps = deps or default_answer_deps()
    result = retrieve(question, strategy, sink, deps.retrieval)
    ans = generate_answer(
        result.question,
        result.chunks,
        sink,
        deps,
        retrieval=result.trace,
        strategy=result.strategy,
    )
    return ans.model_copy(update={"latency_ms": (time.perf_counter() - t0) * 1000})


__all__ = [
    "AnswerDeps",
    "GenerationError",
    "ParsedCitations",
    "answer",
    "author_year_label",
    "default_answer_deps",
    "generate_answer",
    "make_excerpt",
    "make_sources",
    "parse_citations",
    "surname",
]
