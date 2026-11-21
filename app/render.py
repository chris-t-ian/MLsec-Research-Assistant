"""Chainlit rendering of core events and results. Presentation only: no retrieval, agent or
security decisions happen here.

Core calls the sinks from a worker thread (``cl.make_async``); every UI update is scheduled on
the Chainlit event loop with ``cl.run_sync``.

All text that comes from papers, arXiv or Semantic Scholar is untrusted: it is escaped before
it is put into markdown, and links are only built from validated arXiv IDs / DOIs.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import quote

import chainlit as cl
from chainlit.utils import utc_now

from research_assistant.agent import TurnResult
from research_assistant.generation.answer import citation_ref_name, render_with_references
from research_assistant.models import Citation, SecurityEvent, ToolResult

logger = logging.getLogger(__name__)

IN_KB_BADGE = "`in knowledge base`"
MAX_TABLE_ROWS = 20
MAX_CELL_CHARS = 160

# --------------------------------------------------------------------------- markdown helpers

# Brackets are escaped, so parentheses cannot form links and need no escaping.
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]#|<>~])")
_WS = re.compile(r"\s+")
_ARXIV_ID = re.compile(r"^(?:\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$")
_DOI = re.compile(r"^10\.\d{4,9}/\S+$")


def md(text: Any, max_chars: int | None = None) -> str:
    """Untrusted text as inline markdown: one line, specials escaped, optionally truncated."""
    flat = _WS.sub(" ", str(text if text is not None else "")).strip()
    if max_chars is not None and len(flat) > max_chars:
        flat = flat[: max_chars - 1].rstrip() + "…"
    return _MD_SPECIAL.sub(r"\\\1", flat)


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Markdown table; cells are escaped and truncated."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append("| " + " | ".join(md(c, MAX_CELL_CHARS) or " " for c in row) + " |")
    return "\n".join(lines)


def arxiv_url(arxiv_id: Any) -> str | None:
    if isinstance(arxiv_id, str) and _ARXIV_ID.match(arxiv_id):
        return f"https://arxiv.org/abs/{arxiv_id}"
    return None


def paper_link(title: Any, arxiv_id: Any = None, doi: Any = None) -> str:
    text = md(title, 200) or "(untitled)"
    url = arxiv_url(arxiv_id)
    if url is None and isinstance(doi, str) and _DOI.match(doi):
        url = "https://doi.org/" + quote(doi, safe="/")
    return f"[{text}]({url})" if url else text


def fmt_number(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return "-" if value is None else str(value)
    if isinstance(value, float):
        if value == float("inf"):
            return "∞"
        return f"{value:.4g}"
    return str(value)


def fmt_latency(metadata: Mapping[str, Any] | None) -> str:
    ms = (metadata or {}).get("latency_ms")
    if not isinstance(ms, int | float):
        return ""
    return f"_{ms / 1000:.1f} s_" if ms >= 1000 else f"_{ms:.0f} ms_"


# --------------------------------------------------------------------------- step names

STEP_LABELS = {
    # agent nodes
    "validate_input": "Validate input",
    "detect_injection": "Prompt-injection check",
    "agent": "Plan (agent model)",
    "tools": "Run tools",
    "finalize": "Write answer",
    # retrieval stages
    "self_query": "Self-query filters",
    "query_translation": "Query rewrites",
    "retrieve": "Retrieve candidates",
    "fuse": "Fuse rankings (RRF)",
    "rerank": "Rerank (cross-encoder)",
    "select": "Select top chunks",
    # generation
    "generate": "Generate answer",
    "cite": "Check citations",
    # ingest
    "fetch": "Download from arXiv",
    "parse_scan": "Parse and scan for hidden injections",
    "chunk": "Chunk",
    "index": "Index",
}

TOOL_LABELS = {
    "search_knowledge_base": "Search knowledge base",
    "dp_accountant": "DP accountant",
    "citation_graph": "Citation graph",
    "arxiv_search": "arXiv search",
    "arxiv_ingest_request": "Ingest request",
    "bibtex_export": "BibTeX export",
    "confirm_ingest": "Ingest paper",
}

_LLM_STAGES = {"agent", "self_query", "query_translation", "generate"}


def step_type(step_id: str, name: str) -> str:
    """Chainlit step type for a core step id (``agent:``, ``retrieval:``, ``generation:``...)."""
    prefix = step_id.split(":", 1)[0]
    if name in _LLM_STAGES:
        return "llm"
    if prefix == "retrieval":
        return "retrieval"
    if name == "tools" or prefix == "ingest":
        return "tool"
    return "run"


def step_label(name: str) -> str:
    return STEP_LABELS.get(name, name.replace("_", " ").capitalize())


# --------------------------------------------------------------------------- step outputs


def _top_table(hits: Any) -> str:
    if not isinstance(hits, list) or not hits:
        return "No chunks."
    rows = [
        (i, h.get("chunk_id", ""), fmt_number(h.get("score")))
        for i, h in enumerate(hits[:MAX_TABLE_ROWS], 1)
        if isinstance(h, dict)
    ]
    return table(["#", "Chunk", "Score"], rows)


def _filters_text(filters: Any) -> str:
    if not isinstance(filters, dict):
        return "none"
    parts = []
    if filters.get("year_min") is not None:
        parts.append(f"year ≥ {filters['year_min']}")
    if filters.get("year_max") is not None:
        parts.append(f"year ≤ {filters['year_max']}")
    if filters.get("topic"):
        parts.append(f"topic = {md(filters['topic'])}")
    if filters.get("authors"):
        parts.append("authors: " + ", ".join(md(a) for a in filters["authors"]))
    if filters.get("arxiv_ids"):
        parts.append("papers: " + ", ".join(md(a) for a in filters["arxiv_ids"]))
    if filters.get("section_type") not in (None, "any"):
        parts.append(f"sections: {md(filters['section_type'])}")
    return "; ".join(parts) or "none"


def format_step_output(name: str, output: Any, metadata: Mapping[str, Any] | None = None) -> str:
    """Readable markdown for one finished core step."""
    meta = metadata or {}
    out = output if isinstance(output, dict) else {}
    lines: list[str] = []

    if isinstance(output, str):  # ingest stages report a short status string
        lines.append(md(output))
        if meta.get("error"):
            lines.append(f"⚠️ {md(meta['error'])}")
    elif name == "validate_input":
        lines.append("Input accepted." if out.get("ok") else "Input rejected.")
        if out.get("removed_chars"):
            lines.append(f"Removed {out['removed_chars']} invisible or control characters.")
    elif name == "detect_injection":
        # Verdict only: scores and matched patterns would help tune an attack.
        verdict = {"allow": "no injection detected", "warn": "possible injection (warned)"}
        lines.append(f"Verdict: {verdict.get(str(out.get('verdict')), 'blocked')}")
    elif name == "agent":
        if out.get("canary_leak"):
            lines.append("Output withheld by the security layer.")
        elif out.get("error"):
            lines.append(f"⚠️ {md(out['error'])}")
        elif out.get("tool_calls"):
            calls = ", ".join(TOOL_LABELS.get(t, md(t)) for t in out["tool_calls"])
            lines.append(f"Requested tools: {calls}")
        else:
            lines.append("Enough evidence collected.")
    elif name == "tools":
        rows = [
            (
                TOOL_LABELS.get(c.get("tool", ""), c.get("tool", "")),
                ("ok" if c.get("ok") else "failed") if c.get("allowed") else "denied",
                fmt_latency({"latency_ms": c["ms"]}).strip("_") if "ms" in c else "",
            )
            for c in out.get("calls", [])
            if isinstance(c, dict)
        ]
        lines.append(table(["Tool", "Status", "Time"], rows) if rows else "No tool calls.")
    elif name == "finalize":
        if out.get("blocked") or out.get("canary_leak"):
            lines.append("The answer was withheld.")
        else:
            lines.append(
                f"{out.get('n_sources', 0)} sources, {out.get('n_tool_results', 0)} tool results; "
                f"cited {len(out.get('citations', []))} sources"
            )
    elif name == "self_query":
        lines.append(f"**Search query:** {md(out.get('semantic_query', ''))}")
        lines.append(f"**Filters:** {_filters_text(out.get('filters'))}")
    elif name == "query_translation":
        rewrites = out.get("rewrites") or []
        lines.append("**Rewrites:**" if rewrites else "No rewrites (original query used).")
        lines += [f"{i}. {md(r, 400)}" for i, r in enumerate(rewrites, 1)]
    elif name == "retrieve":
        lists = out.get("lists") or {}
        lines.append(", ".join(f"{md(k)}: {v} candidates" for k, v in lists.items()) or "none")
    elif name in ("fuse", "rerank"):
        lines.append(_top_table(out.get("top")))
    elif name == "select":
        lines.append(_top_table(out.get("chunks")))
    elif name == "generate":
        if out.get("error"):
            lines.append(f"⚠️ {md(out['error'])}")
        else:
            lines.append(f"{out.get('chars', 0)} characters from {md(meta.get('model') or '')}")
    elif name == "cite":
        cited = ", ".join(f"[{n}]" for n in out.get("citations", [])) or "none"
        lines.append(f"Cited sources: {cited}")
        if out.get("tool_citations"):
            lines.append("Cited tool results: " + ", ".join(md(t) for t in out["tool_citations"]))
        if out.get("errors"):
            lines.append(f"Removed {len(out['errors'])} invalid citation marker(s).")
    elif out:
        lines.append(f"```json\n{json.dumps(out, ensure_ascii=False, default=str)[:2000]}\n```")

    usage = meta.get("token_usage")
    if isinstance(usage, dict) and usage.get("total_tokens"):
        lines.append(f"_{usage['total_tokens']} tokens_")
    for note in meta.get("notes") or []:
        lines.append(f"_Note: {md(note, 300)}_")
    latency = fmt_latency(meta)
    if latency:
        lines.append(latency)
    return "\n\n".join(lines)


# --------------------------------------------------------------------------- tool outputs


def format_search(data: dict[str, Any]) -> str:
    sources = data.get("sources") or []
    head = f"**Query:** {md(data.get('query', ''))}\n\n"
    if not sources:
        return head + "No matching passages."
    rows = [
        (s.get("number"), s.get("ref"), s.get("title"), s.get("section") or "", s.get("arxiv_id"))
        for s in sources[:MAX_TABLE_ROWS]
    ]
    return head + table(["#", "Reference", "Title", "Section", "arXiv"], rows)


def format_dp(data: dict[str, Any]) -> str:
    rows: list[tuple[str, str]] = [
        ("Mode", "epsilon for given noise" if data.get("mode") == "epsilon" else "noise search"),
        ("Accountant", str(data.get("accountant", "")).upper()),
        ("ε (epsilon)", fmt_number(data.get("epsilon"))),
        ("δ (delta)", fmt_number(data.get("delta"))),
        ("σ (noise multiplier)", fmt_number(data.get("noise_multiplier"))),  # noqa: RUF001
        ("q (sample rate)", fmt_number(data.get("sample_rate"))),
        ("T (steps)", fmt_number(data.get("steps"))),
    ]
    if data.get("target_epsilon") is not None:
        rows.append(("Target ε", fmt_number(data.get("target_epsilon"))))
    rows += [
        ("Sampling", str(data.get("sampling", ""))),
        ("Neighboring relation", str(data.get("neighboring_relation", "")).replace("_", " ")),
    ]
    text = table(["Parameter", "Value"], rows)
    warnings = data.get("warnings") or []
    if warnings:
        text += "\n\n" + "\n".join(f"- ⚠️ {md(w, 400)}" for w in warnings)
    return text


def _paper_line(p: dict[str, Any], abstract_key: str | None = None) -> str:
    authors = p.get("authors") or []
    bits = [paper_link(p.get("title"), p.get("arxiv_id"), p.get("doi"))]
    if p.get("year"):
        bits.append(f"({p['year']})")
    line = " ".join(bits)
    if authors:
        line += " · " + md(", ".join(str(a) for a in authors[:5]), 200)
    if p.get("citation_count") is not None:
        line += f" · {p['citation_count']} citations"
    if p.get("arxiv_id"):
        line += f" · arXiv:{md(p['arxiv_id'])}"
    if p.get("in_kb") or p.get("already_in_kb"):
        line += f" {IN_KB_BADGE}"
    line = f"- {line}"
    if abstract_key and p.get(abstract_key):
        line += f"\n  _{md(p[abstract_key], 300)}_"
    return line


def format_citation_graph(data: dict[str, Any]) -> str:
    target = md(data.get("arxiv_id", ""))
    head = (
        f"**Papers citing arXiv:{target}**"
        if data.get("direction") == "citations"
        else f"**References of arXiv:{target}**"
    )
    papers = data.get("papers") or []
    if not papers:
        return head + "\n\nNo papers found."
    return head + "\n\n" + "\n".join(_paper_line(p) for p in papers if isinstance(p, dict))


def format_arxiv_search(data: dict[str, Any]) -> str:
    results = data.get("results") or []
    if not results:
        return "No papers found on arXiv."
    return "\n".join(_paper_line(p, "abstract_snippet") for p in results if isinstance(p, dict))


def format_bibtex(data: dict[str, Any]) -> str:
    bib = str(data.get("bibtex", "")).replace("```", "'''")
    text = f"```bibtex\n{bib}\n```"
    if data.get("not_found"):
        text += "\n\nNot found on arXiv: " + ", ".join(md(i) for i in data["not_found"])
    return text


def format_findings(findings: Sequence[dict[str, Any]]) -> str:
    if not findings:
        return "Injection scan: no hidden or AI-directed content found."
    rows = [
        (
            f.get("severity", ""),
            str(f.get("type", "")).replace("_", " "),
            f.get("location", ""),
            f.get("snippet", ""),
        )
        for f in findings[:MAX_TABLE_ROWS]
    ]
    more = (
        f"\n\n…and {len(findings) - MAX_TABLE_ROWS} more." if len(findings) > MAX_TABLE_ROWS else ""
    )
    return (
        f"Injection scan: **{len(findings)} finding(s)**. Affected chunks are flagged and "
        "the text is always treated as untrusted data.\n\n"
        + table(["Severity", "Type", "Location", "Snippet"], rows)
        + more
    )


def format_ingest_result(data: dict[str, Any]) -> str:
    res = data.get("result") or {}
    lines = [
        f"Added {paper_link(res.get('title'), res.get('arxiv_id'))} "
        f"(arXiv:{md(res.get('arxiv_id', ''))}) to the knowledge base: "
        f"{res.get('n_chunks', 0)} chunks from {md(res.get('source', ''))}."
    ]
    if res.get("n_quarantined"):
        lines.append(f"{res['n_quarantined']} chunk(s) were quarantined and are never retrieved.")
    if res.get("n_suspicious"):
        lines.append(f"{res['n_suspicious']} chunk(s) are flagged as suspicious.")
    lines.append(format_findings(data.get("findings") or []))
    return "\n\n".join(lines)


def format_tool_result(name: str, result: ToolResult) -> str:
    if not result.ok:
        return f"⚠️ {md(result.error or 'The tool failed.', 500)}"
    data = result.data if isinstance(result.data, dict) else {}
    match name:
        case "search_knowledge_base":
            return format_search(data)
        case "dp_accountant":
            return format_dp(data)
        case "citation_graph":
            return format_citation_graph(data)
        case "arxiv_search":
            return format_arxiv_search(data)
        case "bibtex_export":
            return format_bibtex(data)
        case "arxiv_ingest_request":
            return md(data.get("message", "Ingest request created."), 500)
        case "confirm_ingest":
            return format_ingest_result(data)
    return f"```json\n{json.dumps(data, ensure_ascii=False, default=str)[:3000]}\n```"


def bibtex_file(result: ToolResult) -> cl.File | None:
    if not (result.ok and isinstance(result.data, dict) and result.data.get("bibtex")):
        return None
    return cl.File(
        name="references.bib",
        content=str(result.data["bibtex"]).encode("utf-8"),
        mime="text/x-bibtex",
        display="inline",
    )


# --------------------------------------------------------------------------- sinks


class ChainlitSink:
    """EventSink rendering core steps as nested ``cl.Step`` and streaming tokens to ``message``.

    Steps nest by start/finish order: a step started while another is open becomes its child
    (retrieval stages inside the ``tools`` node, generation inside ``finalize``).
    """

    def __init__(self, root: cl.Step, message: cl.Message) -> None:
        self.root = root
        self.message = message
        self.files: list[cl.File] = []  # downloads to attach to the final message
        self.streamed = False  # True once a token reached the message (it exists in the UI)
        self._open: list[tuple[str, str, cl.Step]] = []  # (step id, core name, step)

    # -- EventSink (called from the worker thread)

    def step_started(self, step_id: str, name: str, input: Any) -> None:
        self._run(self._start(step_id, name))

    def step_finished(
        self, step_id: str, output: Any, metadata: dict[str, Any] | None = None
    ) -> None:
        self._run(self._finish(step_id, output, metadata))

    def tool_result(self, name: str, args: dict[str, Any], result: Any) -> None:
        self._run(self._tool(name, args, result))

    def token(self, text: str) -> None:
        if text:
            self.streamed = True
            self._run(self.message.stream_token(text))

    # -- async implementations

    @staticmethod
    def _run(coro: Any) -> None:
        try:
            cl.run_sync(coro)
        except Exception:  # rendering must never break the agent turn
            logger.exception("Rendering an event failed")

    def _parent_id(self) -> str:
        return self._open[-1][2].id if self._open else self.root.id

    async def _start(self, step_id: str, name: str) -> None:
        step = cl.Step(
            name=step_label(name),
            type=step_type(step_id, name),  # type: ignore[arg-type]
            parent_id=self._parent_id(),
            show_input=False,
        )
        step.start = utc_now()
        self._open.append((step_id, name, step))
        await step.send()

    async def _finish(self, step_id: str, output: Any, metadata: dict[str, Any] | None) -> None:
        idx = next(
            (i for i in range(len(self._open) - 1, -1, -1) if self._open[i][0] == step_id), None
        )
        if idx is None:
            return
        # Close the step and anything left open inside it.
        closing = self._open[idx:]
        del self._open[idx:]
        for sid, name, step in reversed(closing):
            if sid == step_id:
                step.output = format_step_output(name, output, metadata)
                if output == "failed" or (isinstance(output, dict) and output.get("error")):
                    step.is_error = True
            step.end = utc_now()
            await step.update()

    async def _tool(self, name: str, args: dict[str, Any], result: ToolResult) -> None:
        step = cl.Step(
            name=TOOL_LABELS.get(name, name),
            type="tool",
            parent_id=self._parent_id(),
            show_input="json",
        )
        step.start = utc_now()
        step.input = args
        step.output = format_tool_result(name, result)
        step.is_error = not result.ok
        if name == "bibtex_export" and (file := bibtex_file(result)) is not None:
            step.elements = [file]
            self.files.append(
                cl.File(name=file.name, content=file.content, mime=file.mime, display="inline")
            )
        step.end = utc_now()
        await step.send()


INGEST_STAGES = ("fetch", "parse_scan", "chunk", "index")


class IngestTaskSink:
    """EventSink showing the ingest stages (``ingest:<id>:<stage>``) in a ``cl.TaskList``."""

    def __init__(self, task_list: cl.TaskList, tasks: Mapping[str, cl.Task]) -> None:
        self.task_list = task_list
        self.tasks = dict(tasks)

    @classmethod
    async def create(cls, arxiv_id: str) -> IngestTaskSink:
        task_list = cl.TaskList(status=f"Adding arXiv:{arxiv_id}…")
        tasks = {}
        for stage in INGEST_STAGES:
            tasks[stage] = cl.Task(title=STEP_LABELS[stage], status=cl.TaskStatus.READY)
            await task_list.add_task(tasks[stage])
        await task_list.send()
        return cls(task_list, tasks)

    def _set(self, step_id: str, status: cl.TaskStatus) -> None:
        task = self.tasks.get(step_id.rsplit(":", 1)[-1])
        if task is None:
            return
        task.status = status
        try:
            cl.run_sync(self.task_list.update())
        except Exception:
            logger.exception("Updating the ingest task list failed")

    def step_started(self, step_id: str, name: str, input: Any) -> None:
        self._set(step_id, cl.TaskStatus.RUNNING)

    def step_finished(
        self, step_id: str, output: Any, metadata: dict[str, Any] | None = None
    ) -> None:
        self._set(step_id, cl.TaskStatus.FAILED if output == "failed" else cl.TaskStatus.DONE)

    def tool_result(self, name: str, args: dict[str, Any], result: Any) -> None:
        pass

    def token(self, text: str) -> None:
        pass

    async def finish(self, ok: bool) -> None:
        if not ok:
            for task in self.tasks.values():
                if task.status in (cl.TaskStatus.RUNNING, cl.TaskStatus.READY):
                    task.status = cl.TaskStatus.FAILED
        self.task_list.status = "Done" if ok else "Failed"
        await self.task_list.update()


# --------------------------------------------------------------------------- answer


def source_element(c: Citation) -> cl.Text:
    """Side-panel element for one citation, named like the reference in the answer text."""
    parts = [f"**{md(c.title, 300)}**", f"**Reference:** {md(c.label)}"]
    if c.section:
        parts.append(f"**Section:** {md(c.section, 200)}")
    if c.excerpt:
        parts.append(f"> {md(c.excerpt, 1200)}")
    url = arxiv_url(c.arxiv_id)
    parts.append(f"[arXiv:{md(c.arxiv_id)}]({url})" if url else f"arXiv:{md(c.arxiv_id)}")
    parts.append(f"_Chunk {md(c.chunk_id)}_")
    return cl.Text(name=citation_ref_name(c), content="\n\n".join(parts), display="side")


INPUT_REJECTED_PREFIX = "I couldn't process that message."
INJECTION_BLOCKED = (
    "I can't help with that request. It reads like an attempt to change my instructions or "
    "make me reveal them, so I stopped before searching or calling any tools. Ask me about ML "
    "privacy and security research instead, for example membership inference, DP-SGD or "
    "prompt-injection defenses."
)
OUTPUT_WITHHELD = (
    "I withheld this answer because a safety check on the generated text failed. Please "
    "rephrase your question and try again."
)
INJECTION_WARNED = (
    "**Note:** part of your message looked like an instruction aimed at the assistant rather "
    "than a research question. I ignored that part and only answered the research question."
)


def security_notice(events: Sequence[SecurityEvent], blocked: bool) -> tuple[str | None, bool]:
    """User-facing text for blocked or warned turns: ``(text, replaces_answer)``.

    Explains why without internals (no scores, patterns or policy details).
    """
    kinds = {e.kind: e for e in events}
    if blocked:
        if "input_rejected" in kinds:
            # Input-validation reasons are written for users (empty, too long, ...).
            return f"{INPUT_REJECTED_PREFIX} {kinds['input_rejected'].detail}", True
        if "injection_block" in kinds:
            return INJECTION_BLOCKED, True
        return OUTPUT_WITHHELD, True
    if "injection_warn" in kinds:
        return INJECTION_WARNED, False
    return None, False


def answer_markdown(result: TurnResult) -> str:
    """Final message text: references named like the source elements, plus security notes."""
    notice, replaces = security_notice(result.security_events, result.blocked)
    if notice is not None and replaces:
        return notice
    raw = result.raw_answer or result.answer
    text = render_with_references(raw, result.citations, result.tool_citations)
    if notice is not None:
        text = f"> {notice}\n\n{text}"
    return text


def run_summary(result: TurnResult) -> str:
    tools = sum(1 for o in result.tool_results if o.allowed)
    parts = [f"{tools} tool call(s)"]
    if result.token_usage.total_tokens:
        parts.append(f"{result.token_usage.total_tokens} tokens")
    if result.model:
        parts.append(md(result.model))
    parts.append(f"{result.latency_ms / 1000:.1f} s")
    return " · ".join(parts)
