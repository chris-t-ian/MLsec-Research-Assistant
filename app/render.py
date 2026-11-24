"""Chainlit rendering of core events and results. Presentation only: no retrieval, agent or
security decisions happen here.

Core calls the sinks from a worker thread (``cl.make_async``); every UI update is scheduled on
the Chainlit event loop with ``cl.run_sync``.

All text that comes from papers, arXiv or Semantic Scholar is untrusted: it is escaped before
it is put into markdown, and links are only built from validated arXiv IDs / DOIs.
"""

from __future__ import annotations

import html
import json
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import quote

import chainlit as cl
import plotly.graph_objects as go
from chainlit.config import FILES_DIRECTORY
from chainlit.utils import utc_now
from plotly.subplots import make_subplots

from research_assistant.agent import TurnResult
from research_assistant.app_guide import GuideAnswer, HelpSection, help_ref_name
from research_assistant.generation.answer import citation_ref_name, render_with_references
from research_assistant.models import Answer, Citation, ClaimCheck, SecurityEvent, ToolResult
from research_assistant.rag_trace import CandidateRow, RagTrace, SearchView
from research_assistant.verification.claims import (
    badge_positions,
    borrows_citation,
    strip_markers,
    summary_line,
)

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
    "verify": "Verify answer",
    "verify_claims": "Verifying claims",
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
    # App Guide
    "guide_search": "Search help page",
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

# What the collapsed "Research" step shows while a stage runs. Chainlit's "Using"/"Used"
# prefix is blanked in .chainlit/translations/en-US.json so these read on their own.
ROOT_STEP_NAME = "Research"
PROGRESS_LABELS = {
    "validate_input": "Validating input…",
    "detect_injection": "Checking for prompt injection…",
    "agent": "Planning…",
    "tools": "Running tools…",
    "retrieve": "Searching papers…",
    "rerank": "Reranking results…",
    "finalize": "Writing answer…",
    "generate": "Writing answer…",
    "verify": "Verifying claims…",
    "verify_claims": "Verifying claims…",
}
REVISION_LABEL = "Revising answer…"
CHECKING_CLAIMS = "_Checking the claims against the cited sources…_"


def ensure_files_dir() -> None:
    """Recreate Chainlit's ``.files`` folder if it is gone: any Chainlit process that shuts
    down in this folder (e.g. a second ``chainlit run`` that failed to bind the port) deletes
    it, and then every element send fails."""
    FILES_DIRECTORY.mkdir(exist_ok=True)


def progress_label(step_id: str, name: str) -> str | None:
    """Label for the root step while ``name`` runs (None: keep the current one)."""
    if name == "generate" and ":revision" in step_id:
        return REVISION_LABEL
    return PROGRESS_LABELS.get(name)


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
    elif name == "verify":
        if out.get("canary_leak"):
            lines.append("The revised answer was withheld.")
        elif out.get("status") == "error":
            lines.append("⚠️ The claim check could not be completed.")
        else:
            lines.append(f"Policy: {md(out.get('policy', ''))}")
            if out.get("summary"):
                lines.append(md(out["summary"]))
            if out.get("regenerated"):
                lines.append("The answer was rewritten once without the unsupported claims.")
            if out.get("removed"):
                lines.append(f"Removed {out['removed']} unsupported sentence(s).")
    elif name == "verify_claims":
        if out.get("error"):
            lines.append(f"⚠️ {md(out['error'])}")
        else:
            counts = out.get("counts") or {}
            lines.append(
                ", ".join(f"{CLAIM_LABELS.get(k, md(k)).lower()}: {v}" for k, v in counts.items())
                or "No claims to check."
            )
            if out.get("fabricated"):
                lines.append(f"{out['fabricated']} verdict(s) quoted text not in the source.")
        if meta.get("judge_calls"):
            lines.append(f"_{meta['judge_calls']} judge call(s)_")
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
    elif name == "guide_search":
        sections = out.get("sections") or []
        lines.append("Help sections: " + (", ".join(md(t) for t in sections) or "none"))
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
        self.streamed = False  # True once the message exists in the UI (token or draft)
        # Source elements sent with the draft, reused for the final message (by name).
        self.source_elements: dict[str, cl.Text] = {}
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

    def answer_ready(self, answer: Answer, verifying: bool) -> None:
        self._run(self._draft(answer, verifying))

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
        label = progress_label(step_id, name)
        if label is not None and label != self.root.name:
            self.root.name = label
            await self.root.update()
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

    async def _draft(self, answer: Answer, verifying: bool) -> None:
        """Show the answer with linked sources while its claims are still being checked."""
        elements = [source_element(c) for c in answer.citations]
        self.source_elements = {e.name: e for e in elements}
        self.message.content = draft_markdown(answer, verifying)
        self.message.elements = list(elements)
        ensure_files_dir()
        if self.streamed:
            await self.message.update()
        else:
            await self.message.send()
            self.streamed = True

    def reuse_source(self, c: Citation) -> cl.Text:
        """The draft's element for ``c`` (same id, so no duplicate), else a new one."""
        return self.source_elements.get(citation_ref_name(c)) or source_element(c)

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

    def answer_ready(self, answer: Answer, verifying: bool) -> None:
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
OUTPUT_FILTERED = (
    "**Note:** links or images to sites outside arxiv.org, semanticscholar.org and doi.org "
    "were removed from this answer."
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
    notes = []
    if "injection_warn" in kinds:
        notes.append(INJECTION_WARNED)
    if "output_filtered" in kinds:
        notes.append(OUTPUT_FILTERED)
    return ("\n>\n> ".join(notes) if notes else None), False


# --------------------------------------------------------------------------- claim badges

# One glyph per claim status. The badge after each sentence is a side element named
# "<glyph> C<n>"; public/claims.js gives these element links the CSS classes
# claim-badge claim-<status> (public/custom.css), the glyph alone still reads without it.
CLAIM_GLYPHS = {
    "supported": "✓",
    "partially_supported": "◐",
    "unsupported": "✗",
    "uncited": "?",
}
CLAIM_LABELS = {
    "supported": "Supported",
    "partially_supported": "Partially supported",
    "unsupported": "Unsupported",
    "uncited": "No citation",
}


def claim_ref_name(check: ClaimCheck) -> str:
    """Badge / side-element name of a claim, e.g. ``✓ C3``."""
    return f"{CLAIM_GLYPHS[check.status]} C{check.index}"


def _cited_names(check: ClaimCheck, citations: Sequence[Citation]) -> list[str]:
    by_number = {c.number: c for c in citations}
    return [md(citation_ref_name(by_number[n])) for n in check.cited if n in by_number]


def claim_element(check: ClaimCheck, citations: Sequence[Citation] = ()) -> cl.Text:
    """Side panel for one claim: verdict, rationale and the cited chunk with the evidence span
    in bold. All text from the answer and the papers is escaped."""
    parts = [
        f"**C{check.index} · {CLAIM_LABELS[check.status]}**",
        f"> {md(strip_markers(check.sentence), 1500)}",
    ]
    if check.status == "uncited":
        parts.append(
            "This sentence states a fact, but neither it nor a later sentence of its paragraph "
            "cites a source, so it was not checked."
        )
        return cl.Text(name=claim_ref_name(check), content="\n\n".join(parts), display="side")
    cited = _cited_names(check, citations)
    if cited:
        via = " (via the next cited sentence of the paragraph)" if borrows_citation(check) else ""
        parts.append(f"**Cites{via}:** " + "; ".join(cited))
    if check.rationale:
        parts.append(f"**Judge:** {md(check.rationale, 600)}")
    if check.note:
        parts.append(
            "⚠️ **Fabricated evidence:** the judge quoted text that does not occur in the cited "
            "source, so the claim counts as unsupported."
        )
    if check.evidence_span and check.evidence_source is not None:
        src = next((c for c in citations if c.number == check.evidence_source), None)
        where = md(citation_ref_name(src)) if src else f"source {check.evidence_source}"
        if src and src.section:
            where += f", section {md(src.section, 200)}"
        before = md(check.evidence_before)
        after = md(check.evidence_after)
        lead = " " if check.evidence_before.endswith(" ") else ""
        trail = " " if check.evidence_after.startswith(" ") else ""
        parts.append(f"**Evidence** ({where}):")
        parts.append(f"> {before}{lead}**{md(check.evidence_span)}**{trail}{after}")
    return cl.Text(name=claim_ref_name(check), content="\n\n".join(parts), display="side")


def _answer_with_badges(result: TurnResult, raw: str) -> str:
    """The answer with claim badges: after a sentence's own citation, after the next cited
    sentence of its paragraph when it borrows that citation, or at the paragraph end when
    nothing cites it (see :func:`badge_positions`). Stale spans get no badge."""
    positions = badge_positions(raw, result.claim_checks)
    by_pos: dict[int, list[ClaimCheck]] = {}
    for c in sorted(result.claim_checks, key=lambda c: c.index):
        if c.index in positions:
            by_pos.setdefault(positions[c.index], []).append(c)
    out: list[str] = []
    pos = 0
    for at in sorted(by_pos):
        out.append(
            render_with_references(
                raw[pos:at], result.citations, result.tool_citations, strip=False
            )
        )
        out.append("".join(f" {claim_ref_name(c)}" for c in by_pos[at]))
        pos = at
    out.append(
        render_with_references(raw[pos:], result.citations, result.tool_citations, strip=False)
    )
    return "".join(out).strip()


def verification_line(result: TurnResult) -> str | None:
    """One-line summary of the claim check, e.g. "Claim check: 12 of 13 claims supported"."""
    info = result.verification
    if info is None or info.policy == "off":
        return None
    if info.status == "error":
        return "_Claim check unavailable for this answer._"
    if not result.claim_checks:
        return None
    line = f"Claim check: {summary_line(result.claim_checks)}"
    if info.regenerated:
        line += " · answer revised once"
    return f"_{md(line)}_"


def draft_markdown(answer: Answer, verifying: bool) -> str:
    """The answer before claim verification, references named like the source elements."""
    text = render_with_references(
        answer.raw_text or answer.text, answer.citations, answer.tool_citations
    )
    return f"{text}\n\n{CHECKING_CLAIMS}" if verifying else text


def answer_markdown(result: TurnResult) -> str:
    """Final message text: references named like the source elements, a badge per checked
    sentence, the claim-check summary and security notes."""
    notice, replaces = security_notice(result.security_events, result.blocked)
    if notice is not None and replaces:
        return notice
    raw = result.raw_answer or result.answer
    if result.claim_checks and result.raw_answer:
        text = _answer_with_badges(result, raw)
    else:
        text = render_with_references(raw, result.citations, result.tool_citations)
    line = verification_line(result)
    if line:
        text = f"{text}\n\n{line}"
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


# --------------------------------------------------------------------------- RAG view

RAG_STEP_NAME = "How this answer was built"

# Categorical slots in fixed order (validated palette, light steps; they also clear 3:1 on the
# dark Chainlit surface). Slot 8 is replaced by a neutral gray for "Other papers".
PAPER_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
OTHER_COLOR = "#8c8b86"
OTHER_PAPERS = "Other papers"
# Latency groups keep one color each (color follows the entity, not the rank).
GROUP_COLORS = {
    "security": "#1baf7a",
    "agent": "#6250d6",
    "retrieval": "#2a78d6",
    "tools": "#eda100",
    "generation": "#eb6834",
}
GROUP_LABELS = {
    "security": "Security checks",
    "agent": "Agent",
    "retrieval": "Retrieval",
    "tools": "Tools / overhead",
    "generation": "Generation",
}
# Light-mode ink; public/custom.css re-inks chart text for dark mode.
CHART_INK = "#52514e"
CHART_GRID = "rgba(128, 128, 128, 0.22)"
MAX_CHART_ROWS = 25

# (field, panel title, reversed axis). BM25 scores are not comparable across queries, so the
# rank is shown; its axis is reversed so that "better" is to the right in every panel.
STAGE_PANELS: tuple[tuple[str, str, bool], ...] = (
    ("dense_score", "Dense score (cosine)", False),
    ("bm25_rank", "BM25 rank", True),
    ("rrf_score", "RRF score", False),
    ("rerank_score", "Rerank score", False),
)
STATUS_STYLE = {
    # status: (symbol, size, opacity, legend text)
    "cited": ("circle", 12, 1.0, "cited in the answer"),
    "unused": ("circle-open", 12, 1.0, "given to the model, not cited"),
    "not_selected": ("circle", 7, 0.45, "candidate, not selected"),
}
TRANSLATION_LABELS = {
    "multi_query": "Paraphrases (multi-query)",
    "hyde": "Hypothetical passage (HyDE)",
    "decompose": "Sub-questions (decomposition)",
}


def plain(text: Any, max_chars: int = 120) -> str:
    """Untrusted text for Plotly labels: Plotly renders some HTML tags (incl. links), so
    entities are escaped (Plotly decodes them for display, as plain text)."""
    flat = _WS.sub(" ", str(text if text is not None else "")).strip()
    if len(flat) > max_chars:
        flat = flat[: max_chars - 1].rstrip() + "…"
    return html.escape(flat, quote=False)


def _short_chunk(chunk_id: str) -> str:
    """``1610.05820:3:12`` -> ``s3·c12`` (section and chunk index)."""
    parts = chunk_id.split(":")
    if len(parts) == 3:
        return f"s{parts[1]}·c{parts[2]}"
    return chunk_id[-12:]


def candidate_label(row: CandidateRow) -> str:
    if row.source_number is not None:
        return plain(f"[{row.source_number}] {row.paper_label}", 60)
    return plain(f"{row.paper_label} · {_short_chunk(row.chunk_id)}", 60)


def paper_colors(rows: Sequence[CandidateRow]) -> dict[str, tuple[str, str]]:
    """arXiv ID -> (legend name, color); papers beyond the palette share the gray 'Other'."""
    out: dict[str, tuple[str, str]] = {}
    for r in rows:
        if r.arxiv_id in out:
            continue
        i = len([v for v in out.values() if v[0] != OTHER_PAPERS])
        if i < len(PAPER_COLORS):
            out[r.arxiv_id] = (plain(r.paper_label, 40), PAPER_COLORS[i])
        else:
            out[r.arxiv_id] = (OTHER_PAPERS, OTHER_COLOR)
    return out


def _fmt_opt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return str(value) if isinstance(value, int) else f"{value:.{digits}g}"


def _hover(row: CandidateRow) -> str:
    status = STATUS_STYLE[row.status][3]
    return (
        f"<b>{candidate_label(row)}</b><br>"
        f"chunk {plain(row.chunk_id, 40)} · {status}<br>"
        f"dense {_fmt_opt(row.dense_score)} · BM25 rank {_fmt_opt(row.bm25_rank)}<br>"
        f"RRF {_fmt_opt(row.rrf_score)} · rerank {_fmt_opt(row.rerank_score)}<br>"
        f"final rank {_fmt_opt(row.final_rank)}"
    )


def _base_layout(fig: go.Figure, height: int) -> None:
    fig.update_layout(
        height=height,
        margin={"l": 8, "r": 16, "t": 40, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, system-ui, sans-serif", "size": 12, "color": CHART_INK},
        hoverlabel={"font": {"size": 12}, "align": "left"},
        legend={"orientation": "h", "yanchor": "top", "y": -0.04, "x": 0, "font": {"size": 11}},
    )
    fig.update_xaxes(gridcolor=CHART_GRID, zeroline=False, linecolor=CHART_GRID)
    fig.update_yaxes(gridcolor=CHART_GRID, zeroline=False, linecolor=CHART_GRID)


def candidates_figure(search: SearchView, max_rows: int = MAX_CHART_ROWS) -> go.Figure | None:
    """One panel per stage that ran, one row per candidate chunk, colored by paper."""
    rows = search.candidates[:max_rows]
    panels = [p for p in STAGE_PANELS if any(getattr(r, p[0]) is not None for r in rows)]
    if not rows or not panels:
        return None
    colors = paper_colors(rows)
    labels = [candidate_label(r) for r in rows]
    fig = make_subplots(
        rows=1,
        cols=len(panels),
        shared_yaxes=True,
        horizontal_spacing=0.03,
        subplot_titles=[p[1] for p in panels],
    )
    for col, (field, _title, reverse) in enumerate(panels, start=1):
        by_paper: dict[str, list[CandidateRow]] = {}
        for r in rows:
            if getattr(r, field) is not None:
                by_paper.setdefault(colors[r.arxiv_id][0], []).append(r)
        for name, group in by_paper.items():
            color = colors[group[0].arxiv_id][1]
            fig.add_trace(
                go.Scatter(
                    x=[getattr(r, field) for r in group],
                    y=[candidate_label(r) for r in group],
                    mode="markers",
                    name=name,
                    legendgroup=name,
                    showlegend=False,
                    marker={
                        "color": color,
                        "symbol": [STATUS_STYLE[r.status][0] for r in group],
                        "size": [STATUS_STYLE[r.status][1] for r in group],
                        "opacity": [STATUS_STYLE[r.status][2] for r in group],
                        "line": {"width": 2, "color": color},
                    },
                    hovertext=[_hover(r) for r in group],
                    hoverinfo="text",
                ),
                row=1,
                col=col,
            )
        if reverse:
            fig.update_xaxes(autorange="reversed", row=1, col=col)
    # Legend: one entry per paper, then the marker styles (identity is never color alone:
    # rows are labeled with the paper, and the status uses marker shape and size).
    seen: set[str] = set()
    for name, color in colors.values():
        if name in seen:
            continue
        seen.add(name)
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=name,
                legendgroup=name,
                marker={"color": color, "size": 10},
            )
        )
    for symbol, size, opacity, text in STATUS_STYLE.values():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=text,
                marker={
                    "color": OTHER_COLOR,
                    "symbol": symbol,
                    "size": size,
                    "opacity": opacity,
                    "line": {"width": 2, "color": OTHER_COLOR},
                },
            )
        )
    _base_layout(fig, height=140 + 24 * len(rows))
    fig.update_yaxes(
        categoryorder="array", categoryarray=labels, autorange="reversed", tickfont={"size": 11}
    )
    fig.update_annotations(font={"size": 12, "color": CHART_INK})
    return fig


def latency_figure(trace: RagTrace) -> go.Figure | None:
    """Horizontal bars: wall time per pipeline stage, in pipeline order (top to bottom)."""
    stages = trace.latencies
    if not stages:
        return None
    fig = go.Figure()
    for group, color in GROUP_COLORS.items():
        items = [s for s in stages if s.group == group]
        if not items:
            continue
        fig.add_trace(
            go.Bar(
                x=[s.latency_ms / 1000 for s in items],
                y=[s.label for s in items],
                orientation="h",
                name=GROUP_LABELS[group],
                marker={"color": color, "cornerradius": 4},
                text=[fmt_seconds(s.latency_ms) for s in items],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}: %{x:.2f} s<extra></extra>",
            )
        )
    _base_layout(fig, height=120 + 30 * len(stages))
    fig.update_layout(
        bargap=0.35,
        margin={"t": 64},
        title={"text": "Latency per stage", "font": {"size": 13}, "y": 0.98, "yanchor": "top"},
        legend={"y": 1.0, "yanchor": "bottom", "x": 0, "xanchor": "left"},
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=[s.label for s in stages],
        autorange="reversed",
        gridcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(ticksuffix=" s", rangemode="tozero")
    return fig


def fmt_seconds(ms: float) -> str:
    return f"{ms / 1000:.1f} s" if ms >= 100 else f"{ms:.0f} ms"


def _search_markdown(s: SearchView, n_searches: int) -> str:
    head = f"**Search {s.number}**" if n_searches > 1 else "**Knowledge-base search**"
    lines = [f"{head} · strategy `{md(s.strategy)}` · {fmt_seconds(s.latency_ms)}"]
    lines.append(f"- Query: {md(s.question, 400)}")
    if s.semantic_query and s.semantic_query != s.question:
        lines.append(f"- Search query after self-query: {md(s.semantic_query, 400)}")
    if s.query_translation and s.query_translation != "none":
        label = TRANSLATION_LABELS.get(s.query_translation, md(s.query_translation))
        if s.rewrites:
            lines.append(f"- {label}:")
            limit = 600 if s.query_translation == "hyde" else 300
            lines += [f"    {i}. {md(r, limit)}" for i, r in enumerate(s.rewrites, 1)]
        else:
            lines.append(f"- {label}: none (original query used)")
    else:
        lines.append("- Rewrites: none (strategy searches with the query as is)")
    if "self_query" in s.stages:
        filters = s.filters.model_dump() if s.filters is not None else None
        lines.append(f"- Extracted filters: {_filters_text(filters)}")
    else:
        lines.append("- Filters: not used by this strategy")
    shown = min(len(s.candidates), MAX_CHART_ROWS)
    lines.append(
        f"- Candidates: {s.n_candidates_total} unique chunks across stages "
        f"(chart shows the top {shown})"
    )
    lines += [f"- _Note: {md(n, 300)}_" for n in s.notes]
    return "\n".join(lines)


def _source_line(src: Any) -> str:
    section = f" · {md(src.section, 80)}" if src.section else ""
    return f"- **[{src.number}] {md(src.label)}**: {md(src.title, 140)}{section}"


def rag_markdown(trace: RagTrace) -> str:
    """Text part of the RAG view: rewrites and filters per search, cited vs. unused sources."""
    parts: list[str] = []
    if trace.searches:
        parts += [_search_markdown(s, len(trace.searches)) for s in trace.searches]
    else:
        parts.append(
            "_No knowledge-base search in this turn, so there is no candidates chart; the "
            "answer is based on tool results only._"
        )
    if trace.sources:
        cited, unused = trace.cited, trace.unused
        lines = [f"**Sources given to the answer model:** {len(cited)} cited, {len(unused)} unused"]
        if cited:
            lines.append("\n✅ **Cited**")
            lines += [_source_line(s) for s in cited]
        if unused:
            lines.append("\n○ **Retrieved but not cited**")
            lines += [_source_line(s) for s in unused]
        parts.append("\n".join(lines))
    if trace.searches:
        parts.append(
            "_Charts below: candidates per stage (filled = cited, open = unused, faint = not "
            "selected; further right is better) and latency per stage._"
        )
    else:
        parts.append("_Chart below: latency per stage._")
    return "\n\n".join(parts)


def rag_elements(trace: RagTrace) -> list[cl.Plotly]:
    elements: list[cl.Plotly] = []
    for s in trace.searches:
        fig = candidates_figure(s)
        if fig is not None:
            suffix = f" (search {s.number})" if len(trace.searches) > 1 else ""
            elements.append(
                cl.Plotly(
                    name=f"Candidates per stage{suffix}", figure=fig, display="inline", size="large"
                )
            )
    fig = latency_figure(trace)
    if fig is not None:
        elements.append(
            cl.Plotly(name="Latency per stage", figure=fig, display="inline", size="large")
        )
    return elements


def rag_summary(trace: RagTrace) -> str:
    n = len(trace.searches)
    bits = [f"{n} search{'es' if n != 1 else ''}"]
    if trace.sources:
        bits.append(f"{len(trace.cited)}/{len(trace.sources)} sources cited")
    bits.append(fmt_seconds(trace.total_latency_ms))
    return " · ".join(bits)


# --------------------------------------------------------------------------- App Guide


def help_element(section: HelpSection) -> cl.Text:
    """Side-panel element for a cited help section (help.md is our own trusted text)."""
    return cl.Text(
        name=help_ref_name(section),
        content=f"**{section.title}**\n\n{section.text}",
        display="side",
    )


def guide_markdown(ans: GuideAnswer) -> str:
    return ans.text
