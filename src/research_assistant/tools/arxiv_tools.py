"""arXiv tools: search, ingest request + confirmation, BibTeX export.

All arXiv traffic goes through the shared :class:`ArxivClient` in :class:`ToolDeps`
(at most one request every 3 s, timeout, retries, typed errors).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from pydantic import Field

from research_assistant.events import EventSink
from research_assistant.ingest.arxiv_fetch import ArxivMetadata
from research_assistant.ingest.pipeline import ingest_paper
from research_assistant.ingest.process import IngestError
from research_assistant.models import IngestRequest, ToolResult
from research_assistant.tools.arxiv_ids import ArxivIdStr, parse_arxiv_id
from research_assistant.tools.base import (
    ToolArgs,
    ToolDeps,
    clean_text,
    default_deps,
    fail,
    run_safely,
)

SEARCH_MAX_RESULTS = 10
ABSTRACT_SNIPPET_CHARS = 300
MAX_AUTHORS_SHOWN = 10
BIBTEX_MAX_IDS = 50
MAX_QUERY_CHARS = 300
MAX_QUERY_TERMS = 20


# --------------------------------------------------------------------------- search


class ArxivSearchArgs(ToolArgs):
    query: str = Field(
        min_length=1,
        max_length=MAX_QUERY_CHARS,
        description=(
            'Keywords to search for, e.g. "membership inference diffusion models". Use double '
            'quotes for exact phrases, e.g. "\\"label-only\\" attack". All terms must match.'
        ),
    )
    max_results: int = Field(
        default=5, ge=1, le=SEARCH_MAX_RESULTS, description="Number of results (1-10)."
    )


_TERM_RE = re.compile(r'"([^"]+)"|(\S+)')
_WORD_CLEAN_RE = re.compile(r"[^\w\-.]+")
_BOOLEAN_WORDS = {"AND", "OR", "ANDNOT", "NOT"}


def build_search_query(query: str) -> str:
    """Turn free text into an arXiv ``search_query``: every term ANDed over all fields.

    Only word characters, ``-`` and ``.`` survive, so the model cannot inject arXiv query
    operators or field prefixes. Raises ValueError if no usable term remains.
    """
    terms: list[str] = []
    for m in _TERM_RE.finditer(query):
        if m[1] is not None:
            words = [w for w in (_WORD_CLEAN_RE.sub(" ", m[1]).split()) if w]
            if words:
                terms.append('all:"' + " ".join(words) + '"')
        else:
            word = _WORD_CLEAN_RE.sub("", m[2]).strip("-.")
            if word and word.upper() not in _BOOLEAN_WORDS:
                terms.append(f"all:{word}")
        if len(terms) >= MAX_QUERY_TERMS:
            break
    if not terms:
        raise ValueError("The query contains no searchable words.")
    return " AND ".join(terms)


def _authors(authors: list[str]) -> list[str]:
    shown = [clean_text(a) for a in authors[:MAX_AUTHORS_SHOWN]]
    if len(authors) > MAX_AUTHORS_SHOWN:
        shown.append(f"et al. ({len(authors)} authors)")
    return shown


def arxiv_search(query: str, max_results: int = 5, deps: ToolDeps | None = None) -> ToolResult:
    """Search arXiv (read-only). ``data``: ``{"query", "results": [...]}``."""
    deps = deps or default_deps()

    def run() -> dict[str, Any]:
        args = ArxivSearchArgs(query=query, max_results=max_results)
        search_query = build_search_query(args.query)
        papers = deps.get_arxiv().search(search_query, args.max_results)
        kb_ids = deps.get_kb().paper_ids()
        results = [
            {
                "arxiv_id": p.arxiv_id,
                "version": p.version,
                "title": clean_text(p.title),
                "authors": _authors(p.authors),
                "year": p.year,
                "abstract_snippet": clean_text(p.abstract, ABSTRACT_SNIPPET_CHARS),
                "primary_category": p.categories[0] if p.categories else None,
                "url": f"https://arxiv.org/abs/{p.arxiv_id}",
                "already_in_kb": p.arxiv_id in kb_ids,
            }
            for p in papers[: args.max_results]
        ]
        return {"query": search_query, "results": results}

    return run_safely("arxiv", run)


# --------------------------------------------------------------------------- ingest


class ArxivIngestRequestArgs(ToolArgs):
    arxiv_id: ArxivIdStr = Field(
        description=(
            "arXiv ID of the paper to add, new style (2112.03570, 2112.03570v2) or old style "
            "(cs/0112017). Only use IDs the user gave or that arxiv_search returned."
        )
    )


def arxiv_ingest_request(arxiv_id: str, deps: ToolDeps | None = None) -> ToolResult:
    """Validate an ID and return an :class:`IngestRequest` awaiting user confirmation.

    Does NOT download or index anything; see :func:`confirm_ingest`.
    """
    deps = deps or default_deps()

    def run() -> IngestRequest:
        args = ArxivIngestRequestArgs(arxiv_id=arxiv_id)
        parsed = parse_arxiv_id(args.arxiv_id)
        in_kb = deps.in_kb(parsed.base)
        message = (
            f"arXiv:{parsed.base} is already in the knowledge base. Confirming re-ingests it "
            "and replaces its chunks."
            if in_kb
            else f"Adding arXiv:{parsed.base} to the knowledge base needs user confirmation."
        )
        if parsed.version is not None:
            message += f" The knowledge base stores the latest version, not v{parsed.version}."
        return IngestRequest(
            arxiv_id=parsed.base,
            requested_id=str(parsed),
            already_in_kb=in_kb,
            message=message,
        )

    return run_safely("knowledge_base", run)


def confirm_ingest(
    request: IngestRequest, sink: EventSink, deps: ToolDeps | None = None
) -> ToolResult:
    """Ingest a paper the user has explicitly confirmed. Call from the UI, never the model.

    ``data``: ``{"request": IngestRequest (status completed), "result": IngestResult fields,
    "findings": [Finding, ...]}``, where findings are the injection-scan results.
    """
    deps = deps or default_deps()

    def run() -> ToolResult | dict[str, Any]:
        if request.status != "pending_confirmation":
            return fail("knowledge_base", f"This request is already {request.status}.")
        parsed = parse_arxiv_id(request.arxiv_id)
        try:
            result = ingest_paper(
                parsed.base,
                sink,
                index=deps.index,
                settings=deps.settings,
                client=deps.get_arxiv(),
            )
        except IngestError as exc:  # messages are user-safe by contract
            return fail("knowledge_base", f"Could not add arXiv:{parsed.base}: {exc}")
        finally:
            deps.get_kb().refresh()
        done = request.model_copy(update={"status": "completed", "already_in_kb": True})
        findings = [f.model_dump(mode="json") for f in result.findings]
        return {
            "request": done.model_dump(mode="json"),
            "result": result.model_dump(mode="json", exclude={"findings"}),
            "findings": findings,
        }

    return run_safely("knowledge_base", run)


# --------------------------------------------------------------------------- bibtex


class BibtexExportArgs(ToolArgs):
    arxiv_ids: list[ArxivIdStr] = Field(
        min_length=1,
        max_length=BIBTEX_MAX_IDS,
        description="arXiv IDs to export, e.g. ['1610.05820', 'cs/0112017'] (max 50).",
    )


_UNESCAPED_SPECIAL = re.compile(r"(?<!\\)([&%#])")
_KEY_WORD = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = {"a", "an", "the", "on", "of", "in", "for", "and", "to", "with", "towards", "via"}


def _ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def _bib_value(text: str) -> str:
    """Escape BibTeX specials outside LaTeX commands; drop braces if they are unbalanced."""
    value = _UNESCAPED_SPECIAL.sub(r"\\\1", clean_text(text))
    depth = 0
    for ch in value:
        depth += {"{": 1, "}": -1}.get(ch, 0)
        if depth < 0:
            break
    if depth != 0:
        value = value.replace("{", "").replace("}", "")
    return value


def bibtex_key(meta: ArxivMetadata) -> str:
    last = meta.authors[0].split()[-1] if meta.authors else "anon"
    words = [w.lower() for w in _KEY_WORD.findall(_ascii(meta.title))]
    first = next((w for w in words if w not in _STOPWORDS), "paper")
    key = f"{''.join(_KEY_WORD.findall(_ascii(last))).lower() or 'anon'}{meta.year or ''}{first}"
    return key


def bibtex_entry(meta: ArxivMetadata, key: str, eprint: str) -> str:
    fields: list[tuple[str, str]] = [
        ("title", "{" + _bib_value(meta.title) + "}"),  # double braces keep capitalization
        ("author", " and ".join(_bib_value(a) for a in meta.authors) or "Anonymous"),
    ]
    if meta.year:
        fields.append(("year", str(meta.year)))
    fields += [("eprint", eprint), ("archivePrefix", "arXiv")]
    if meta.categories:
        fields.append(("primaryClass", meta.categories[0]))
    fields.append(("url", f"https://arxiv.org/abs/{eprint}"))
    body = ",\n".join(f"  {name} = {{{value}}}" for name, value in fields)
    return f"@misc{{{key},\n{body}\n}}"


def bibtex_export(arxiv_ids: list[str], deps: ToolDeps | None = None) -> ToolResult:
    """BibTeX ``@misc`` entries from arXiv metadata (one API request for all IDs).

    ``data``: ``{"bibtex": str, "entries": [{"arxiv_id", "key"}], "not_found": [ids]}``.
    """
    deps = deps or default_deps()

    def run() -> ToolResult | dict[str, Any]:
        args = BibtexExportArgs(arxiv_ids=arxiv_ids)
        requested: dict[str, str] = {}  # base -> id as requested (keeps an explicit version)
        for raw in args.arxiv_ids:
            requested.setdefault(parse_arxiv_id(raw).base, raw)
        metas = deps.get_arxiv().fetch_metadata_many(list(requested.values()))
        by_base: dict[str, ArxivMetadata] = {}
        for m in metas:
            by_base.setdefault(m.arxiv_id, m)

        entries: list[str] = []
        exported: list[dict[str, str]] = []
        not_found: list[str] = []
        used: set[str] = set()
        for base, raw in requested.items():
            meta = by_base.get(base)
            if meta is None:
                not_found.append(raw)
                continue
            key = stem = bibtex_key(meta)
            suffix = ord("a")
            while key in used:
                key = f"{stem}{chr(suffix)}"
                suffix += 1
            used.add(key)
            entries.append(bibtex_entry(meta, key, raw))
            exported.append({"arxiv_id": raw, "key": key})
        if not entries:
            return fail("arxiv", "arXiv has no paper for any of these IDs.")
        return {"bibtex": "\n\n".join(entries) + "\n", "entries": exported, "not_found": not_found}

    return run_safely("arxiv", run)
