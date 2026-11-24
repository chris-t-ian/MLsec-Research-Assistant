"""Ingest one paper end to end: fetch -> parse + scan -> chunk -> index.

:func:`ingest_paper` is used by ``scripts/ingest.py`` for the corpus and by the arXiv tool
for papers added at runtime. Progress goes through an :class:`EventSink`; every stage is one
``step_started``/``step_finished`` pair with step id ``ingest:<arxiv_id>:<stage>``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import get_args

from pydantic import BaseModel, Field

from research_assistant.config import Settings, get_settings
from research_assistant.events import EventSink
from research_assistant.ingest.arxiv_fetch import (
    ArxivClient,
    ArxivError,
    ArxivNotFoundError,
    download_paper,
    is_valid_arxiv_id,
    load_complete,
    load_paper_list,
    paper_dir,
)
from research_assistant.ingest.chunking import (
    QUARANTINED,
    SUSPICIOUS,
    TokenCounter,
    chunk_document,
)
from research_assistant.ingest.index import ChunkIndex, ChunkIndexError
from research_assistant.ingest.process import (
    MIN_LATEX_CHARS,
    IngestError,
    parse_paper,
    write_jsonl,
)
from research_assistant.models import Finding, Paper, PaperListEntry, Topic

logger = logging.getLogger(__name__)

STAGES = ("fetch", "parse_scan", "chunk", "index")
_VERSION_RE = re.compile(r"v\d+$")


class IngestResult(BaseModel):
    arxiv_id: str
    title: str
    topic: Topic
    downloaded: bool = Field(description="False if the paper was already in raw_dir.")
    source: str
    fallback_reason: str | None = None
    n_findings: int = Field(ge=0)
    n_chunks: int = Field(ge=0)
    n_quarantined: int = Field(ge=0)
    n_suspicious: int = Field(ge=0)
    findings: list[Finding] = Field(
        default_factory=list, description="Injection-scan findings from the paper source."
    )


def resolve_topic(arxiv_id: str, papers_file: Path, stored: Topic | None) -> Topic:
    """Topic from data/papers.yaml (the source of truth), else from meta.json, else 'other'."""
    if papers_file.is_file():
        try:
            for entry in load_paper_list(papers_file):
                if entry.arxiv_id == arxiv_id:
                    return entry.topic
        except ValueError as exc:
            logger.warning("Ignoring invalid paper list: %s", exc)
    return stored or "other"


def _fetch(
    arxiv_id: str, raw_dir: Path, topic: Topic, settings: Settings, client: ArxivClient | None
) -> tuple[Paper, bool]:
    existing = load_complete(paper_dir(raw_dir, arxiv_id))
    if existing is not None:
        return existing, False
    raw_dir.mkdir(parents=True, exist_ok=True)
    own = client is None
    client = client or ArxivClient(settings.arxiv)
    try:
        paper = download_paper(
            PaperListEntry(arxiv_id=arxiv_id, topic=topic),
            client,
            raw_dir,
            settings.arxiv.max_extract_mb * 1024 * 1024,
        )
    except ArxivNotFoundError as exc:
        raise IngestError(f"arXiv has no paper {arxiv_id}") from exc
    except (ArxivError, OSError) as exc:
        raise IngestError(f"Could not download {arxiv_id} from arXiv") from exc
    finally:
        if own:
            client.close()
    return paper, True


def ingest_paper(
    arxiv_id: str,
    sink: EventSink,
    *,
    index: ChunkIndex | None = None,
    settings: Settings | None = None,
    topic: Topic | None = None,
    client: ArxivClient | None = None,
    count_tokens: TokenCounter | None = None,
    min_latex_chars: int = MIN_LATEX_CHARS,
) -> IngestResult:
    """Fetch (unless already downloaded), parse, scan, chunk and index one paper.

    Re-running for the same paper replaces its chunks in the index. ``topic`` overrides the
    topic from data/papers.yaml. Raises :class:`IngestError` (with a user-safe message) on
    failure; the failing stage is reported to ``sink`` with output ``"failed"``.
    """
    settings = settings or get_settings()
    paths = settings.paths.resolve()
    arxiv_id = arxiv_id.strip()
    if not is_valid_arxiv_id(arxiv_id):
        raise IngestError(f"{arxiv_id[:40]!r} is not a valid arXiv ID")
    arxiv_id = _VERSION_RE.sub("", arxiv_id)  # the index holds one version per paper
    if topic is not None and topic not in get_args(Topic):
        raise IngestError(f"Unknown topic {topic!r}")

    def stage_of(step: str) -> str:
        return step.rsplit(":", 1)[-1]

    def run_stage(stage: str, payload: dict[str, object]) -> str:
        step = f"ingest:{arxiv_id}:{stage}"
        sink.step_started(step, stage, payload)
        return step

    def fail(step: str, exc: Exception) -> IngestError:
        safe = isinstance(exc, IngestError | ChunkIndexError)
        msg = str(exc) if safe else f"{arxiv_id}: {stage_of(step)} failed ({type(exc).__name__})"
        sink.step_finished(step, "failed", {"error": msg})
        return exc if isinstance(exc, IngestError) else IngestError(msg)

    # fetch
    step = run_stage("fetch", {"arxiv_id": arxiv_id})
    try:
        stored = load_complete(paper_dir(paths.raw_dir, arxiv_id))
        chosen: Topic = topic or resolve_topic(
            arxiv_id, paths.papers_file, stored.topic if stored else None
        )
        paper, downloaded = _fetch(arxiv_id, paths.raw_dir, chosen, settings, client)
    except IngestError as exc:
        raise fail(step, exc) from None
    sink.step_finished(
        step,
        "downloaded" if downloaded else "already downloaded",
        {"title": paper.title, "latex_available": paper.latex_available},
    )

    # parse + scan (parse_paper picks LaTeX or PDF and runs the matching scanner)
    step = run_stage("parse_scan", {"arxiv_id": arxiv_id})
    try:
        doc, findings, fallback = parse_paper(
            paper_dir(paths.raw_dir, arxiv_id), arxiv_id, min_latex_chars
        )
    except IngestError as exc:
        raise fail(step, exc) from None
    sink.step_finished(
        step,
        f"{len(doc.sections)} sections from {doc.source}",
        {
            "source": doc.source,
            "fallback_reason": fallback,
            "findings": len(findings),
            "high_severity": sum(f.severity == "high" for f in findings),
        },
    )

    # chunk (also refreshes data/processed/<id>.jsonl)
    step = run_stage("chunk", {"sections": len(doc.sections)})
    try:
        chunks = chunk_document(doc, paper, findings, settings.chunking, count_tokens)
        out = paths.processed_dir / f"{paper_dir(paths.raw_dir, arxiv_id).name}.jsonl"
        write_jsonl(out, (c.model_dump_json() for c in chunks))
    except (OSError, ValueError) as exc:
        raise fail(step, exc) from exc
    n_quarantined = sum(QUARANTINED in c.flags for c in chunks)
    n_suspicious = sum(SUSPICIOUS in c.flags for c in chunks)
    sink.step_finished(
        step,
        f"{len(chunks)} chunks",
        {"quarantined": n_quarantined, "suspicious": n_suspicious},
    )

    # index
    step = run_stage("index", {"chunks": len(chunks), "topic": chosen})
    try:
        index = index or ChunkIndex.from_config(paths.index_dir, settings.retrieval)
        index.replace_paper(arxiv_id, chunks, chosen)
    except (ChunkIndexError, OSError, ValueError) as exc:
        raise fail(step, exc) from exc
    sink.step_finished(step, f"indexed {len(chunks)} chunks", {"topic": chosen})

    return IngestResult(
        arxiv_id=arxiv_id,
        title=paper.title,
        topic=chosen,
        downloaded=downloaded,
        source=doc.source,
        fallback_reason=fallback,
        n_findings=len(findings),
        n_chunks=len(chunks),
        n_quarantined=n_quarantined,
        n_suspicious=n_suspicious,
        findings=findings,
    )


__all__ = ["STAGES", "IngestResult", "ingest_paper", "resolve_topic"]
