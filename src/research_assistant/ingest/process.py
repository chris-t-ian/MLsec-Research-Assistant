"""Turn downloaded papers (data/raw/<id>/) into chunk files (data/processed/<id>.jsonl).

Per paper: parse LaTeX (scan the flattened source for injections first); if there is no
usable LaTeX, fall back to the PDF (scanned with the PDF scanner). Then chunk and write
one JSON object per :class:`Chunk` line. Progress is reported via an :class:`EventSink`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from research_assistant.config import ChunkingConfig
from research_assistant.events import EventSink, NullSink
from research_assistant.ingest.arxiv_fetch import META_FILE, PDF_FILE, SOURCE_DIR
from research_assistant.ingest.chunking import (
    QUARANTINED,
    TokenCounter,
    chunk_document,
    chunk_findings,
)
from research_assistant.ingest.injection_scan import scan_latex, scan_pdf
from research_assistant.ingest.latex_parse import LatexParse, LatexParseError, parse_latex
from research_assistant.ingest.pdf_parse import PdfParseError, parse_pdf
from research_assistant.models import Finding, Paper, ParsedDocument

logger = logging.getLogger(__name__)

# LaTeX output shorter than this is treated as a failed parse (e.g. a stub main file).
MIN_LATEX_CHARS = 2000


class IngestError(RuntimeError):
    """A paper could not be parsed from either LaTeX or PDF."""


class FlaggedChunk(BaseModel):
    chunk_id: str
    section: str | None
    flags: list[str]
    findings: list[Finding] = Field(default_factory=list)


class PaperReport(BaseModel):
    arxiv_id: str
    source: Literal["latex", "pdf"]
    fallback_reason: str | None = None
    n_sections: int = Field(ge=0)
    n_chunks: int = Field(ge=0)
    findings: list[Finding] = Field(default_factory=list)
    flagged: list[FlaggedChunk] = Field(default_factory=list)
    output: Path

    @property
    def n_quarantined(self) -> int:
        return sum(QUARANTINED in f.flags for f in self.flagged)


class FailedIngest(BaseModel):
    arxiv_id: str
    error: str


class CorpusIngestReport(BaseModel):
    papers: list[PaperReport] = Field(default_factory=list)
    failed: list[FailedIngest] = Field(default_factory=list)


def _load_paper(paper_dir: Path) -> Paper | None:
    meta = paper_dir / META_FILE
    if not meta.is_file():
        return None
    try:
        return Paper.model_validate_json(meta.read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        logger.warning("Invalid %s in %s", META_FILE, paper_dir.name)
        return None


def parse_paper(
    paper_dir: Path, arxiv_id: str, min_latex_chars: int = MIN_LATEX_CHARS
) -> tuple[ParsedDocument, list[Finding], str | None]:
    """Parse one paper; returns (document, findings, fallback_reason or None)."""
    source_dir = paper_dir / SOURCE_DIR
    latex: LatexParse | None = None
    reason: str
    if source_dir.is_dir():
        try:
            latex = parse_latex(source_dir, arxiv_id)
            n_chars = sum(len(s.text) for s in latex.document.sections)
            if n_chars >= min_latex_chars:
                return latex.document, _scan_latex(latex), None
            reason = f"LaTeX produced only {n_chars} characters"
        except LatexParseError as exc:
            reason = str(exc)
        except (OSError, RecursionError, ValueError) as exc:
            reason = f"LaTeX parsing failed ({type(exc).__name__})"
    else:
        reason = "no LaTeX source"

    pdf = paper_dir / PDF_FILE
    try:
        if not pdf.is_file():
            raise PdfParseError("no PDF available")
        doc = parse_pdf(pdf, arxiv_id)
        findings = scan_pdf(pdf)
    except (PdfParseError, RuntimeError, ValueError) as exc:
        if latex is not None:  # short LaTeX output still beats nothing
            logger.info("%s: %s; PDF unusable, keeping LaTeX", arxiv_id, reason)
            return latex.document, _scan_latex(latex), None
        raise IngestError(f"{arxiv_id}: {reason}; {exc}") from exc
    return doc, findings, reason


def _scan_latex(parsed: LatexParse) -> list[Finding]:
    """Scan the raw flattened source and attach the section each finding belongs to."""
    return [
        f.model_copy(update={"section_idx": parsed.section_at(f.offset or 0)})
        for f in scan_latex(parsed.flat)
    ]


def write_jsonl(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line + "\n")
    os.replace(tmp, path)


def process_paper(
    paper_dir: Path,
    out_dir: Path,
    config: ChunkingConfig | None = None,
    count_tokens: TokenCounter | None = None,
    min_latex_chars: int = MIN_LATEX_CHARS,
) -> PaperReport:
    arxiv_id = paper_dir.name
    paper = _load_paper(paper_dir)
    doc, findings, fallback = parse_paper(paper_dir, arxiv_id, min_latex_chars)
    chunks = chunk_document(doc, paper, findings, config, count_tokens)
    reasons = chunk_findings(doc, chunks, findings)
    out = out_dir / f"{arxiv_id}.jsonl"
    write_jsonl(out, (c.model_dump_json() for c in chunks))
    return PaperReport(
        arxiv_id=arxiv_id,
        source=doc.source,
        fallback_reason=fallback,
        n_sections=len(doc.sections),
        n_chunks=len(chunks),
        findings=findings,
        flagged=[
            FlaggedChunk(
                chunk_id=c.id, section=c.section, flags=c.flags, findings=reasons.get(c.id, [])
            )
            for c in chunks
            if c.flags
        ],
        output=out,
    )


def process_corpus(
    raw_dir: Path,
    out_dir: Path,
    config: ChunkingConfig | None = None,
    ids: Iterable[str] | None = None,
    sink: EventSink | None = None,
    count_tokens: TokenCounter | None = None,
    min_latex_chars: int = MIN_LATEX_CHARS,
) -> CorpusIngestReport:
    sink = sink or NullSink()
    wanted = set(ids) if ids is not None else None
    dirs = sorted(
        d
        for d in raw_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".") and (wanted is None or d.name in wanted)
    )
    report = CorpusIngestReport()
    for d in dirs:
        step = f"process:{d.name}"
        sink.step_started(step, "process_paper", {"arxiv_id": d.name})
        try:
            r = process_paper(d, out_dir, config, count_tokens, min_latex_chars)
        except IngestError as exc:
            logger.warning("%s", exc)
            report.failed.append(FailedIngest(arxiv_id=d.name, error=str(exc)))
            sink.step_finished(step, "failed", {"error": str(exc)})
            continue
        report.papers.append(r)
        sink.step_finished(
            step,
            f"{r.n_chunks} chunks",
            {
                "source": r.source,
                "fallback_reason": r.fallback_reason,
                "findings": len(r.findings),
                "flagged": len(r.flagged),
            },
        )
    return report
