"""Parse, scan and chunk downloaded papers: data/raw/<id>/ -> data/processed/<id>.jsonl.

Usage:
    uv run python scripts/process_corpus.py                  # all papers in data/raw
    uv run python scripts/process_corpus.py --ids 1607.00133 2012.07805
    uv run python scripts/process_corpus.py --findings       # also list every finding

Prints chunks per paper, the papers that used the PDF fallback, and all flagged chunks.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from research_assistant.config import ConfigError, get_settings
from research_assistant.ingest.process import CorpusIngestReport, process_corpus

logger = logging.getLogger("process_corpus")


class ProgressSink:
    """Prints one line per finished paper."""

    def __init__(self, total: int) -> None:
        self._total = total
        self._done = 0

    def step_started(self, step_id: str, name: str, input: Any) -> None:
        pass

    def step_finished(
        self, step_id: str, output: Any, metadata: dict[str, Any] | None = None
    ) -> None:
        self._done += 1
        meta = metadata or {}
        arxiv_id = step_id.split(":", 1)[-1]
        extra = f" [{meta['source']}]" if "source" in meta else ""
        if meta.get("flagged"):
            extra += f" flagged={meta['flagged']}"
        print(f"[{self._done}/{self._total}] {arxiv_id}: {output}{extra}", flush=True)

    def tool_result(self, name: str, args: dict[str, Any], result: Any) -> None:
        pass

    def token(self, text: str) -> None:
        pass


def print_summary(report: CorpusIngestReport, show_findings: bool) -> None:
    papers = report.papers
    print("\n=== Chunks per paper ===")
    for p in papers:
        print(f"  {p.arxiv_id:<12} {p.n_chunks:>4} chunks  {p.n_sections:>3} sections  {p.source}")
    print(f"  total: {sum(p.n_chunks for p in papers)} chunks in {len(papers)} papers")

    fallback = [p for p in papers if p.source == "pdf"]
    print(f"\n=== PDF fallback ({len(fallback)}) ===")
    for p in fallback:
        print(f"  {p.arxiv_id:<12} {p.fallback_reason}")

    flagged = [(p, c) for p in papers for c in p.flagged]
    print(f"\n=== Flagged chunks ({len(flagged)}) ===")
    for _, c in flagged:
        print(f"  {c.chunk_id:<20} {','.join(c.flags):<12} {c.section}")
        for f in c.findings:
            print(f"      {f.severity:<6} {f.type} at {f.location}: {f.snippet[:100]}")

    sev: Counter[str] = Counter(f.severity for p in papers for f in p.findings)
    print(
        f"\n=== Findings: {sev['high']} high, {sev['medium']} medium, {sev['low']} low "
        f"(low = short hidden fragments, e.g. \\phantom alignment; not flagged) ==="
    )
    if show_findings:
        for p in papers:
            for f in p.findings:
                print(f"  {p.arxiv_id:<12} {f.severity:<6} {f.type:<20} {f.location}: {f.snippet}")

    if report.failed:
        print(f"\n=== Failed ({len(report.failed)}) ===")
        for f in report.failed:
            print(f"  {f.arxiv_id:<12} {f.error}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, help="input directory (default: settings)")
    parser.add_argument("--out-dir", type=Path, help="output directory (default: settings)")
    parser.add_argument("--ids", nargs="+", help="only process these arXiv IDs")
    parser.add_argument("--findings", action="store_true", help="also list all findings")
    args = parser.parse_args(argv)

    try:
        settings = get_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    paths = settings.paths.resolve()
    raw_dir = args.raw_dir or paths.raw_dir
    out_dir = args.out_dir or paths.processed_dir
    if not raw_dir.is_dir():
        print(f"No raw corpus at {raw_dir}; run scripts/build_corpus.py first.", file=sys.stderr)
        return 1

    n = len(args.ids) if args.ids else sum(1 for d in raw_dir.iterdir() if d.is_dir())
    report = process_corpus(raw_dir, out_dir, settings.chunking, ids=args.ids, sink=ProgressSink(n))
    print_summary(report, args.findings)
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
