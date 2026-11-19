"""Download the paper corpus listed in data/papers.yaml into data/raw/.

Usage:
    uv run python scripts/build_corpus.py               # all papers
    uv run python scripts/build_corpus.py --limit 3     # first 3 entries
    uv run python scripts/build_corpus.py --ids 1607.00133 2012.07805

Already-downloaded papers are skipped, so the script can be re-run to retry failures.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from research_assistant.config import ConfigError, get_settings
from research_assistant.ingest.arxiv_fetch import ArxivClient, build_corpus, load_paper_list

logger = logging.getLogger("build_corpus")


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
        arxiv_id = step_id.split(":", 1)[-1]
        extra = ""
        if metadata and "latex_available" in metadata:
            extra = " (LaTeX)" if metadata["latex_available"] else " (PDF only)"
        print(f"[{self._done}/{self._total}] {arxiv_id}: {output}{extra}", flush=True)

    def tool_result(self, name: str, args: dict[str, Any], result: Any) -> None:
        pass

    def token(self, text: str) -> None:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--papers", type=Path, help="paper list (default: settings)")
    parser.add_argument("--raw-dir", type=Path, help="output directory (default: settings)")
    parser.add_argument("--limit", type=int, help="only process the first N entries")
    parser.add_argument("--ids", nargs="+", help="only process these arXiv IDs")
    args = parser.parse_args(argv)

    try:
        settings = get_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    paths = settings.paths.resolve()
    papers_file = args.papers or paths.papers_file
    raw_dir = args.raw_dir or paths.raw_dir

    try:
        entries = load_paper_list(papers_file)
    except (OSError, ValueError) as exc:
        print(f"Could not read paper list {papers_file}: {exc}", file=sys.stderr)
        return 2
    if args.ids:
        wanted = set(args.ids)
        unknown = wanted - {e.arxiv_id for e in entries}
        if unknown:
            print(f"Not in {papers_file.name}: {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
        entries = [e for e in entries if e.arxiv_id in wanted]
    if args.limit is not None:
        entries = entries[: args.limit]

    print(f"Processing {len(entries)} papers into {raw_dir} (>= 3 s between arXiv requests)")
    with ArxivClient(settings.arxiv) as client:
        report = build_corpus(
            entries, client, raw_dir, settings.arxiv, sink=ProgressSink(len(entries))
        )

    print()
    print("Summary")
    print(f"  downloaded:      {len(report.downloaded)}")
    print(f"  skipped:         {len(report.skipped)}")
    print(f"  failed:          {len(report.failed)}")
    print(f"  LaTeX available: {len(report.latex_available)}")
    for failure in report.failed:
        print(f"    FAILED {failure.arxiv_id}: {failure.error}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
