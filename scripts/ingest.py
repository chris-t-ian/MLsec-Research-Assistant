"""Index papers into Chroma + BM25 (data/index/): fetch -> parse + scan -> chunk -> index.

Usage:
    uv run python scripts/ingest.py                      # index papers in data/raw not yet indexed
    uv run python scripts/ingest.py --rebuild            # wipe the index, re-index everything
    uv run python scripts/ingest.py --paper 1607.00133   # (re-)index one paper (fetched if missing)

Papers are ingested one by one; a failing paper is reported and skipped.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from research_assistant.config import ConfigError, get_settings
from research_assistant.ingest.index import ChunkIndex, ChunkIndexError
from research_assistant.ingest.pipeline import IngestResult, ingest_paper
from research_assistant.ingest.process import IngestError

logger = logging.getLogger("ingest")


class TqdmSink:
    """Shows the current paper and stage in the progress bar."""

    def __init__(self, bar: tqdm) -> None:  # type: ignore[type-arg]
        self._bar = bar

    def step_started(self, step_id: str, name: str, input: Any) -> None:
        self._bar.set_postfix_str(step_id.removeprefix("ingest:"), refresh=True)

    def step_finished(
        self, step_id: str, output: Any, metadata: dict[str, Any] | None = None
    ) -> None:
        if output == "failed":
            self._bar.write(f"  {step_id}: {(metadata or {}).get('error', 'failed')}")

    def tool_result(self, name: str, args: dict[str, Any], result: Any) -> None:
        pass

    def token(self, text: str) -> None:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--rebuild", action="store_true", help="delete the index first")
    group.add_argument(
        "--paper", action="append", metavar="ID", help="(re-)index this arXiv ID (repeatable)"
    )
    args = parser.parse_args(argv)

    try:
        settings = get_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    paths = settings.paths.resolve()

    if args.paper:
        ids: list[str] = args.paper
    else:
        if not paths.raw_dir.is_dir():
            print(f"No raw corpus at {paths.raw_dir}; run scripts/build_corpus.py first.")
            return 1
        ids = sorted(
            d.name for d in paths.raw_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
        )

    if args.rebuild:
        ChunkIndex.wipe(paths.index_dir)
    try:
        index = ChunkIndex.from_config(paths.index_dir, settings.retrieval, autosave=False)
    except ChunkIndexError as exc:
        print(f"Index error: {exc}", file=sys.stderr)
        return 2

    skipped: list[str] = []
    if not args.paper:
        done = index.paper_ids()
        skipped = [i for i in ids if i in done]
        ids = [i for i in ids if i not in done]

    results: list[IngestResult] = []
    failed: list[tuple[str, str]] = []
    try:
        with logging_redirect_tqdm(), tqdm(ids, unit="paper") as bar:
            sink = TqdmSink(bar)
            for arxiv_id in bar:
                try:
                    results.append(ingest_paper(arxiv_id, sink, index=index, settings=settings))
                except IngestError as exc:
                    failed.append((arxiv_id, str(exc)))
                if len(results) % 10 == 0:
                    index.save()  # checkpoint: an interrupted run resumes from here
    finally:
        index.save()

    n_chunks = sum(r.n_chunks for r in results)
    print(
        f"\nIndexed {len(results)} papers ({n_chunks} chunks); "
        f"skipped {len(skipped)} already indexed."
    )
    print(f"Index now holds {index.count()} chunks from {len(index.paper_ids())} papers.")
    pdf = [r for r in results if r.source == "pdf"]
    if pdf:
        print(f"PDF fallback: {', '.join(r.arxiv_id for r in pdf)}")
    quarantined = [(r.arxiv_id, r.n_quarantined) for r in results if r.n_quarantined]
    if quarantined:
        print("Quarantined chunks: " + ", ".join(f"{a} ({n})" for a, n in quarantined))
    if failed:
        print(f"Failed ({len(failed)}):")
        for arxiv_id, err in failed:
            print(f"  {arxiv_id}: {err}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
