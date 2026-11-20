"""Run queries against the dense and BM25 indexes and print the top chunks (manual check).

Usage:
    uv run python scripts/query_index.py                       # three built-in sample queries
    uv run python scripts/query_index.py "how does DP-SGD clip gradients?" -k 3
    uv run python scripts/query_index.py "..." --mode bm25 --include-quarantined
"""

from __future__ import annotations

import argparse
import re
import sys

from research_assistant.config import ConfigError, get_settings
from research_assistant.ingest.index import ChunkIndex, ChunkIndexError
from research_assistant.models import RetrievalResult

SAMPLE_QUERIES = (
    "How does DP-SGD bound the contribution of a single training example?",
    "How do shadow models enable membership inference attacks?",
    "Can large language models leak memorized training data verbatim?",
)
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")  # chunk text is untrusted


def _show(hits: list[RetrievalResult], width: int) -> None:
    for h in hits:
        c = h.chunk
        snippet = _CONTROL.sub(" ", " ".join(c.text.split()))[:width]
        print(f"  {h.rank}. [{h.score:.3f}] {c.id}  {c.title[:60]} / {c.section or '-'}")
        print(f"     {snippet}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("queries", nargs="*", help="queries (default: built-in samples)")
    parser.add_argument("-k", type=int, default=3, help="hits per retriever (default 3)")
    parser.add_argument("--mode", choices=("both", "dense", "bm25"), default="both")
    parser.add_argument("--include-quarantined", action="store_true")
    parser.add_argument("--width", type=int, default=240, help="snippet length in characters")
    args = parser.parse_args(argv)

    try:
        settings = get_settings()
        paths = settings.paths.resolve()
        index = ChunkIndex.from_config(paths.index_dir, settings.retrieval)
    except (ConfigError, ChunkIndexError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    if index.count() == 0:
        print("The index is empty; run scripts/ingest.py first.", file=sys.stderr)
        return 1

    for query in args.queries or SAMPLE_QUERIES:
        print(f"\n=== {query}")
        if args.mode in ("both", "dense"):
            print(" dense:")
            _show(index.dense_search(query, args.k, args.include_quarantined), args.width)
        if args.mode in ("both", "bm25"):
            print(" bm25:")
            _show(index.bm25_search(query, args.k, args.include_quarantined), args.width)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
