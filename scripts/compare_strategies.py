"""Run every retrieval strategy on a few questions; print top-5 arXiv IDs and latency.

Usage:
    uv run python scripts/compare_strategies.py                    # 3 built-in questions
    uv run python scripts/compare_strategies.py "question" ...     # your own questions
    uv run python scripts/compare_strategies.py --strategies hybrid full

Strategies come from config/strategies.yaml. Models are loaded once before timing, so the
latencies show retrieval cost, not model start-up.
"""

from __future__ import annotations

import argparse
import logging
import sys

from research_assistant.config import ConfigError, get_settings
from research_assistant.ingest.index import ChunkIndexError
from research_assistant.retrieval.rerank import CrossEncoderReranker, RerankError
from research_assistant.retrieval.strategies import (
    RetrievalDeps,
    RetrievalError,
    StrategyConfigError,
    load_strategies,
    retrieve,
)

logger = logging.getLogger("compare_strategies")

# From eval/target_questions.txt (MIA #1, MIA #2, DP-SGD #15).
QUESTIONS = (
    "How does LiRA work, and why does it report TPR at low FPR?",
    "How do shadow-model attacks (Shokri et al.) differ from loss-threshold attacks "
    "(Yeom et al.) in assumptions and strength?",
    "Walk through DP-SGD: per-sample clipping, noise addition, and privacy accounting.",
)
TOP_N = 5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("questions", nargs="*", help="questions (default: 3 built-in)")
    parser.add_argument("--strategies", nargs="*", help="strategy names (default: all)")
    parser.add_argument("-v", "--verbose", action="store_true", help="show rewrites and notes")
    args = parser.parse_args(argv)

    try:
        settings = get_settings()
        strategies = load_strategies(settings.paths.resolve().strategies_file)
        deps = RetrievalDeps.from_settings(settings)
    except (ConfigError, StrategyConfigError, ChunkIndexError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    selected = args.strategies or list(strategies)
    unknown = [s for s in selected if s not in strategies]
    if unknown:
        print(f"Unknown strategies: {', '.join(unknown)}", file=sys.stderr)
        return 2
    if deps.llm is None:
        print("Note: no OPENROUTER_API_KEY; LLM steps fall back to the plain question.")

    print("Loading models ...", flush=True)
    deps.index.dense_search("warm up", 1)  # loads the embedding model
    _ = deps.catalog
    if any(strategies[s].rerank for s in selected) and isinstance(
        deps.reranker, CrossEncoderReranker
    ):
        try:
            deps.reranker.warm_up()
        except RerankError as exc:
            print(f"Warning: {exc}; rerank strategies will keep the fused order.")

    totals: dict[str, float] = dict.fromkeys(selected, 0.0)
    for qi, question in enumerate(args.questions or QUESTIONS, start=1):
        print(f"\n=== Q{qi}: {question}")
        print(f"  {'strategy':<20} {'ms':>8} {'tokens':>7}  top-{TOP_N} arXiv IDs")
        for name in selected:
            try:
                res = retrieve(question, strategies[name], None, deps)
            except RetrievalError as exc:
                print(f"  {name:<20} skipped: {exc}")
                continue
            except Exception as exc:  # friendly message, no traceback
                logger.debug("retrieval failed", exc_info=True)
                print(f"  {name:<20} failed ({type(exc).__name__})")
                continue
            ms = res.trace.total_latency_ms
            totals[name] += ms
            ids = " ".join(c.chunk.arxiv_id for c in res.chunks[:TOP_N])
            tokens = res.trace.token_usage.total_tokens
            print(f"  {name:<20} {ms:>8.0f} {tokens:>7}  {ids}")
            if args.verbose:
                stages = ", ".join(f"{s.stage}={s.latency_ms:.0f}" for s in res.trace.stages)
                print(f"  {'':<20} stages(ms): {stages}")
                if res.trace.filters:
                    print(
                        f"  {'':<20} filters: {res.trace.filters.model_dump(exclude_defaults=True)}"
                    )
                for rw in res.trace.rewrites:
                    print(f"  {'':<20} rewrite: {rw[:110]}")
            for s in res.trace.stages:
                for note in s.notes:
                    print(f"  {'':<20} note [{s.stage}]: {note}")

    n = len(args.questions or QUESTIONS)
    print("\nMean latency per question (ms):")
    for name in selected:
        print(f"  {name:<20} {totals[name] / n:>8.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
