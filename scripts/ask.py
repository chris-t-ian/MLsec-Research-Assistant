"""Answer a question from the indexed papers with citations (streams the raw answer live).

Usage:
    uv run python scripts/ask.py
    uv run python scripts/ask.py "How does DP-SGD bound per-example influence?" -s hybrid
    uv run python scripts/ask.py --datamarking --quiet
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from typing import Any

from research_assistant.config import ConfigError, get_settings
from research_assistant.generation.answer import AnswerDeps, GenerationError, answer
from research_assistant.retrieval.strategies import RetrievalError, StrategyConfigError

DEFAULT_QUESTION = (
    "How does the likelihood ratio attack improve on earlier membership inference attacks?"
)
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")  # excerpts are untrusted text

logger = logging.getLogger("ask")


class ConsoleSink:
    """Prints step progress to stderr and streamed tokens to stdout."""

    def __init__(self, show_tokens: bool) -> None:
        self.show_tokens = show_tokens

    def step_started(self, step_id: str, name: str, input: Any) -> None:
        print(f"[{name}] ...", file=sys.stderr, flush=True)

    def step_finished(
        self, step_id: str, output: Any, metadata: dict[str, Any] | None = None
    ) -> None:
        ms = (metadata or {}).get("latency_ms")
        if ms is not None:
            print(f"[{step_id.rsplit(':', 1)[-1]}] done in {ms:.0f} ms", file=sys.stderr)

    def tool_result(self, name: str, args: dict[str, Any], result: Any) -> None:
        pass

    def token(self, text: str) -> None:
        if self.show_tokens:
            print(text, end="", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("-s", "--strategy", default="hybrid", help="name in strategies.yaml")
    parser.add_argument("--datamarking", action="store_true", help="enable datamarking")
    parser.add_argument("--quiet", action="store_true", help="do not stream raw tokens")
    args = parser.parse_args(argv)

    try:
        settings = get_settings()
        if args.datamarking:
            gen = settings.generation.model_copy(update={"datamarking": True})
            settings = settings.model_copy(update={"generation": gen})
        deps = AnswerDeps.from_settings(settings)
    except (ConfigError, GenerationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Q: {args.question}\n", flush=True)
    if not args.quiet:
        print("--- streamed (raw markers) ---")
    try:
        ans = answer(args.question, args.strategy, ConsoleSink(not args.quiet), deps)
    except (RetrievalError, GenerationError, StrategyConfigError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # friendly message, no traceback
        logger.debug("ask failed", exc_info=True)
        print(f"\nUnexpected error ({type(exc).__name__}).", file=sys.stderr)
        return 1

    print("\n\n--- answer ---")
    print(ans.text)
    print("\n--- sources ---")
    for c in ans.citations:
        excerpt = _CONTROL.sub(" ", c.excerpt)
        print(f"[{c.number}] {c.label}: {c.title} ({c.url})")
        print(f"    section: {c.section or '-'} | chunk: {c.chunk_id}")
        print(f'    "{excerpt}"')
    if ans.citation_errors:
        print("\n--- citation errors (removed) ---")
        for e in ans.citation_errors:
            print(f"{e.marker}: {e.reason}")
    u = ans.token_usage
    print(
        f"\nstrategy={ans.strategy} model={ans.model} "
        f"tokens(in/out/total)={u.input_tokens}/{u.output_tokens}/{u.total_tokens} "
        f"latency={ans.latency_ms / 1000:.1f}s (generation {ans.generation_latency_ms / 1000:.1f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
