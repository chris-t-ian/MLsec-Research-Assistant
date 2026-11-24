"""Chat with the research agent in the terminal (one session, conversation memory).

Usage:
    uv run python scripts/chat_cli.py                       # interactive
    uv run python scripts/chat_cli.py "Which papers cite 1610.05820?" "And what is LiRA?"
    uv run python scripts/chat_cli.py -s hybrid --stream --steps

Pending ingest requests are confirmed with a y/N prompt (interactive terminals only).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from typing import Any

from research_assistant.agent import (
    AgentConfigError,
    ResearchAgent,
    SessionState,
    TurnResult,
)
from research_assistant.config import ConfigError, get_settings

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")  # tool output / excerpts are untrusted

logger = logging.getLogger("chat_cli")


def _clean(text: str) -> str:
    return _CONTROL.sub(" ", text)


class ConsoleSink:
    """Agent steps and tool calls to stderr, streamed tokens (raw markers) to stdout."""

    def __init__(self, show_steps: bool, stream: bool) -> None:
        self.show_steps = show_steps
        self.stream = stream

    def step_started(self, step_id: str, name: str, input: Any) -> None:
        if self.show_steps or step_id.startswith("agent:"):
            print(f"  [{name}]", file=sys.stderr, flush=True)

    def step_finished(
        self, step_id: str, output: Any, metadata: dict[str, Any] | None = None
    ) -> None:
        if self.show_steps:
            ms = (metadata or {}).get("latency_ms")
            if ms is not None:
                print(f"  [{step_id}] {ms:.0f} ms", file=sys.stderr)

    def tool_result(self, name: str, args: dict[str, Any], result: Any) -> None:
        status = "ok" if getattr(result, "ok", False) else f"failed: {getattr(result, 'error', '')}"
        shown = _clean(json.dumps(args, ensure_ascii=False))[:160]
        print(f"  -> {name}({shown}) {status}", file=sys.stderr, flush=True)

    def token(self, text: str) -> None:
        if self.stream:
            print(text, end="", flush=True)

    def answer_ready(self, answer: Any, verifying: bool) -> None:
        if verifying and self.show_steps:
            print("\n  .. checking claims", file=sys.stderr, flush=True)


def print_turn(res: TurnResult) -> None:
    sys.stderr.flush()
    print("\n" + _clean(res.answer), flush=True)
    if res.citations:
        print("\nSources:")
        for c in res.citations:
            print(f"  [{c.number}] {c.label}: {_clean(c.title)} ({c.url}) | {c.chunk_id}")
    if res.tool_citations:
        print("\nTool results cited:")
        for t in res.tool_citations:
            print(f"  [T{t.number}] {t.name} ({'ok' if t.ok else 'failed'})")
    for e in res.security_events:
        score = f" score={e.score:.2f}" if e.score is not None else ""
        print(f"  ! security: {e.kind}{score}: {_clean(e.detail)}", file=sys.stderr)
    u = res.token_usage
    print(
        f"\n({res.turn_id}, model={res.model}, tools={len(res.tool_results)}, "
        f"tokens={u.total_tokens}, {res.latency_ms / 1000:.1f}s)",
        file=sys.stderr,
        flush=True,
    )


def handle_pending(agent: ResearchAgent, session: SessionState, res: TurnResult) -> None:
    for req in res.pending_ingests:
        print(f"\nIngest request: {_clean(req.message)}")
        if not sys.stdin.isatty():
            print("  (not confirmed: run interactively to confirm ingest requests)")
            continue
        answer = input(f"  Add arXiv:{req.arxiv_id} to the knowledge base? [y/N] ").strip()
        confirmed = answer.lower() in {"y", "yes"}
        result = agent.confirm_pending_ingest(
            session, req.arxiv_id, confirmed, ConsoleSink(False, False)
        )
        print(f"  {'Added.' if result.ok else _clean(result.error or 'Not added.')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("questions", nargs="*", help="questions to ask in one session")
    parser.add_argument("-s", "--strategy", default=None, help="name in strategies.yaml")
    parser.add_argument("--stream", action="store_true", help="stream raw answer tokens")
    parser.add_argument("--steps", action="store_true", help="show all pipeline steps")
    args = parser.parse_args(argv)

    try:
        get_settings()
        agent = ResearchAgent()
    except (ConfigError, AgentConfigError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    session = SessionState(strategy=args.strategy)
    sink = ConsoleSink(args.steps, args.stream)

    def ask(question: str) -> None:
        print(f"\nQ: {question}", flush=True)
        res = agent.run_turn(question, session, sink)
        print_turn(res)
        handle_pending(agent, session, res)

    if args.questions:
        for q in args.questions:
            ask(q)
        return 0

    print("Research assistant (empty line or Ctrl-D to quit).")
    while True:
        try:
            q = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q.strip():
            return 0
        ask(q)


if __name__ == "__main__":
    raise SystemExit(main())
