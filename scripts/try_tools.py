"""Smoke test: run every agent tool once against the real APIs and print the ToolResults.

    uv run python scripts/try_tools.py
    uv run python scripts/try_tools.py --confirm-ingest 2112.03570   # asks before ingesting

The tools go through the LangChain StructuredTool wrappers (as the agent will call them);
``confirm_ingest`` is not a tool and only runs with --confirm-ingest after a y/N prompt,
because it downloads, parses and indexes the paper into data/index.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from research_assistant.events import RecordingSink
from research_assistant.models import IngestRequest, ToolResult
from research_assistant.tools import build_tools, confirm_ingest, default_deps

CALLS: list[tuple[str, dict[str, Any]]] = [
    ("arxiv_search", {"query": "membership inference attacks", "max_results": 3}),
    ("arxiv_ingest_request", {"arxiv_id": "2112.03570v2"}),
    ("citation_graph", {"arxiv_id": "1610.05820", "direction": "citations", "limit": 5}),
    (
        "dp_accountant",
        {
            "mode": "epsilon",
            "noise_multiplier": 1.1,
            "sample_rate": 256 / 60000,
            "steps": 10000,
            "delta": 1e-5,
            "accountant": "rdp",
            "dataset_size": 60000,
        },
    ),
    (
        "dp_accountant",
        {
            "mode": "noise",
            "target_epsilon": 3.0,
            "sample_rate": 256 / 60000,
            "steps": 10000,
            "delta": 1e-5,
            "accountant": "pld",
        },
    ),
    ("bibtex_export", {"arxiv_ids": ["1610.05820", "cs/0112017"]}),
]


def show(name: str, args: dict[str, Any], result: ToolResult, seconds: float) -> None:
    print(f"\n=== {name}({json.dumps(args)})  [{seconds:.1f}s]")
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-ingest", metavar="ARXIV_ID", help="also ingest this paper")
    opts = parser.parse_args()

    deps = default_deps()
    tools = {t.name: t for t in build_tools(deps)}
    failures = 0
    for i, (name, args) in enumerate(CALLS):
        start = time.perf_counter()
        msg = tools[name].invoke(
            {"type": "tool_call", "id": f"call-{i}", "name": name, "args": args}
        )
        result: ToolResult = msg.artifact
        show(name, args, result, time.perf_counter() - start)
        failures += not result.ok

    if opts.confirm_ingest:
        req = (
            tools["arxiv_ingest_request"]
            .invoke(
                {
                    "type": "tool_call",
                    "id": "ingest",
                    "name": "arxiv_ingest_request",
                    "args": {"arxiv_id": opts.confirm_ingest},
                }
            )
            .artifact
        )
        if not req.ok:
            print(f"\nIngest request rejected: {req.error}")
            return 1
        request = IngestRequest.model_validate(req.data)
        answer = input(f"\n{request.message}\nIngest arXiv:{request.arxiv_id} now? [y/N] ")
        if answer.strip().lower() == "y":
            sink = RecordingSink()
            start = time.perf_counter()
            result = confirm_ingest(request, sink, deps)
            show(
                "confirm_ingest",
                {"arxiv_id": request.arxiv_id},
                result,
                time.perf_counter() - start,
            )
            for event in sink.events:
                if event.kind == "step_finished":
                    print(f"  {event.data['step_id']}: {event.data['output']}")
            failures += not result.ok
        else:
            print("Not ingested.")

    print(f"\n{failures} tool call(s) failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
