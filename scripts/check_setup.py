"""Check that the app can start: configuration, secrets, strategies and the search index.

Usage:
    uv run python scripts/check_setup.py          # checks for the web app (incl. login secrets)
    uv run python scripts/check_setup.py --cli    # checks for CLI scripts (no login secrets)

Exits 1 with a readable list of problems, 0 when everything is ready.
"""

from __future__ import annotations

import argparse
import sys

from research_assistant.startup import format_problems, run_checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cli", action="store_true", help="skip the web-login secrets")
    args = parser.parse_args(argv)
    problems = run_checks("cli" if args.cli else "app")
    if problems:
        print(format_problems(problems), file=sys.stderr)
        return 1
    print("Setup OK: configuration, secrets, strategies and search index are ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
