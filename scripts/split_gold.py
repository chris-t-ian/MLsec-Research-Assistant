"""Assign the dev/test split of eval/gold_questions.yaml once (stratified, fixed seed).

Usage:
    uv run python scripts/split_gold.py --dry-run     # show the allocation, write nothing
    uv run python scripts/split_gold.py               # write `topic` and `split` fields

Strata are (type, topic). `topic` is the majority topic of a question's gold papers in
data/papers.yaml (out_of_scope questions form their own stratum); a hand-written `topic` is
kept. Each stratum gets a dev share proportional to its size. The script refuses to run if
any question already has a split, so splits are never reassigned.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from research_assistant.abtest.gold import GoldFileError, write_splits
from research_assistant.config import PROJECT_ROOT, ConfigError, get_settings

DEFAULT_GOLD = PROJECT_ROOT / "eval" / "gold_questions.yaml"
N_DEV = 20
SEED = 20261122


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--n-dev", type=int, default=N_DEV, help=f"default {N_DEV}")
    parser.add_argument("--seed", type=int, default=SEED, help=f"default {SEED}")
    parser.add_argument("--dry-run", action="store_true", help="print, do not write")
    args = parser.parse_args(argv)
    try:
        papers = get_settings().paths.resolve().papers_file
        report = write_splits(args.gold, papers, args.n_dev, args.seed, dry_run=args.dry_run)
    except (GoldFileError, ConfigError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"dev={report.n_dev} test={report.n_test} seed={report.seed}")
    print(f"{'stratum (type/topic)':<50} {'dev':>4} {'test':>5}")
    for stratum, counts in report.by_stratum.items():
        print(f"{stratum:<50} {counts['dev']:>4} {counts['test']:>5}")
    print("(dry run, nothing written)" if args.dry_run else f"Wrote splits to {args.gold}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
