"""Offline A/B evaluation: every strategy x every gold question through the agent.

Usage:
    uv run python eval/run_eval.py                                  # all strategies, dev split
    uv run python eval/run_eval.py --limit 5 --strategies baseline_dense hybrid
    uv run python eval/run_eval.py --questions 1 7 15
    uv run python eval/run_eval.py --verification off flag regenerate     # claim checks
    uv run python eval/run_eval.py --resume eval/results/20261122-120000
    uv run python eval/run_eval.py --split test --final             # final comparison only

--verification adds claim-verification modes as an experiment dimension (default: off
only); every (strategy, mode) pair is one arm. Each (arm, question) turn runs in a fresh
session with the models from config/settings.yaml at temperature 0, is judged by the judge
model and is appended to eval/results/<timestamp>/results.jsonl right away; --resume skips
finished turns. The run ends with summary.csv, report.md (with a verification-effect table
when several modes ran) and plots in the same directory. Test-split runs need
--final and are logged to eval/results/test_runs.log.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from research_assistant.abtest.gold import GoldFileError, file_sha256, load_gold
from research_assistant.abtest.report import summarize, verification_effect, write_report
from research_assistant.abtest.runner import (
    RESULTS_FILE,
    EvalRunError,
    ResultStore,
    RunManifest,
    append_test_log,
    build_agent,
    build_judge,
    build_measure_fn,
    build_turn_fn,
    git_info,
    load_manifest,
    run_eval,
    run_timestamp,
    save_manifest,
    utc_now,
)
from research_assistant.abtest.stats import DEFAULT_RESAMPLES, DEFAULT_SEED
from research_assistant.agent.graph import AgentConfigError
from research_assistant.config import PROJECT_ROOT, ConfigError, Settings, get_settings
from research_assistant.retrieval.strategies import StrategyConfigError, strategy_names

EVAL_DIR = PROJECT_ROOT / "eval"
DEFAULT_GOLD = EVAL_DIR / "gold_questions.yaml"
RESULTS_DIR = EVAL_DIR / "results"
TEST_LOG = RESULTS_DIR / "test_runs.log"

logger = logging.getLogger("run_eval")


class ConsoleProgress:
    """Prints one line per (strategy, question) to stderr."""

    def step_started(self, step_id: str, name: str, input: Any) -> None:
        print(f"{name} ...", file=sys.stderr, flush=True)

    def step_finished(
        self, step_id: str, output: Any, metadata: dict[str, Any] | None = None
    ) -> None:
        print(f"    {output}", file=sys.stderr, flush=True)

    def tool_result(self, name: str, args: dict[str, Any], result: Any) -> None:
        pass

    def token(self, text: str) -> None:
        pass

    def answer_ready(self, answer: Any, verifying: bool) -> None:
        pass


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def new_manifest(args: argparse.Namespace, settings: Settings) -> RunManifest:
    if args.split == "test" and not args.final:
        raise EvalRunError("Running on the test split requires --final.")
    if args.final and args.split != "test":
        raise EvalRunError("--final is only for --split test.")
    questions = load_gold(args.gold)
    unsplit = [q.id for q in questions if q.split is None]
    if unsplit:
        raise EvalRunError(
            f"{len(unsplit)} question(s) have no split; run scripts/split_gold.py first."
        )
    pool = [q for q in questions if q.split == args.split]
    if args.questions:
        wanted = [str(x) for x in args.questions]
        by_id = {q.id: q for q in questions}
        unknown = [w for w in wanted if w not in by_id]
        if unknown:
            raise EvalRunError(f"unknown question id(s): {', '.join(unknown)}")
        wrong = [w for w in wanted if by_id[w].split != args.split]
        if wrong:
            raise EvalRunError(f"question(s) {', '.join(wrong)} are not in the {args.split} split")
        pool = [q for q in pool if q.id in set(wanted)]
    if args.limit is not None:
        pool = pool[: args.limit]
    if not pool:
        raise EvalRunError("no questions selected")

    available = strategy_names(settings)
    strategies = args.strategies or available
    unknown_s = [s for s in strategies if s not in available]
    if unknown_s:
        raise EvalRunError(
            f"unknown strategy: {', '.join(unknown_s)} (available: {', '.join(available)})"
        )
    commit, dirty = git_info()
    run_id = run_timestamp()
    return RunManifest(
        run_id=run_id,
        created_at=utc_now(),
        split=args.split,
        final=args.final,
        strategies=list(dict.fromkeys(strategies)),
        verification_modes=list(dict.fromkeys(args.verification or ["off"])),
        question_ids=[q.id for q in pool],
        main_model=settings.models.main,
        judge_model=settings.models.judge,
        temperature=0.0,
        k=args.k,
        seed=args.seed,
        n_resamples=args.resamples,
        git_commit=commit,
        git_dirty=dirty,
        gold_file=_rel(args.gold),
        gold_sha256=file_sha256(args.gold),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strategies", nargs="+", help="names in config/strategies.yaml")
    parser.add_argument("--limit", type=int, help="first N questions of the split")
    parser.add_argument("--questions", nargs="+", help="question ids")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--final", action="store_true", help="required for --split test")
    parser.add_argument(
        "--verification",
        nargs="+",
        choices=["off", "flag", "remove", "regenerate"],
        help="claim-verification modes to compare (default: off)",
    )
    parser.add_argument("--resume", type=Path, help="run directory to continue")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--k", type=int, default=8, help="recall@k cutoff in chunks")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="bootstrap seed")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    args = parser.parse_args(argv)

    try:
        settings = get_settings()
        if args.resume:
            selection = (args.strategies, args.limit, args.questions, args.final, args.verification)
            if any(x not in (None, False) for x in selection) or args.split != "dev":
                raise EvalRunError(
                    "--resume reuses the run's manifest; do not combine it with "
                    "--strategies/--limit/--questions/--split/--final/--verification."
                )
            run_dir = args.resume
            manifest = load_manifest(run_dir)
            gold_path = PROJECT_ROOT / manifest.gold_file
            if file_sha256(gold_path) != manifest.gold_sha256:
                raise EvalRunError(
                    "the gold file changed since this run started; start a new run instead"
                )
            if (manifest.main_model, manifest.judge_model) != (
                settings.models.main,
                settings.models.judge,
            ):
                raise EvalRunError("models in config/settings.yaml changed since this run began")
            if manifest.split == "test":
                append_test_log(TEST_LOG, manifest, run_dir, "resume")
        else:
            manifest = new_manifest(args, settings)
            gold_path = args.gold
            run_dir = RESULTS_DIR / manifest.run_id
            save_manifest(run_dir, manifest)
            if manifest.split == "test":
                append_test_log(TEST_LOG, manifest, run_dir, "start")
        questions = load_gold(gold_path)
        print(
            f"Run {manifest.run_id}: {len(manifest.question_ids)} {manifest.split} questions x "
            f"{len(manifest.arms())} arms ({', '.join(manifest.arms())}) -> {_rel(run_dir)}",
            file=sys.stderr,
        )
        agent = build_agent(settings)
        turn_fn = build_turn_fn(settings, agent)
        judge = build_judge(settings)
        outcome = run_eval(
            manifest,
            questions,
            ResultStore(run_dir / RESULTS_FILE),
            turn_fn,
            judge,
            ConsoleProgress(),
            measure=build_measure_fn(agent),
        )
    except (ConfigError, GoldFileError, EvalRunError, AgentConfigError, StrategyConfigError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted. Continue with --resume.", file=sys.stderr)
        return 130

    summary = summarize(
        outcome.records,
        questions,
        manifest.arms(),
        k=manifest.k,
        seed=manifest.seed,
        n_resamples=manifest.n_resamples,
    )
    effect = verification_effect(
        outcome.records,
        questions,
        manifest.strategies,
        manifest.verification_modes,
        k=manifest.k,
        seed=manifest.seed,
        n_resamples=manifest.n_resamples,
    )
    report = write_report(
        run_dir, summary, manifest, questions, failed=outcome.failed, effect=effect
    )
    judge_errors = sum(1 for r in outcome.records if r.judge is None or not r.judge.ok)
    print(
        f"Done: {outcome.n_new} new, {outcome.n_rejudged} re-judged, "
        f"{len(outcome.failed)} failed, {judge_errors} judge error(s). Report: {_rel(report)}",
        file=sys.stderr,
    )
    if outcome.failed or judge_errors:
        print(f"Rerun with --resume {_rel(run_dir)} to fill the gaps.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
