"""Retrieval-only tuning: strategies x parameter grid on the dev questions, no generation.

Usage:
    uv run python eval/tune_retrieval.py                                   # all strategies
    uv run python eval/tune_retrieval.py --strategies baseline_dense full \\
        --grid top_k=5,8,12 --grid fetch_k=30,60 --grid max_chunks_per_paper=2,3
    uv run python eval/tune_retrieval.py --limit 5 --grid top_k=5,8
    uv run python eval/tune_retrieval.py --resume eval/results/tuning-20261123-100000
    uv run python eval/tune_retrieval.py --split test --final \
        --strategies baseline_dense full full_tuned             # confirm the dev choice

Each (configuration, question) runs the retrieval pipeline on the gold question and scores
the selected chunks against gold_arxiv_ids (paper level). No answer model, no judge. Query
translation and self-query use models.retrieval_tuning from config/settings.yaml (no
fallback), and their outputs are cached per run, so grid variants share the same rewrites.
Results go to eval/results/tuning-<timestamp>/ (results.jsonl, llm_cache.jsonl,
summary.csv, report.md). Tuning (--grid) runs on the dev split only: tuning on test
questions would leak them into the choice. --split test --final confirms fixed strategies
from strategies.yaml on the whole test split (no grid) in eval/results/confirm-<timestamp>/
and is logged to eval/results/test_runs.log.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from research_assistant.abtest.gold import GoldFileError, file_sha256, load_gold
from research_assistant.abtest.retrieval_tuning import (
    LLM_CACHE_FILE,
    RESULTS_FILE,
    CachedLLM,
    CachedReranker,
    RecordStore,
    TuningError,
    TuningManifest,
    check_split_selection,
    expand_grid,
    load_manifest,
    manifest_configs,
    parse_grid,
    render_report,
    run_tuning,
    save_manifest,
    summarize_tuning,
    variants_from_manifest,
    write_summary_csv,
)
from research_assistant.abtest.runner import append_test_log, git_info, run_timestamp, utc_now
from research_assistant.abtest.stats import DEFAULT_RESAMPLES, DEFAULT_SEED
from research_assistant.config import PROJECT_ROOT, ConfigError, Settings, get_settings
from research_assistant.ingest.chunking import tiktoken_counter
from research_assistant.llm import ChatStructuredLLM, LLMConfigError, get_chat_model
from research_assistant.models import RetrievalResult
from research_assistant.retrieval.strategies import (
    RetrievalDeps,
    StrategyConfig,
    StrategyConfigError,
    load_strategies,
    retrieve,
)

EVAL_DIR = PROJECT_ROOT / "eval"
DEFAULT_GOLD = EVAL_DIR / "gold_questions.yaml"
RESULTS_DIR = EVAL_DIR / "results"
TEST_LOG = RESULTS_DIR / "test_runs.log"
REFERENCE = "baseline_dense"

logger = logging.getLogger("tune_retrieval")


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def rewriter_model(settings: Settings) -> str:
    return settings.models.retrieval_tuning or settings.models.judge


def build_rewriter(settings: Settings, model: str) -> ChatStructuredLLM:
    """The rewriter at temperature 0 with no silent fallback to another model."""
    models = settings.models.model_copy(update={"fallback": model})
    llm_cfg = settings.llm.model_copy(update={"temperature": 0.0})
    pinned = settings.model_copy(update={"models": models, "llm": llm_cfg})
    return ChatStructuredLLM(
        get_chat_model("judge", pinned, model=model), max_attempts=settings.llm.max_retries
    )


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--strategies", nargs="+", help="base strategies (default: all)")
    p.add_argument(
        "--grid",
        action="append",
        default=[],
        metavar="PARAM=V1,V2",
        help="values to try for fetch_k, top_k, rerank_k or max_chunks_per_paper (repeatable)",
    )
    p.add_argument("--questions", nargs="+", help="question ids (default: whole dev split)")
    p.add_argument("--limit", type=int, help="first N questions of the split")
    p.add_argument("--split", default="dev", help="dev (tuning) or test (with --final)")
    p.add_argument(
        "--final", action="store_true", help="confirm fixed strategies on the test split"
    )
    p.add_argument("--reference", default=REFERENCE, help=f"comparison base (default {REFERENCE})")
    p.add_argument("--resume", type=Path, help="tuning run directory to continue")
    p.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    return p.parse_args(argv)


def new_manifest(args: argparse.Namespace, settings: Settings) -> TuningManifest:
    check_split_selection(
        args.split,
        final=args.final,
        grid=bool(args.grid),
        subset=bool(args.questions or args.limit),
    )
    questions = [q for q in load_gold(args.gold) if q.split == args.split]
    if not questions:
        raise TuningError(f"no {args.split} questions; run scripts/split_gold.py first")
    if args.questions:
        wanted = set(args.questions)
        unknown = wanted - {q.id for q in questions}
        if unknown:
            raise TuningError(f"not {args.split} question ids: {', '.join(sorted(unknown))}")
        questions = [q for q in questions if q.id in wanted]
    if args.limit:
        questions = questions[: args.limit]

    available = load_strategies(settings.paths.resolve().strategies_file)
    names = args.strategies or list(available)
    unknown = [n for n in names if n not in available]
    if unknown:
        raise TuningError(f"unknown strategies: {', '.join(unknown)}")
    if args.reference not in available:
        raise TuningError(f"unknown reference strategy {args.reference!r}")
    if args.reference not in names:
        names = [args.reference, *names]  # the reference is always measured
    grid = parse_grid(args.grid)
    variants, skipped = expand_grid([available[n] for n in names], grid)
    commit, dirty = git_info()
    run_id = f"{'confirm' if args.final else 'tuning'}-{run_timestamp()}"
    return TuningManifest(
        run_id=run_id,
        created_at=utc_now(),
        split=args.split,
        final=args.final,
        question_ids=[q.id for q in questions],
        strategies=names,
        grid=grid,
        reference=args.reference,
        rewriter_model=rewriter_model(settings),
        seed=args.seed,
        n_resamples=args.resamples,
        git_commit=commit,
        git_dirty=dirty,
        gold_file=_rel(args.gold),
        gold_sha256=file_sha256(args.gold),
        configs=manifest_configs(variants),
        skipped=skipped,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        settings = get_settings()
        if args.resume:
            if args.strategies or args.grid or args.questions or args.limit:
                raise TuningError(
                    "--resume reuses the run's manifest; do not combine it with "
                    "--strategies/--grid/--questions/--limit"
                )
            run_dir = args.resume
            manifest = load_manifest(run_dir)
            gold_path = PROJECT_ROOT / manifest.gold_file
            if file_sha256(gold_path) != manifest.gold_sha256:
                raise TuningError("the gold file changed since this run began")
            if rewriter_model(settings) != manifest.rewriter_model:
                raise TuningError("models.retrieval_tuning changed since this run began")
            if manifest.split == "test":
                append_test_log(TEST_LOG, manifest, run_dir, "resume")
        else:
            manifest = new_manifest(args, settings)
            run_dir = RESULTS_DIR / manifest.run_id
            save_manifest(run_dir, manifest)
            if manifest.split == "test":
                append_test_log(TEST_LOG, manifest, run_dir, "start")
            gold_path = args.gold
        by_id = {q.id: q for q in load_gold(gold_path)}
        questions = [by_id[i] for i in manifest.question_ids]
        variants = variants_from_manifest(manifest)

        deps = RetrievalDeps.from_settings(settings)
        llm = CachedLLM(build_rewriter(settings, manifest.rewriter_model), run_dir / LLM_CACHE_FILE)
        deps.llm = llm
        if deps.reranker is not None:
            deps.reranker = CachedReranker(deps.reranker)
        count = tiktoken_counter()

        def retrieve_fn(question: str, cfg: StrategyConfig) -> RetrievalResult:
            return retrieve(question, cfg, deps=deps)

        n_in = sum(1 for q in questions if q.in_scope)
        print(
            f"Run {manifest.run_id}: {n_in} in-scope {manifest.split} questions x {len(variants)} "
            f"configurations, rewriter {manifest.rewriter_model} -> {_rel(run_dir)}",
            file=sys.stderr,
        )
        if manifest.skipped:
            print(f"Skipped {len(manifest.skipped)} invalid combinations", file=sys.stderr)
        store = RecordStore(run_dir / RESULTS_FILE)
        outcome = run_tuning(
            questions,
            variants,
            retrieve_fn,
            store,
            llm,
            count,
            progress=lambda msg: print(msg, file=sys.stderr, flush=True),
        )
    except KeyboardInterrupt:
        print("\nInterrupted. Continue with --resume.", file=sys.stderr)
        return 130
    except (
        TuningError,
        GoldFileError,
        StrategyConfigError,
        ConfigError,
        LLMConfigError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    rows = summarize_tuning(
        store.load(),
        questions,
        variants,
        manifest.reference,
        n_resamples=manifest.n_resamples,
        seed=manifest.seed,
    )
    write_summary_csv(run_dir / "summary.csv", rows)
    stats = (
        f"Rewriter calls this session: {llm.misses} new, {llm.hits} from cache; "
        f"{llm.spent.input_tokens:,} input / {llm.spent.output_tokens:,} output tokens"
    )
    (run_dir / "report.md").write_text(
        render_report(manifest, rows, failed=outcome.failed, llm_stats=stats), encoding="utf-8"
    )
    print(f"{stats}\nReport: {_rel(run_dir / 'report.md')}", file=sys.stderr)
    if outcome.failed:
        print(f"{len(outcome.failed)} retrievals failed; rerun with --resume.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
