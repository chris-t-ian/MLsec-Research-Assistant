"""Retrieval-only tuning: strategies x parameter grid on the gold questions, no generation.

Each (config, question) runs the retrieval pipeline (:func:`retrieve`) on the gold question
and stores the selected chunks; paper-level metrics are computed against ``gold_arxiv_ids``.
No answer model and no judge are involved. The only LLM calls are query translation and
self-query, and they go through :class:`CachedLLM`: the prompts are deterministic, so every
grid variant of a strategy reuses the same rewrites. Differences between variants of one
strategy are then caused by the parameters alone, and a rewrite is paid for once.

Out-of-scope questions have no gold papers and are skipped.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from research_assistant.abtest.gold import GoldQuestion, Split
from research_assistant.abtest.metrics import recall_at_k, reciprocal_rank
from research_assistant.abtest.stats import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    Estimate,
    bootstrap_mean,
    paired_bootstrap,
)
from research_assistant.llm import LLMCallError, StructuredLLM
from research_assistant.models import RetrievalResult, TokenUsage
from research_assistant.retrieval.rerank import Reranker
from research_assistant.retrieval.strategies import StrategyConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

TUNABLE = ("fetch_k", "top_k", "rerank_k", "max_chunks_per_paper")
RESULTS_FILE = "results.jsonl"
LLM_CACHE_FILE = "llm_cache.jsonl"
MANIFEST_FILE = "manifest.json"
MAX_CONSECUTIVE_FAILURES = 3

POWER_NOTE = (
    "Dev questions only, and few of them: paired bootstrap CIs are wide, so only large "
    "differences are detectable, and a CI that includes 0 means *no detectable difference*. "
    "Picking the best of many grid variants on the same questions overfits to them; confirm "
    "the chosen configuration once on the test split with eval/run_eval.py."
)
CONFIRM_NOTE = (
    "Test split, fixed configurations chosen on dev; no parameter was tuned on these "
    "questions, so the differences are an unbiased check of the dev choice. A CI that "
    "includes 0 means *no detectable difference*, not *no difference*."
)


class TuningError(RuntimeError):
    """Invalid grid, or the run cannot continue (e.g. the rewriter LLM keeps failing)."""


# --------------------------------------------------------------------------- grid


def parse_grid(specs: Sequence[str]) -> dict[str, list[int]]:
    """``["top_k=5,8,12", "fetch_k=30,60"]`` -> ``{"top_k": [5, 8, 12], "fetch_k": [30, 60]}``."""
    grid: dict[str, list[int]] = {}
    for spec in specs:
        name, sep, values = spec.partition("=")
        name = name.strip()
        if not sep or name not in TUNABLE:
            raise TuningError(
                f"grid entry {spec!r}: use <param>=<v1>,<v2>,... with param in {', '.join(TUNABLE)}"
            )
        try:
            ints = [int(v) for v in values.split(",") if v.strip()]
        except ValueError as exc:
            raise TuningError(f"grid entry {spec!r}: values must be integers") from exc
        if not ints or any(v <= 0 for v in ints):
            raise TuningError(f"grid entry {spec!r}: give one or more positive integers")
        if name in grid:
            raise TuningError(f"grid parameter {name} given twice")
        grid[name] = list(dict.fromkeys(ints))
    return grid


def check_split_selection(split: str, *, final: bool, grid: bool, subset: bool) -> None:
    """Tuning (a grid) only on dev. The test split is only for a ``final`` confirmation of
    fixed strategies on all test questions, so nothing can be tuned on it."""
    if split not in ("dev", "test"):
        raise TuningError(f"unknown split {split!r}; use dev or test")
    if split == "dev":
        if final:
            raise TuningError("--final is only for the test split")
        return
    if not final:
        raise TuningError(
            "the test split is for the final confirmation only; add --final (the run is "
            "logged to eval/results/test_runs.log)"
        )
    if grid:
        raise TuningError(
            "no --grid on the test split: tuning on test questions would leak them into the "
            "choice. Pass the chosen strategies from strategies.yaml with --strategies."
        )
    if subset:
        raise TuningError("a test confirmation uses the whole test split (no --questions/--limit)")


def variant_name(base: str, params: Mapping[str, int]) -> str:
    return base + "".join(f"__{k}{v}" for k, v in sorted(params.items()))


@dataclass(frozen=True)
class Variant:
    """One configuration under test: a strategy from strategies.yaml with grid overrides."""

    name: str
    base: str
    params: dict[str, int]  # only the overridden values that differ from the base
    config: StrategyConfig


def expand_grid(
    bases: Sequence[StrategyConfig], grid: Mapping[str, Sequence[int]]
) -> tuple[list[Variant], list[str]]:
    """Every base strategy x every grid combination. ``rerank_k`` only varies for strategies
    that rerank. Combinations that violate the strategy constraints (e.g. top_k > fetch_k)
    are skipped and returned as messages. The unmodified base is always included."""
    variants: dict[str, Variant] = {}
    skipped: list[str] = []
    for base in bases:
        keys = [k for k in grid if k != "rerank_k" or base.rerank]
        combos = itertools.product(*(grid[k] for k in keys)) if keys else [()]
        candidates: list[dict[str, int]] = [{}]
        candidates += [dict(zip(keys, c, strict=True)) for c in combos]
        for combo in candidates:
            data = base.model_dump()
            changed = {k: v for k, v in combo.items() if data[k] != v}
            name = variant_name(base.name, changed)
            if name in variants:
                continue
            data.update(changed)
            data["name"] = name
            try:
                cfg = StrategyConfig.model_validate(data)
            except ValidationError as exc:
                msg = exc.errors()[0]["msg"].removeprefix("Value error, ")
                skipped.append(f"{name}: {msg}")
                continue
            variants[name] = Variant(name=name, base=base.name, params=changed, config=cfg)
    return list(variants.values()), skipped


# --------------------------------------------------------------------------- LLM cache


class _CacheEntry(BaseModel):
    key: str
    schema_name: str
    data: dict[str, Any]
    usage: TokenUsage


class CachedLLM:
    """:class:`StructuredLLM` that answers repeated (schema, system, user) prompts from a
    JSONL cache. Only successful calls are cached; failures are counted and re-raised."""

    def __init__(self, inner: StructuredLLM | None, path: Path) -> None:
        self._inner = inner
        self.path = path
        self._cache: dict[str, _CacheEntry] = {}
        self.hits = 0
        self.misses = 0
        self.failures = 0
        self.spent = TokenUsage()  # tokens of calls made in this process
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    entry = _CacheEntry.model_validate_json(line)
                except ValidationError:
                    continue
                self._cache[entry.key] = entry

    @staticmethod
    def key(schema: type[BaseModel], system: str, user: str) -> str:
        raw = "\0".join([schema.__name__, system, user]).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def invoke_structured(self, schema: type[T], system: str, user: str) -> tuple[T, TokenUsage]:
        key = self.key(schema, system, user)
        entry = self._cache.get(key)
        if entry is not None:
            self.hits += 1
            return schema.model_validate(entry.data), TokenUsage()
        if self._inner is None:
            self.failures += 1
            raise LLMCallError("no rewriter LLM configured")
        self.misses += 1
        try:
            out, usage = self._inner.invoke_structured(schema, system, user)
        except LLMCallError:
            self.failures += 1
            raise
        entry = _CacheEntry(
            key=key, schema_name=schema.__name__, data=out.model_dump(), usage=usage
        )
        self._cache[key] = entry
        self.spent = self.spent + usage
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")
        return out, usage


class CachedReranker:
    """Reranker that remembers (query, text) scores in memory. Cross-encoder scores are
    deterministic, so grid variants of a reranking strategy (which rerank largely the same
    candidates) only pay for pairs not scored before. Wall-clock latency of later variants
    is therefore not comparable and is left out of the ranking."""

    def __init__(self, inner: Reranker) -> None:
        self._inner = inner
        self._scores: dict[tuple[str, str], float] = {}

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        missing = list(dict.fromkeys(t for t in texts if (query, t) not in self._scores))
        if missing:
            for text, value in zip(missing, self._inner.score(query, missing), strict=True):
                self._scores[(query, text)] = value
        return [self._scores[(query, t)] for t in texts]


# --------------------------------------------------------------------------- records


class RetrievalRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question_id: str
    config: str
    base_strategy: str
    params: dict[str, int] = Field(default_factory=dict)
    papers: list[str] = Field(description="arXiv ID of each selected chunk, in rank order.")
    chunk_ids: list[str]
    context_tokens: int = Field(ge=0, description="Tokens of the selected chunk texts.")
    latency_s: float = Field(ge=0)
    llm_usage: TokenUsage = TokenUsage()
    rewrites: list[str] = Field(default_factory=list)
    filters: dict[str, Any] | None = None
    notes: list[str] = Field(default_factory=list)

    @property
    def key(self) -> tuple[str, str]:
        return (self.config, self.question_id)


def to_record(
    question: GoldQuestion,
    variant: Variant,
    result: RetrievalResult,
    count_tokens: Callable[[str], int],
) -> RetrievalRecord:
    trace = result.trace
    return RetrievalRecord(
        question_id=question.id,
        config=variant.name,
        base_strategy=variant.base,
        params=variant.params,
        papers=[c.chunk.arxiv_id for c in result.chunks],
        chunk_ids=[c.chunk.id for c in result.chunks],
        context_tokens=sum(count_tokens(c.chunk.text) for c in result.chunks),
        latency_s=round(trace.total_latency_ms / 1000, 3),
        llm_usage=trace.token_usage,
        rewrites=trace.rewrites,
        filters=trace.filters.model_dump() if trace.filters else None,
        notes=[f"{s.stage}: {n}" for s in trace.stages for n in s.notes],
    )


class RecordStore:
    """Append-only JSONL; the last line per (config, question) wins."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[tuple[str, str], RetrievalRecord]:
        out: dict[tuple[str, str], RetrievalRecord] = {}
        if not self.path.exists():
            return out
        for n, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rec = RetrievalRecord.model_validate_json(line)
            except ValidationError:
                logger.warning("Skipping unreadable line %d in %s", n, self.path)
                continue
            out[rec.key] = rec
        return out

    def append(self, record: RetrievalRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")


class TuningManifest(BaseModel):
    run_id: str
    created_at: str
    split: Split
    final: bool = Field(
        default=False, description="Test-split confirmation of fixed configurations, no grid."
    )
    question_ids: list[str]
    strategies: list[str]
    grid: dict[str, list[int]]
    reference: str
    rewriter_model: str
    seed: int
    n_resamples: int
    git_commit: str
    git_dirty: bool
    gold_file: str
    gold_sha256: str
    configs: list[dict[str, Any]] = Field(description="Every variant's full StrategyConfig.")
    skipped: list[str] = Field(default_factory=list)


def save_manifest(run_dir: Path, manifest: TuningManifest) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / MANIFEST_FILE).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def load_manifest(run_dir: Path) -> TuningManifest:
    try:
        return TuningManifest.model_validate_json(
            (run_dir / MANIFEST_FILE).read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as exc:
        raise TuningError(f"{run_dir} has no readable {MANIFEST_FILE}") from exc


def variants_from_manifest(manifest: TuningManifest) -> list[Variant]:
    out = []
    for data in manifest.configs:
        cfg = StrategyConfig.model_validate(data["config"])
        out.append(Variant(name=cfg.name, base=data["base"], params=data["params"], config=cfg))
    return out


def manifest_configs(variants: Sequence[Variant]) -> list[dict[str, Any]]:
    return [{"base": v.base, "params": v.params, "config": v.config.model_dump()} for v in variants]


# --------------------------------------------------------------------------- run

RetrieveFn = Callable[[str, StrategyConfig], RetrievalResult]


@dataclass
class TuningOutcome:
    done: int = 0
    skipped: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)


def run_tuning(
    questions: Sequence[GoldQuestion],
    variants: Sequence[Variant],
    retrieve_fn: RetrieveFn,
    store: RecordStore,
    llm: CachedLLM,
    count_tokens: Callable[[str], int],
    *,
    progress: Callable[[str], None] | None = None,
    max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
) -> TuningOutcome:
    """Retrieve every (variant, question) not yet stored. A retrieval during which a rewriter
    call failed (the pipeline then silently falls back to the raw question) is not stored,
    so a rerun repeats it; after ``max_consecutive_failures`` such retrievals in a row the
    run stops with :class:`TuningError`."""
    say = progress or (lambda _msg: None)
    existing = store.load()
    in_scope = [q for q in questions if q.in_scope]
    total = len(in_scope) * len(variants)
    outcome = TuningOutcome()
    consecutive = 0
    n = 0
    for q in in_scope:
        for v in variants:
            n += 1
            if (v.name, q.id) in existing:
                outcome.skipped += 1
                continue
            failures_before = llm.failures
            t0 = time.perf_counter()
            result = retrieve_fn(q.question, v.config)
            if llm.failures > failures_before:
                outcome.failed.append((v.name, q.id))
                consecutive += 1
                say(f"[{n}/{total}] {v.name} q{q.id}: rewriter LLM failed, not stored")
                if consecutive >= max_consecutive_failures:
                    raise TuningError(
                        f"{consecutive} retrievals in a row had a failing rewriter LLM. "
                        "Stopped; finished retrievals are saved, continue with --resume."
                    )
                continue
            consecutive = 0
            record = to_record(q, v, result, count_tokens)
            store.append(record)
            outcome.done += 1
            say(
                f"[{n}/{total}] {v.name} q{q.id}: {len(record.papers)} chunks, "
                f"{time.perf_counter() - t0:.1f}s"
            )
    return outcome


# --------------------------------------------------------------------------- metrics


@dataclass(frozen=True)
class TuningMetric:
    name: str
    label: str
    higher_is_better: bool
    unit: str = ""


METRICS: tuple[TuningMetric, ...] = (
    TuningMetric("gold_coverage", "Gold papers found", True),
    TuningMetric("recall_any", "Recall (any gold, all returned)", True),
    TuningMetric("recall@3", "Recall@3", True),
    TuningMetric("mrr", "MRR", True),
    TuningMetric("context_tokens", "Context tokens", False, "tok"),
)
PRIMARY = "gold_coverage"


def gold_coverage(papers: Sequence[str], gold: Sequence[str]) -> float | None:
    """Share of the question's gold papers that appear among the selected chunks."""
    if not gold:
        return None
    return len(set(gold) & set(papers)) / len(set(gold))


def tuning_metrics(record: RetrievalRecord, question: GoldQuestion) -> dict[str, float | None]:
    gold = question.gold_arxiv_ids
    papers = record.papers
    return {
        "gold_coverage": gold_coverage(papers, gold),
        "recall_any": recall_at_k(papers, gold, len(papers)) if papers else 0.0,
        "recall@3": recall_at_k(papers, gold, 3),
        "mrr": reciprocal_rank(papers, gold),
        "context_tokens": float(record.context_tokens),
    }


class TuningRow(BaseModel):
    config: str
    base_strategy: str
    params: dict[str, int]
    n_questions: int
    estimates: dict[str, Estimate]
    vs_reference: dict[str, Estimate]
    pareto: bool = False  # not dominated on (gold_coverage up, context_tokens down)


def summarize_tuning(
    records: Mapping[tuple[str, str], RetrievalRecord],
    questions: Sequence[GoldQuestion],
    variants: Sequence[Variant],
    reference: str,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> list[TuningRow]:
    """Per variant: mean of each metric with a bootstrap CI, and the paired difference to
    ``reference`` over the questions both have. Sorted by gold coverage, then MRR, then
    fewer context tokens."""
    qs = [q for q in questions if q.in_scope]
    values: dict[str, dict[str, dict[str, float]]] = {}
    for v in variants:
        per_q: dict[str, dict[str, float]] = {}
        for q in qs:
            rec = records.get((v.name, q.id))
            if rec is None:
                continue
            for m, val in tuning_metrics(rec, q).items():
                if val is not None:
                    per_q.setdefault(m, {})[q.id] = val
        values[v.name] = per_q

    ref = values.get(reference, {})
    rows: list[TuningRow] = []
    for v in variants:
        per_q = values[v.name]
        est: dict[str, Estimate] = {}
        diff: dict[str, Estimate] = {}
        for m in METRICS:
            vals = per_q.get(m.name, {})
            est[m.name] = bootstrap_mean(list(vals.values()), n_resamples=n_resamples, seed=seed)
            ref_vals = ref.get(m.name, {})
            shared = [qid for qid in vals if qid in ref_vals]
            if v.name != reference and shared:
                diff[m.name] = paired_bootstrap(
                    [vals[i] for i in shared],
                    [ref_vals[i] for i in shared],
                    n_resamples=n_resamples,
                    seed=seed,
                )
        n_q = len({qid for vals in per_q.values() for qid in vals})
        rows.append(
            TuningRow(
                config=v.name,
                base_strategy=v.base,
                params=v.params,
                n_questions=n_q,
                estimates=est,
                vs_reference=diff,
            )
        )
    _mark_pareto(rows)

    def _sort_key(r: TuningRow) -> tuple[float, float, float]:
        def mean(m: str, default: float) -> float:
            value = r.estimates[m].mean
            return default if value is None else value

        return (-mean(PRIMARY, -1.0), -mean("mrr", -1.0), mean("context_tokens", float("inf")))

    return sorted(rows, key=_sort_key)


def _mark_pareto(rows: list[TuningRow]) -> None:
    points = {
        r.config: (r.estimates[PRIMARY].mean, r.estimates["context_tokens"].mean) for r in rows
    }
    for r in rows:
        cov, tok = points[r.config]
        if cov is None or tok is None:
            continue
        dominated = any(
            c is not None and t is not None and c >= cov and t <= tok and (c > cov or t < tok)
            for name, (c, t) in points.items()
            if name != r.config
        )
        r.pareto = not dominated


# --------------------------------------------------------------------------- output


def _fmt(value: float | None, unit: str = "") -> str:
    if value is None:
        return "-"
    if unit in ("tok",):
        return f"{value:,.0f}"
    if unit == "s":
        return f"{value:.1f}"
    return f"{value:.2f}"


def _fmt_diff(est: Estimate | None, unit: str = "") -> str:
    if est is None or est.mean is None:
        return "-"
    if est.ci_low is None or est.ci_high is None:
        return f"{_fmt(est.mean, unit)} (n={est.n})"
    sign = "+" if est.mean >= 0 else ""
    return f"{sign}{_fmt(est.mean, unit)} [{_fmt(est.ci_low, unit)}, {_fmt(est.ci_high, unit)}]"


def write_summary_csv(path: Path, rows: Sequence[TuningRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "config",
                "base_strategy",
                "params",
                "metric",
                "n",
                "mean",
                "ci_low",
                "ci_high",
                "diff_vs_ref",
                "diff_ci_low",
                "diff_ci_high",
                "pareto",
            ]
        )
        for r in rows:
            for m in METRICS:
                e = r.estimates[m.name]
                d = r.vs_reference.get(m.name)
                w.writerow(
                    [
                        r.config,
                        r.base_strategy,
                        json.dumps(r.params, sort_keys=True),
                        m.name,
                        e.n,
                        e.mean,
                        e.ci_low,
                        e.ci_high,
                        d.mean if d else None,
                        d.ci_low if d else None,
                        d.ci_high if d else None,
                        r.pareto,
                    ]
                )


def render_report(
    manifest: TuningManifest,
    rows: Sequence[TuningRow],
    *,
    failed: Sequence[tuple[str, str]] = (),
    llm_stats: str = "",
) -> str:
    n_in_scope = max((r.n_questions for r in rows), default=0)
    lines = [
        f"# {'Retrieval confirmation' if manifest.final else 'Retrieval tuning'} {manifest.run_id}",
        "",
        f"- Split: **{manifest.split}**, {n_in_scope} in-scope questions with gold papers "
        f"(out-of-scope questions are skipped)",
        f"- Strategies: {', '.join(manifest.strategies)}; grid: "
        + (", ".join(f"{k}={v}" for k, v in manifest.grid.items()) or "none"),
        f"- Configurations: {len(rows)}; reference: `{manifest.reference}`",
        f"- Rewriter model (query translation, self-query): `{manifest.rewriter_model}`. "
        "Production uses `models.judge`; strategies with query translation or filters may "
        "behave somewhat differently there.",
        f"- Commit `{manifest.git_commit[:10]}`{' (dirty tree)' if manifest.git_dirty else ''}, "
        f"gold sha256 `{manifest.gold_sha256[:12]}`, bootstrap {manifest.n_resamples} "
        f"resamples, seed {manifest.seed}",
    ]
    if llm_stats:
        lines.append(f"- {llm_stats}")
    if failed:
        lines.append(f"- **Missing:** {len(failed)} retrievals failed; rerun with --resume.")
    if manifest.skipped:
        lines.append(
            f"- Skipped invalid combinations: {len(manifest.skipped)} (listed in manifest.json)"
        )
    lines += [
        "",
        f"> {CONFIRM_NOTE if manifest.final else POWER_NOTE}",
        "",
        "## Metrics",
        "",
        "Paper level: a chunk counts as a hit if it comes from one of the question's gold papers.",
        "",
        "- **Gold papers found**: share of the question's gold papers among the selected chunks "
        "(matters for comparison questions with several gold papers). Primary sort key.",
        "- **Recall (any)**: 1 if any gold paper is among the selected chunks.",
        "- **Recall@3**: same, within the top 3 chunks.",
        "- **MRR**: 1 / rank of the first gold chunk (0 if none).",
        "- **Context tokens**: tokens of the selected chunk texts, i.e. what one search adds to "
        "the answer model's prompt (the cost side of top_k).",
        "- Latency is not compared: rewrites and reranker scores are cached across grid "
        "variants, so only the first variant of a strategy pays for them.",
        "- **Pareto**: no other configuration finds at least as many gold papers with fewer "
        "context tokens.",
        "",
        "## Ranking",
        "",
        "| # | Configuration | Pareto | Gold found | Recall (any) | Recall@3 | MRR | Context tok |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, start=1):
        e = r.estimates
        lines.append(
            f"| {i} | `{r.config}` | {'✓' if r.pareto else ''} | "
            + " | ".join(_fmt(e[m.name].mean, m.unit) for m in METRICS)
            + " |"
        )
    lines += [
        "",
        f"## Difference to `{manifest.reference}` (paired bootstrap, 95% CI)",
        "",
        "| Configuration | n | Δ Gold found | Δ MRR | Δ Context tok |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        if r.config == manifest.reference:
            continue
        d = r.vs_reference
        n = d[PRIMARY].n if PRIMARY in d else 0
        lines.append(
            f"| `{r.config}` | {n} | {_fmt_diff(d.get(PRIMARY))} | {_fmt_diff(d.get('mrr'))} | "
            f"{_fmt_diff(d.get('context_tokens'), 'tok')} |"
        )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "LLM_CACHE_FILE",
    "MANIFEST_FILE",
    "METRICS",
    "RESULTS_FILE",
    "TUNABLE",
    "CachedLLM",
    "CachedReranker",
    "RecordStore",
    "RetrievalRecord",
    "TuningError",
    "TuningManifest",
    "TuningOutcome",
    "TuningRow",
    "Variant",
    "check_split_selection",
    "expand_grid",
    "gold_coverage",
    "load_manifest",
    "manifest_configs",
    "parse_grid",
    "render_report",
    "run_tuning",
    "save_manifest",
    "summarize_tuning",
    "to_record",
    "tuning_metrics",
    "variant_name",
    "variants_from_manifest",
    "write_summary_csv",
]
