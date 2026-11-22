"""Summaries (means, paired differences to the baseline) and the run report: summary.csv,
report.md and two matplotlib figures."""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from pathlib import Path

from matplotlib.figure import Figure
from pydantic import BaseModel

from research_assistant.abtest.gold import GoldQuestion
from research_assistant.abtest.metrics import (
    ClaimVerifier,
    MetricSpec,
    compute_metrics,
    metric_specs,
)
from research_assistant.abtest.records import EvalRecord
from research_assistant.abtest.runner import BASELINE, RunManifest
from research_assistant.abtest.stats import Estimate, bootstrap_mean, paired_bootstrap

SUMMARY_CSV = "summary.csv"
REPORT_MD = "report.md"
METRICS_PNG = "metrics_ci.png"
SCATTER_PNG = "recall_vs_latency.png"

POWER_NOTE = (
    "With about 40 test questions (and 20 dev questions), paired bootstrap CIs are wide: only "
    "large differences between strategies are detectable, and a CI that includes 0 means "
    "*no detectable difference*, not *no difference*. Differences on dev questions guide "
    "development only; the test split is for the final comparison."
)

# Reference categorical palette (dataviz skill, light surface), fixed order by strategy.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


class MetricSummary(BaseModel):
    strategy: str
    metric: str
    estimate: Estimate
    diff_vs_baseline: Estimate | None = None  # None for the baseline itself / no baseline


class Summary(BaseModel):
    strategies: list[str]
    baseline: str | None
    k: int
    seed: int
    n_resamples: int
    n_questions: int
    rows: list[MetricSummary]

    def get(self, strategy: str, metric: str) -> MetricSummary | None:
        return next((r for r in self.rows if r.strategy == strategy and r.metric == metric), None)


def per_question_values(
    records: Sequence[EvalRecord],
    questions: Sequence[GoldQuestion],
    k: int,
    claim_verifier: ClaimVerifier | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """strategy -> metric -> question id -> value (undefined values left out)."""
    by_id = {q.id: q for q in questions}
    out: dict[str, dict[str, dict[str, float]]] = {}
    for rec in records:
        q = by_id.get(rec.question_id)
        if q is None:
            continue
        for metric, value in compute_metrics(rec, q, k, claim_verifier).items():
            if value is not None:
                out.setdefault(rec.strategy, {}).setdefault(metric, {})[q.id] = value
    return out


def summarize(
    records: Sequence[EvalRecord],
    questions: Sequence[GoldQuestion],
    strategies: Sequence[str],
    *,
    k: int,
    seed: int,
    n_resamples: int,
    baseline: str = BASELINE,
    claim_verifier: ClaimVerifier | None = None,
) -> Summary:
    values = per_question_values(records, questions, k, claim_verifier)
    base = baseline if baseline in strategies else None
    rows: list[MetricSummary] = []
    for strategy in strategies:
        for spec in metric_specs(k):
            vals = values.get(strategy, {}).get(spec.name, {})
            est = bootstrap_mean(list(vals.values()), n_resamples=n_resamples, seed=seed)
            diff = None
            if base is not None and strategy != base:
                base_vals = values.get(base, {}).get(spec.name, {})
                paired = sorted(set(vals) & set(base_vals))
                diff = paired_bootstrap(
                    [vals[q] for q in paired],
                    [base_vals[q] for q in paired],
                    n_resamples=n_resamples,
                    seed=seed,
                )
            rows.append(
                MetricSummary(
                    strategy=strategy, metric=spec.name, estimate=est, diff_vs_baseline=diff
                )
            )
    n_questions = len({r.question_id for r in records})
    return Summary(
        strategies=list(strategies),
        baseline=base,
        k=k,
        seed=seed,
        n_resamples=n_resamples,
        n_questions=n_questions,
        rows=rows,
    )


# --------------------------------------------------------------------------- formatting


def _fmt(x: float | None, spec: MetricSpec, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    if spec.unit == "tokens":
        s = f"{x:+,.0f}" if signed else f"{x:,.0f}"
    elif spec.unit == "s":
        s = f"{x:+.1f}" if signed else f"{x:.1f}"
    else:
        s = f"{x:+.3f}" if signed else f"{x:.3f}"
    return s


def _fmt_est(e: Estimate | None, spec: MetricSpec, signed: bool = False) -> str:
    if e is None or e.mean is None:
        return "-"
    mean = _fmt(e.mean, spec, signed)
    if e.ci_low is None or e.ci_high is None:
        return mean
    return f"{mean} [{_fmt(e.ci_low, spec, signed)}, {_fmt(e.ci_high, spec, signed)}]"


def write_summary_csv(summary: Summary, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "strategy",
                "metric",
                "n",
                "mean",
                "ci_low",
                "ci_high",
                "baseline",
                "n_paired",
                "diff_vs_baseline",
                "diff_ci_low",
                "diff_ci_high",
            ]
        )
        for r in summary.rows:
            d = r.diff_vs_baseline
            w.writerow(
                [
                    r.strategy,
                    r.metric,
                    r.estimate.n,
                    r.estimate.mean,
                    r.estimate.ci_low,
                    r.estimate.ci_high,
                    summary.baseline if d is not None else "",
                    d.n if d else "",
                    d.mean if d else "",
                    d.ci_low if d else "",
                    d.ci_high if d else "",
                ]
            )


def render_report(
    summary: Summary,
    manifest: RunManifest,
    questions: Sequence[GoldQuestion],
    *,
    failed: Sequence[str] = (),
    figures: Sequence[str] = (),
) -> str:
    specs = metric_specs(summary.k)
    selected = [q for q in questions if q.id in set(manifest.question_ids)]
    n_in = sum(1 for q in selected if q.in_scope)
    n_oos = len(selected) - n_in
    commit = manifest.git_commit[:12] + (" (dirty)" if manifest.git_dirty else "")
    lines = [
        f"# Evaluation report - {manifest.run_id}",
        "",
        f"- **Split:** {manifest.split}{' (final test run)' if manifest.final else ''}",
        f"- **Questions:** {len(selected)} ({n_in} in scope, {n_oos} out of scope); "
        f"questions with results: {summary.n_questions}",
        f"- **Strategies:** {', '.join(manifest.strategies)}",
        f"- **Baseline:** {summary.baseline or 'not in this run (no paired differences)'}",
        f"- **Models:** answer `{manifest.main_model}`, judge `{manifest.judge_model}`, "
        f"temperature {manifest.temperature}",
        f"- **Git commit:** `{commit}`",
        f"- **Gold file:** `{manifest.gold_file}` (sha256 `{manifest.gold_sha256[:12]}`)",
        f"- **Bootstrap:** {summary.n_resamples:,} resamples over questions, seed "
        f"{summary.seed}, 95% percentile CIs",
        "",
        f"> {POWER_NOTE}",
        "",
    ]
    if failed:
        lines += [
            f"**Incomplete:** {len(failed)} turn(s) failed and are missing from the results "
            f"({', '.join(failed)}). Rerun with `--resume` to fill them in.",
            "",
        ]
    lines += [
        "## Metrics",
        "",
        "| Metric | Group | Definition |",
        "|---|---|---|",
    ]
    lines += [f"| {s.label} | {s.group} | {s.description} |" for s in specs]
    lines += [
        "",
        "Retrieval and answer metrics use in-scope questions only, correct refusal uses "
        "out-of-scope questions only, cost uses all. Retrieved chunks are those of all "
        "`search_knowledge_base` calls of the turn, in call order. `n` counts questions with a "
        "defined value (e.g. gold-citation rate needs at least one citation); differences use "
        "only questions defined for both strategies (`n` paired). The supported-claim rate is "
        "a hook for step 14 and is empty until a claim verifier is plugged in.",
        "",
    ]
    for strategy in summary.strategies:
        lines += [f"## {strategy}", ""]
        if summary.baseline and strategy != summary.baseline:
            lines += [
                f"| Metric | n | Mean [95% CI] | n paired | Δ vs {summary.baseline} [95% CI] |",
                "|---|---:|---|---:|---|",
            ]
        else:
            lines += ["| Metric | n | Mean [95% CI] |", "|---|---:|---|"]
        for spec in specs:
            row = summary.get(strategy, spec.name)
            if row is None:
                continue
            unit = f" ({spec.unit})" if spec.unit else ""
            cells = [f"{spec.label}{unit}", str(row.estimate.n), _fmt_est(row.estimate, spec)]
            if summary.baseline and strategy != summary.baseline:
                d = row.diff_vs_baseline
                cells += [str(d.n if d else 0), _fmt_est(d, spec, signed=True)]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    if figures:
        lines += ["## Figures", ""]
        lines += [f"![{name}]({name})" for name in figures]
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- plots


def _style(ax) -> None:  # type: ignore[no-untyped-def]
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _color(i: int) -> str:
    return SERIES_COLORS[i % len(SERIES_COLORS)]


def plot_metric_means(summary: Summary, path: Path) -> None:
    """One panel per metric: strategy means with 95% CI whiskers (horizontal)."""
    specs = [
        s
        for s in metric_specs(summary.k)
        if any(
            (r := summary.get(st, s.name)) and r.estimate.mean is not None
            for st in summary.strategies
        )
    ]
    if not specs:
        return
    ncols = 2
    nrows = math.ceil(len(specs) / ncols)
    n = len(summary.strategies)
    fig = Figure(figsize=(10, 0.45 * n * nrows + 1.2 * nrows + 0.6), facecolor=SURFACE)
    axes = fig.subplots(nrows, ncols, squeeze=False)
    for ax, spec in zip(axes.flat, specs, strict=False):
        _style(ax)
        for i, strategy in enumerate(summary.strategies):
            row = summary.get(strategy, spec.name)
            if row is None or row.estimate.mean is None:
                continue
            e = row.estimate
            y = n - 1 - i
            if e.ci_low is not None and e.ci_high is not None:
                ax.errorbar(
                    e.mean,
                    y,
                    xerr=[[e.mean - e.ci_low], [e.ci_high - e.mean]],
                    fmt="none",
                    ecolor=_color(i),
                    elinewidth=2,
                    capsize=4,
                )
            ax.plot(
                e.mean,
                y,
                "o",
                color=_color(i),
                markersize=8,
                markeredgecolor=SURFACE,
                markeredgewidth=2,
            )
        ax.set_yticks(range(n))
        ax.set_yticklabels(
            [
                f"{s} (n={r.estimate.n})" if (r := summary.get(s, spec.name)) else s
                for s in reversed(summary.strategies)
            ]
        )
        unit = f" ({spec.unit})" if spec.unit else ""
        ax.set_title(spec.label + unit, color=INK, fontsize=10, loc="left")
        if not spec.unit:
            ax.set_xlim(-0.02, 1.02)
    for ax in list(axes.flat)[len(specs) :]:
        ax.set_visible(False)
    fig.suptitle("Metric means with 95% bootstrap CIs", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def plot_recall_vs_latency(summary: Summary, path: Path) -> None:
    """Mean recall@k vs mean latency per strategy, CI whiskers on both axes."""
    recall = f"recall@{summary.k}"
    fig = Figure(figsize=(7, 4.5), facecolor=SURFACE)
    ax = fig.subplots()
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    plotted = False
    for i, strategy in enumerate(summary.strategies):
        r, lat = summary.get(strategy, recall), summary.get(strategy, "latency_s")
        if not r or not lat or r.estimate.mean is None or lat.estimate.mean is None:
            continue
        x, y = lat.estimate, r.estimate
        xerr = (
            [[x.mean - x.ci_low], [x.ci_high - x.mean]]
            if x.ci_low is not None and x.ci_high is not None
            else None
        )
        yerr = (
            [[y.mean - y.ci_low], [y.ci_high - y.mean]]
            if y.ci_low is not None and y.ci_high is not None
            else None
        )
        ax.errorbar(
            x.mean,
            y.mean,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            color=_color(i),
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            elinewidth=1.5,
            capsize=3,
            label=strategy,
        )
        plotted = True
    if not plotted:
        return
    ax.set_xlabel("Mean latency per question (s)", color=INK_2, fontsize=9)
    ax.set_ylabel(f"Recall@{summary.k} (paper level)", color=INK_2, fontsize=9)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"Recall@{summary.k} vs latency (95% CIs)", color=INK, fontsize=11, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def write_report(
    run_dir: Path,
    summary: Summary,
    manifest: RunManifest,
    questions: Sequence[GoldQuestion],
    *,
    failed: Sequence[str] = (),
) -> Path:
    write_summary_csv(summary, run_dir / SUMMARY_CSV)
    figures = []
    plot_metric_means(summary, run_dir / METRICS_PNG)
    plot_recall_vs_latency(summary, run_dir / SCATTER_PNG)
    for name in (METRICS_PNG, SCATTER_PNG):
        if (run_dir / name).exists():
            figures.append(name)
    report = run_dir / REPORT_MD
    report.write_text(
        render_report(summary, manifest, questions, failed=failed, figures=figures),
        encoding="utf-8",
    )
    return report


__all__ = [
    "POWER_NOTE",
    "Summary",
    "per_question_values",
    "plot_metric_means",
    "plot_recall_vs_latency",
    "render_report",
    "summarize",
    "write_report",
    "write_summary_csv",
]
