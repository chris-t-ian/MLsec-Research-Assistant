"""Bootstrap confidence intervals over questions (fixed seed, reproducible)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 0


class Estimate(BaseModel):
    """A mean (or mean difference) with a percentile bootstrap CI. ``n`` is always reported;
    the CI is None for n < 2."""

    n: int
    mean: float | None
    ci_low: float | None = None
    ci_high: float | None = None


def _resampled_means(values: np.ndarray, n_resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    return values[idx].mean(axis=1)


def _estimate(values: np.ndarray, n_resamples: int, seed: int, level: float) -> Estimate:
    n = len(values)
    if n == 0:
        return Estimate(n=0, mean=None)
    mean = float(values.mean())
    if n < 2:
        return Estimate(n=n, mean=mean)
    means = _resampled_means(values, n_resamples, seed)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return Estimate(n=n, mean=mean, ci_low=float(lo), ci_high=float(hi))


def bootstrap_mean(
    values: Sequence[float],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = 0.95,
) -> Estimate:
    """Mean of ``values`` with a percentile bootstrap CI (questions resampled)."""
    return _estimate(np.asarray(values, dtype=float), n_resamples, seed, level)


def paired_bootstrap(
    treatment: Sequence[float],
    baseline: Sequence[float],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = 0.95,
) -> Estimate:
    """Mean of the paired differences ``treatment[i] - baseline[i]`` with a paired bootstrap
    CI: questions are resampled with replacement and both arms keep their pairing."""
    t = np.asarray(treatment, dtype=float)
    b = np.asarray(baseline, dtype=float)
    if t.shape != b.shape:
        raise ValueError("paired samples must have the same length")
    return _estimate(t - b, n_resamples, seed, level)


__all__ = ["DEFAULT_RESAMPLES", "DEFAULT_SEED", "Estimate", "bootstrap_mean", "paired_bootstrap"]
