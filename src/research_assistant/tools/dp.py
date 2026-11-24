"""dp_accountant tool: privacy accounting for DP-SGD with Google's ``dp-accounting`` library.

DP-SGD is modeled as ``steps`` self-compositions of a Poisson-subsampled Gaussian mechanism
(``PoissonSampledDpEvent(sample_rate, GaussianDpEvent(noise_multiplier))``) under the
add-or-remove-one neighboring relation, the setting of Abadi et al. (2016) and Opacus/TF Privacy.

* ``rdp``: Rényi DP accountant (``dp_accounting.rdp.RdpAccountant``, default orders); fast, a
  valid but slightly loose upper bound.
* ``pld``: privacy loss distribution accountant (``dp_accounting.pld.PLDAccountant``); tighter,
  but takes ~1-5 s per evaluation on a laptop CPU.

Noise mode binary-searches (bisection on a log scale) the smallest noise multiplier whose
epsilon is at most ``target_epsilon``; epsilon is monotonically decreasing in the noise.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import Field, model_validator

from research_assistant.models import ToolResult
from research_assistant.tools.base import ToolArgs, ToolDeps, fail, run_safely

if TYPE_CHECKING:  # dp_accounting pulls in scipy (~2 s); imported lazily below
    import dp_accounting

Accountant = Literal["rdp", "pld"]
MAX_STEPS = 10_000_000
SIGMA_MIN = 0.1  # below this, accounting is slow and DP-SGD is not meaningful
SIGMA_MAX = 1000.0
REL_TOL = 1e-3  # noise mode: relative precision of the returned noise multiplier


class DPAccountantArgs(ToolArgs):
    mode: Literal["epsilon", "noise"] = Field(
        default="epsilon",
        description=(
            "'epsilon': compute epsilon for a given noise_multiplier. "
            "'noise': find the smallest noise_multiplier reaching target_epsilon."
        ),
    )
    noise_multiplier: float | None = Field(
        default=None,
        gt=0,
        description="Gaussian noise std / clipping norm (sigma). Required for mode='epsilon'.",
    )
    sample_rate: float = Field(
        gt=0, le=1, description="Poisson sampling rate q = batch_size / dataset_size, in (0, 1]."
    )
    steps: int = Field(ge=1, le=MAX_STEPS, description="Number of DP-SGD steps (iterations).")
    delta: float = Field(gt=0, lt=1, description="Target delta, in (0, 1); typically < 1/n.")
    target_epsilon: float | None = Field(
        default=None, gt=0, description="Target epsilon. Required for mode='noise'."
    )
    accountant: Accountant = Field(
        default="rdp", description="'rdp' (fast, looser) or 'pld' (tighter, slower)."
    )
    dataset_size: int | None = Field(
        default=None, ge=1, description="Optional number of training examples n (to check delta)."
    )

    @model_validator(mode="after")
    def _check_mode(self) -> Self:
        if self.mode == "epsilon" and self.noise_multiplier is None:
            raise ValueError("mode 'epsilon' requires noise_multiplier")
        if self.mode == "noise" and self.target_epsilon is None:
            raise ValueError("mode 'noise' requires target_epsilon")
        return self


def _make_accountant(kind: Accountant) -> dp_accounting.PrivacyAccountant:
    from dp_accounting import pld, rdp

    return rdp.RdpAccountant() if kind == "rdp" else pld.PLDAccountant()


def dp_sgd_event(noise_multiplier: float, sample_rate: float, steps: int) -> dp_accounting.DpEvent:
    import dp_accounting

    return dp_accounting.SelfComposedDpEvent(
        dp_accounting.PoissonSampledDpEvent(
            sample_rate, dp_accounting.GaussianDpEvent(noise_multiplier)
        ),
        steps,
    )


def compute_epsilon(
    noise_multiplier: float, sample_rate: float, steps: int, delta: float, accountant: Accountant
) -> float:
    """Epsilon of DP-SGD at ``delta`` (may be ``inf`` for tiny noise)."""
    acc = _make_accountant(accountant)
    acc.compose(dp_sgd_event(noise_multiplier, sample_rate, steps))
    return float(acc.get_epsilon(delta))


class NoiseSearchError(ValueError):
    """No noise multiplier in [SIGMA_MIN, SIGMA_MAX] satisfies the target (user-safe)."""


def find_noise_multiplier(
    target_epsilon: float,
    eps_of: Callable[[float], float],
    start: float = 1.0,
    factor: float = 2.0,
    rel_tol: float = REL_TOL,
) -> tuple[float, float, bool]:
    """Smallest sigma (within ``rel_tol``) with ``eps_of(sigma) <= target_epsilon``.

    Brackets geometrically from ``start`` by ``factor``, then bisects on a log scale; the
    returned sigma always satisfies the target. Returns ``(sigma, epsilon, hit_floor)``, where
    ``hit_floor`` means even ``SIGMA_MIN`` meets the target.
    """
    cache: dict[float, float] = {}

    def eps(sigma: float) -> float:
        if sigma not in cache:
            cache[sigma] = eps_of(sigma)
        return cache[sigma]

    def ok(sigma: float) -> bool:
        return eps(sigma) <= target_epsilon

    sigma = min(max(start, SIGMA_MIN), SIGMA_MAX)
    if ok(sigma):
        hi = sigma
        lo = max(hi / factor, SIGMA_MIN)
        while ok(lo):
            if lo <= SIGMA_MIN:
                return lo, eps(lo), True
            hi, lo = lo, max(lo / factor, SIGMA_MIN)
    else:
        lo = sigma
        hi = min(lo * factor, SIGMA_MAX)
        while not ok(hi):
            if hi >= SIGMA_MAX:
                raise NoiseSearchError(
                    f"epsilon <= {target_epsilon:g} is not reachable with a noise multiplier "
                    f"up to {SIGMA_MAX:g}; use fewer steps or a lower sample rate."
                )
            lo, hi = hi, min(hi * factor, SIGMA_MAX)

    while hi / lo > 1 + rel_tol:
        mid = math.sqrt(lo * hi)
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi, eps(hi), False


def dp_accountant(
    mode: str = "epsilon",
    sample_rate: float | None = None,
    steps: int | None = None,
    delta: float | None = None,
    noise_multiplier: float | None = None,
    target_epsilon: float | None = None,
    accountant: str = "rdp",
    dataset_size: int | None = None,
    deps: ToolDeps | None = None,  # unused; uniform tool signature
) -> ToolResult:
    """DP-SGD epsilon for a noise multiplier, or the noise multiplier for a target epsilon.

    ``data``: ``{"mode", "accountant", "epsilon", "delta", "noise_multiplier", "sample_rate",
    "steps", "target_epsilon", "sampling", "neighboring_relation", "warnings": [...]}``.
    """

    def run() -> ToolResult | dict[str, Any]:
        args = DPAccountantArgs.model_validate(
            {
                "mode": mode,
                "noise_multiplier": noise_multiplier,
                "sample_rate": sample_rate,
                "steps": steps,
                "delta": delta,
                "target_epsilon": target_epsilon,
                "accountant": accountant,
                "dataset_size": dataset_size,
            }
        )
        warnings: list[str] = []
        if args.dataset_size is not None and args.delta > 1 / args.dataset_size:
            warnings.append(
                f"delta = {args.delta:g} is larger than 1/dataset_size = "
                f"{1 / args.dataset_size:.3g}; such a delta permits leaking whole records. "
                "Choose delta well below 1/n."
            )

        def eps_of(sigma: float, kind: Accountant = args.accountant) -> float:
            return compute_epsilon(sigma, args.sample_rate, args.steps, args.delta, kind)

        if args.mode == "epsilon":
            assert args.noise_multiplier is not None
            sigma = args.noise_multiplier
            epsilon = eps_of(sigma)
        else:
            assert args.target_epsilon is not None
            target = args.target_epsilon
            try:
                if args.accountant == "pld":
                    # PLD epsilon <= RDP epsilon, so the RDP solution is an upper bound for
                    # sigma; starting there keeps the number of slow PLD evaluations small.
                    start, _, _ = find_noise_multiplier(target, lambda s: eps_of(s, "rdp"))
                    sigma, epsilon, floor = find_noise_multiplier(
                        target, eps_of, start=start, factor=1.25
                    )
                else:
                    sigma, epsilon, floor = find_noise_multiplier(target, eps_of)
            except NoiseSearchError as exc:
                return fail("dp_accounting", str(exc))
            if floor:
                warnings.append(
                    f"Even noise_multiplier = {SIGMA_MIN:g} meets the target; the search "
                    "does not go lower."
                )

        if not math.isfinite(epsilon):
            return fail(
                "dp_accounting",
                "epsilon is unbounded for these parameters; increase the noise multiplier.",
            )
        return {
            "mode": args.mode,
            "accountant": args.accountant,
            "epsilon": epsilon,
            "delta": args.delta,
            "noise_multiplier": sigma,
            "sample_rate": args.sample_rate,
            "steps": args.steps,
            "target_epsilon": args.target_epsilon,
            "sampling": "poisson",
            "neighboring_relation": "add_or_remove_one",
            "warnings": warnings,
        }

    return run_safely("dp_accounting", run)
