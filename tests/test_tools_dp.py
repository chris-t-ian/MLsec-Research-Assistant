"""dp_accountant: property tests only (no hardcoded epsilon values).

Properties: epsilon falls as the noise multiplier rises and grows with the number of steps;
PLD epsilon <= RDP epsilon; noise mode returns the (near-)smallest sufficient noise.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from research_assistant.tools import dp_accountant
from research_assistant.tools.dp import REL_TOL, compute_epsilon

# Realistic DP-SGD ranges; keeps PLD evaluations around a second each.
sigmas = st.floats(min_value=0.6, max_value=5.0)
rates = st.floats(min_value=1e-3, max_value=0.1)
step_counts = st.integers(min_value=10, max_value=3000)
deltas = st.sampled_from([1e-7, 1e-6, 1e-5, 1e-3])
REL = 1e-9


def eps(**kwargs: Any) -> float:
    result = dp_accountant(**kwargs)
    assert result.ok, result.error
    assert result.source == "dp_accounting"
    return float(result.data["epsilon"])


@settings(max_examples=30, deadline=None)
@given(s1=sigmas, s2=sigmas, q=rates, steps=step_counts, delta=deltas)
def test_rdp_epsilon_falls_as_noise_rises(
    s1: float, s2: float, q: float, steps: int, delta: float
) -> None:
    lo, hi = sorted((s1, s2))
    e_lo = eps(noise_multiplier=lo, sample_rate=q, steps=steps, delta=delta)
    e_hi = eps(noise_multiplier=hi, sample_rate=q, steps=steps, delta=delta)
    assert e_hi <= e_lo * (1 + REL)


@settings(max_examples=30, deadline=None)
@given(sigma=sigmas, q=rates, n1=step_counts, n2=step_counts, delta=deltas)
def test_rdp_epsilon_rises_with_steps(
    sigma: float, q: float, n1: int, n2: int, delta: float
) -> None:
    few, many = sorted((n1, n2))
    e_few = eps(noise_multiplier=sigma, sample_rate=q, steps=few, delta=delta)
    e_many = eps(noise_multiplier=sigma, sample_rate=q, steps=many, delta=delta)
    assert e_many >= e_few * (1 - REL)


@settings(max_examples=4, deadline=None)
@given(s1=sigmas, s2=sigmas, q=rates, steps=step_counts)
def test_pld_epsilon_falls_as_noise_rises(s1: float, s2: float, q: float, steps: int) -> None:
    lo, hi = sorted((s1, s2))
    kw = {"sample_rate": q, "steps": steps, "delta": 1e-5, "accountant": "pld"}
    assert eps(noise_multiplier=hi, **kw) <= eps(noise_multiplier=lo, **kw) * (1 + REL)


@settings(max_examples=8, deadline=None)
@given(sigma=sigmas, q=rates, steps=step_counts, delta=deltas)
def test_pld_epsilon_at_most_rdp(sigma: float, q: float, steps: int, delta: float) -> None:
    kw = {"noise_multiplier": sigma, "sample_rate": q, "steps": steps, "delta": delta}
    assert eps(accountant="pld", **kw) <= eps(accountant="rdp", **kw)


@settings(max_examples=5, deadline=None)
@given(
    target=st.floats(min_value=0.5, max_value=10.0),
    q=rates,
    steps=step_counts,
    delta=deltas,
)
def test_noise_mode_finds_smallest_sufficient_noise(
    target: float, q: float, steps: int, delta: float
) -> None:
    result = dp_accountant(
        mode="noise", target_epsilon=target, sample_rate=q, steps=steps, delta=delta
    )
    assert result.ok, result.error
    sigma = result.data["noise_multiplier"]
    assert result.data["epsilon"] <= target
    assert compute_epsilon(sigma, q, steps, delta, "rdp") <= target
    if not result.data["warnings"]:  # not clamped at the search floor
        assert compute_epsilon(sigma / (1 + 2 * REL_TOL), q, steps, delta, "rdp") > target


@settings(max_examples=5, deadline=None)
@given(target=st.floats(min_value=0.5, max_value=10.0), q=rates, n1=step_counts, n2=step_counts)
def test_noise_mode_needs_more_noise_for_more_steps(
    target: float, q: float, n1: int, n2: int
) -> None:
    few, many = sorted((n1, n2))

    def sigma(steps: int) -> float:
        r = dp_accountant(
            mode="noise", target_epsilon=target, sample_rate=q, steps=steps, delta=1e-5
        )
        assert r.ok, r.error
        return float(r.data["noise_multiplier"])

    assert sigma(many) >= sigma(few) / (1 + 2 * REL_TOL)


def test_noise_mode_pld_needs_no_more_noise_than_rdp() -> None:
    kw: dict[str, Any] = {
        "mode": "noise",
        "target_epsilon": 3.0,
        "sample_rate": 0.01,
        "steps": 500,
        "delta": 1e-5,
    }
    rdp_r = dp_accountant(accountant="rdp", **kw)
    pld_r = dp_accountant(accountant="pld", **kw)
    assert rdp_r.ok and pld_r.ok
    assert pld_r.data["epsilon"] <= 3.0
    assert pld_r.data["noise_multiplier"] <= rdp_r.data["noise_multiplier"]


def test_tool_output_matches_library_and_reports_setup() -> None:
    kw = {"noise_multiplier": 1.1, "sample_rate": 0.01, "steps": 1000, "delta": 1e-5}
    result = dp_accountant(**kw)
    assert result.data["epsilon"] == compute_epsilon(1.1, 0.01, 1000, 1e-5, "rdp")
    assert result.data["accountant"] == "rdp"
    assert result.data["sampling"] == "poisson"
    assert result.data["neighboring_relation"] == "add_or_remove_one"
    assert result.data["warnings"] == []


def test_warns_when_delta_exceeds_one_over_n() -> None:
    kw = {"noise_multiplier": 1.0, "sample_rate": 0.01, "steps": 100}
    warned = dp_accountant(delta=1e-3, dataset_size=60_000, **kw)
    assert warned.ok and len(warned.data["warnings"]) == 1
    assert "1/dataset_size" in warned.data["warnings"][0]
    fine = dp_accountant(delta=1e-6, dataset_size=60_000, **kw)
    assert fine.ok and fine.data["warnings"] == []


def test_unreachable_target_is_reported() -> None:
    result = dp_accountant(
        mode="noise", target_epsilon=1e-4, sample_rate=1.0, steps=10_000_000, delta=1e-10
    )
    assert not result.ok and result.source == "dp_accounting"
    assert "not reachable" in (result.error or "")


BASE: dict[str, Any] = {"noise_multiplier": 1.0, "sample_rate": 0.01, "steps": 100, "delta": 1e-5}


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"sample_rate": 0.0}, "sample_rate"),
        ({"sample_rate": 1.5}, "sample_rate"),
        ({"noise_multiplier": 0.0}, "noise_multiplier"),
        ({"noise_multiplier": -1.0}, "noise_multiplier"),
        ({"delta": 0.0}, "delta"),
        ({"delta": 1.0}, "delta"),
        ({"steps": 0}, "steps"),
        ({"steps": 10_000_001}, "steps"),
        ({"accountant": "gdp"}, "accountant"),
        ({"mode": "budget"}, "mode"),
        ({"dataset_size": 0}, "dataset_size"),
        ({"noise_multiplier": None}, "requires noise_multiplier"),
        ({"mode": "noise"}, "requires target_epsilon"),
        ({"mode": "noise", "target_epsilon": -1.0}, "target_epsilon"),
    ],
)
def test_invalid_ranges_are_rejected(override: dict[str, Any], field: str) -> None:
    result = dp_accountant(**{**BASE, **override})
    assert not result.ok and result.source == "validation"
    assert field in (result.error or "")


def test_sample_rate_one_is_allowed() -> None:
    assert dp_accountant(**{**BASE, "sample_rate": 1.0}).ok
