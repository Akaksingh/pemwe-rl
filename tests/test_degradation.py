"""Degradation-model tests. OWNER: Person A. Covers the brief's items 3 and 4 and the
Gate G1.2 / calibration targets, so a regression fails the suite rather than the paper.
"""

import numpy as np
import pytest

from pemwe import PEMWEEnv, BASELINES
from pemwe.degradation import TERMS


def rollout(cfg, policy, seed=1, n_days=1):
    """Totals over a rollout, using the same accounting as scripts/smoke_test.py."""
    tot = {"h2_kg": 0.0, "dv": 0.0, "cycles": 0, "hours": 0.0}
    for d in range(n_days):
        env = PEMWEEnv(cfg, seed=seed + d)
        obs, _ = env.reset(seed=seed + d)
        policy.reset()
        while True:
            obs, r, term, trunc, info = env.step(policy.act(obs))
            tot["h2_kg"] += info["h2_kg"]
            tot["dv"] += info["dv_deg_uv"]
            tot["cycles"] += int(info["cycled"])
            tot["hours"] += env.dt_min / 60.0
            if term or trunc:
                break
    return tot


class Const:
    """Constant power-fraction policy."""
    def __init__(self, frac):
        self.a = np.array([2.0 * frac - 1.0], dtype=np.float32)

    def reset(self):
        pass

    def act(self, obs, info=None):
        return self.a


class Jittery:
    def __init__(self, seed=0):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.rng = np.random.default_rng(self.seed)

    def act(self, obs, info=None):
        return np.array([self.rng.uniform(-1, 1)], dtype=np.float32)


class Smooth:
    def reset(self):
        pass

    def act(self, obs, info=None):
        return np.array([2 * min(float(obs[0]), 0.55) - 1], dtype=np.float32)


# --- the brief's item 3: degradation is monotonically non-decreasing ---------------

def test_degradation_never_decreases(env):
    obs, _ = env.reset(seed=3)
    prev = 0.0
    rng = np.random.default_rng(7)
    for _ in range(env.n_steps):
        obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, size=1).astype(np.float32))
        assert info["dv_deg_uv"] >= 0.0
        assert info["dv_deg_total_uv"] >= prev
        prev = info["dv_deg_total_uv"]
        if term or trunc:
            break
    assert prev > 0.0


def test_all_five_terms_are_present_and_non_negative(deg):
    total, parts = deg.step_uv(1.8, 0.2, 2.0, 1.0, True, False)
    assert set(parts) == set(TERMS)
    assert all(v >= 0 for v in parts.values())
    assert total == pytest.approx(sum(parts.values()))
    assert all(parts[t] > 0 for t in ("base", "stress", "ramp", "cycle")), \
        "an ON step that ramps hard, cycles and sits above the stress threshold"
    assert parts["idle"] == 0.0


def test_idle_is_not_free(deg):
    """DECISIONS.md 5: without this the degenerate 'park at idle forever' policy wins."""
    total, parts = deg.step_uv(0.0, 0.0, 2.0, 1.0, False, False)
    assert parts["idle"] > 0.0
    assert total > 0.0


def test_basis_is_exactly_linear_in_the_coefficients(deg):
    """The calibration is only valid if dv = coeffs . basis holds exactly."""
    args = (1.4, 0.3, 2.0, 1.0, True, False)
    total, _ = deg.step_uv(*args)
    assert total == pytest.approx(float(deg.coeffs @ deg.basis(*args)), rel=1e-12)


# --- the brief's item 4: full power degrades faster than mid power -----------------

def test_full_power_degrades_faster_than_mid_power(cfg):
    hi = rollout(cfg, Const(1.0), seed=1)
    mid = rollout(cfg, Const(0.35), seed=1)
    assert hi["dv"] > mid["dv"], (
        f"full power {hi['dv']:.1f} uV vs mid power {mid['dv']:.1f} uV")


def test_high_current_stress_term_only_bites_above_the_threshold(deg, stack):
    below = deg.step_uv(0.5 * stack.j_rated, 0.5 * stack.j_rated, stack.j_rated,
                        1.0, True, True)[1]["stress"]
    above = deg.step_uv(0.95 * stack.j_rated, 0.95 * stack.j_rated, stack.j_rated,
                        1.0, True, True)[1]["stress"]
    assert below == 0.0
    assert above > 0.0


# --- Gate G1.2 and the calibration targets ----------------------------------------

def test_gate_g1_2_jittery_degrades_at_least_3x_faster_than_smooth(cfg):
    sm = rollout(cfg, Smooth(), seed=1)
    ji = rollout(cfg, Jittery(0), seed=1)
    ratio = ji["dv"] / sm["dv"]
    assert ratio >= 3.0, f"ratio {ratio:.2f}x -- the model cannot teach the agent"


def test_baseline_is_calibrated_to_the_published_lifetime(cfg):
    """DECISIONS.md 5: the rule-based baseline [8] must reproduce the ~5 yr life of [7]."""
    env = PEMWEEnv(cfg, seed=1)
    t = rollout(cfg, BASELINES["baseline_naive"](cfg), seed=1)
    rate = t["dv"] / t["hours"]
    assert 3.5 <= rate <= 4.5, f"baseline at {rate:.2f} uV/h, target 4.0 +/- 0.5"
    life = env.deg.projected_life_years(rate)
    assert 4.0 <= life <= 6.0, f"projected life {life:.2f} yr, [7] reports ~5"


def test_no_policy_exceeds_the_worst_case_literature_rate(cfg):
    """[4] puts the worst-case intermittency-driven rate at ~50 uV/h. Nothing above it."""
    env = PEMWEEnv(cfg, seed=1)
    for name, pol in [("jittery", Jittery(0)), ("full", Const(1.0)),
                      ("naive", BASELINES["baseline_naive"](cfg))]:
        t = rollout(cfg, pol, seed=1)
        rate = t["dv"] / t["hours"]
        assert rate <= 50.0, f"{name} at {rate:.1f} uV/h exceeds the [4] worst case"


def test_smoother_baseline_degrades_less_than_the_naive_one(cfg):
    """The ramp-limited baseline is the honest strong one; it should buy some life."""
    naive = rollout(cfg, BASELINES["baseline_naive"](cfg), seed=1)
    ramp = rollout(cfg, BASELINES["baseline_ramplimited"](cfg), seed=1)
    assert ramp["dv"] < naive["dv"]
    assert ramp["cycles"] <= naive["cycles"]
