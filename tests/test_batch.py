"""BatchSim must reproduce PEMWEEnv exactly (to floating-point rounding), for any profile and any actions."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pemwe import PEMWEEnv, NaiveLoadFollowing, RampLimitedBaseline
from pemwe.batch import (BatchSim, BatchNaive, BatchRampLimited, BatchHysteresis)
from pemwe.env import synthetic_day
from pemwe.evaluate import rollout_episode

_spec = importlib.util.spec_from_file_location(
    "rule_family_v2", Path(__file__).resolve().parents[1] / "scripts" / "rule_family_v2.py")
rf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rf)

RTOL = 1e-9


@pytest.fixture(scope="module")
def profiles(cfg):
    """Ordinary and deliberately extreme days: smooth, cloudy, very noisy, square wave, saturated, sparse spikes."""
    p_rated = PEMWEEnv(cfg).p_rated
    n = cfg["env"]["steps_per_episode"]
    rng = np.random.default_rng(7)
    days = [synthetic_day(rng, n, p_rated, c) for c in (0.0, 0.1, 0.4, 1.0)]
    t = np.arange(n)
    days.append(np.where((t // 30) % 2 == 0, p_rated, 0.0))                     # on/off every 30 min: many cycles
    days.append(np.full(n, 0.95 * p_rated))                                     # saturated: degradation accumulates, j_rated reached
    days.append(np.where(rng.random(n) < 0.02, rng.random(n) * p_rated, 0.0))   # sparse spikes
    days.append(np.full(n, 0.02 * p_rated))                                     # always below the minimum power
    return np.stack(days)


def _scalar_episodes(cfg, profiles, make_policy):
    env = PEMWEEnv(cfg, profiles=profiles, seed=0)
    out = []
    for i in range(len(profiles)):
        ep, _ = rollout_episode(env, make_policy(i), i, 0)
        out.append(ep)
    return out


def _compare(ep_list, m):
    for key in ("h2_kg", "dv_deg_uv", "curtailed_kwh", "reward_total", "r_yield", "r_deg", "r_ramp",
                "mean_abs_ramp", "mean_eta_lhv"):
        ref = np.array([e[key] for e in ep_list])
        np.testing.assert_allclose(m[key], ref, rtol=RTOL, atol=1e-12, err_msg=key)
    assert (m["n_cycles"] == np.array([e["n_cycles"] for e in ep_list])).all()


def test_random_actions_match_step_by_step(cfg, profiles):
    rng = np.random.default_rng(3)
    B, T = len(profiles), cfg["env"]["steps_per_episode"]
    actions = rng.uniform(-1, 1, (T, B)).astype(np.float32)
    # make the actions sticky so the plant spends time on, off and at the cap
    actions[100:400] = 1.0
    actions[600:640] = -1.0
    env = PEMWEEnv(cfg, profiles=profiles, seed=0)
    sim = BatchSim(cfg, profiles)
    obs_b = sim.reset()
    for i in range(B):
        obs, _ = env.reset(seed=0, options={"day_idx": i})
        np.testing.assert_allclose(obs_b[i], obs, rtol=1e-6, atol=1e-7)
    ref = {k: np.empty((T, B)) for k in ("h2_kg", "dv_deg_uv", "j")}
    for i in range(B):
        env.reset(seed=0, options={"day_idx": i})
        for t in range(T):
            obs, r, _, _, info = env.step(actions[t, i:i + 1])
            ref["h2_kg"][t, i], ref["dv_deg_uv"][t, i], ref["j"][t, i] = info["h2_kg"], info["dv_deg_uv"], info["j"]
            if t in (0, 1, 15, 16, 700, T - 2):                       # observations, incl. the moving average
                pass
    sim.reset()
    got = {k: np.empty((T, B)) for k in ref}
    for t in range(T):
        s = sim.step(actions[t])
        got["h2_kg"][t], got["dv_deg_uv"][t], got["j"][t] = s["h2_kg"], s["dv_step_uv"], s["j"]
    for k in ref:
        np.testing.assert_allclose(got[k], ref[k], rtol=RTOL, atol=1e-13, err_msg=k)


def test_observations_match(cfg, profiles):
    rng = np.random.default_rng(5)
    T = 60
    actions = rng.uniform(-1, 1, (T, len(profiles))).astype(np.float32)
    env = PEMWEEnv(cfg, profiles=profiles, seed=0)
    sim = BatchSim(cfg, profiles)
    sim.reset()
    sim_obs = []
    for t in range(T):
        sim.step(actions[t])
        sim_obs.append(sim.obs())
    for i in range(len(profiles)):
        env.reset(seed=0, options={"day_idx": i})
        for t in range(T):
            o, *_ = env.step(actions[t, i:i + 1])
            np.testing.assert_allclose(sim_obs[t][i], o, rtol=1e-6, atol=1e-7, err_msg=f"episode {i} step {t}")


def test_naive_rule_matches(cfg, profiles):
    ref = _scalar_episodes(cfg, profiles, lambda i: NaiveLoadFollowing(cfg))
    m = BatchSim(cfg, profiles).rollout(BatchNaive(cfg["baseline"]["p_idle_threshold_frac"], cfg["baseline"]["p_idle_frac"]))
    _compare(ref, m)


def test_ramp_limited_rule_matches(cfg, profiles):
    ref = _scalar_episodes(cfg, profiles, lambda i: RampLimitedBaseline(cfg))
    b = cfg["baseline"]
    m = BatchSim(cfg, profiles).rollout(BatchRampLimited(b["p_idle_threshold_frac"], b["p_idle_frac"], b["ramp_limit_frac_per_min"]))
    _compare(ref, m)


@pytest.mark.parametrize("params", [(0.08, 5, 5, 0.003, 0.7), (0.15, 20, 20, 0.001, 0.5), (0.08, 20, 5, 0.01, 1.0)])
def test_hysteresis_rule_matches(cfg, profiles, params):
    ref = _scalar_episodes(cfg, profiles, lambda i: rf.HysteresisRule(*params))
    m = BatchSim(cfg, profiles).rollout(BatchHysteresis(*params))
    _compare(ref, m)


def test_a_grid_of_rules_in_one_batch_matches_one_at_a_time(cfg, profiles):
    """Per-episode rule parameters: (rule, day) pairs flattened into one batch equal separate runs."""
    grid = [(0.08, 5, 5, 0.003, 0.7), (0.15, 20, 5, 0.001, 0.5), (0.08, 20, 20, 0.01, 1.0)]
    D = len(profiles)
    big = np.concatenate([profiles] * len(grid))
    cols = [np.repeat([g[k] for g in grid], D) for k in range(5)]
    m = BatchSim(cfg, big).rollout(BatchHysteresis(*cols))
    for gi, g in enumerate(grid):
        single = BatchSim(cfg, profiles).rollout(BatchHysteresis(*g))
        for key in ("h2_kg", "dv_deg_uv"):
            np.testing.assert_allclose(m[key][gi * D:(gi + 1) * D], single[key], rtol=RTOL, atol=1e-12)
