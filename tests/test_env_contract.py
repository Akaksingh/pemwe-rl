"""CONTRACTS.md section 1, enforced. OWNER: Person A.

Person B's training loop and Person C's analysis are already written against these. If a
change of mine breaks one of them, it must fail here and not in someone else's overnight
run. This file is the reason A can keep editing the physics after the env is "frozen".
"""

import numpy as np
import pytest

from pemwe import PEMWEEnv

# spelled exactly as CONTRACTS.md section 1 lists them, and in that order
INFO_KEYS = [
    "p_renew_w", "p_set_w", "p_stack_w", "j", "v_cell", "h2_kg", "eta_lhv",
    "dv_deg_uv", "dv_deg_total_uv", "is_on", "cycled", "curtailed_w",
    "r_yield", "r_deg", "r_ramp",
]
OBS_FIELDS = ["p_renew", "p_renew_ma15", "p_set_prev", "j_frac", "deg_norm",
              "h2_rate_norm", "tod_sin", "tod_cos"]


def test_spaces_are_frozen(env):
    assert env.observation_space.shape == (8,)
    assert env.observation_space.dtype == np.float32
    assert env.action_space.shape == (1,)
    assert env.action_space.low[0] == -1.0 and env.action_space.high[0] == 1.0


def test_gymnasium_api(env):
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    out = env.step(env.action_space.sample())
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)


def test_every_info_key_present_at_every_step(env):
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    for t in range(env.n_steps):
        obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, size=1).astype(np.float32))
        missing = [k for k in INFO_KEYS if k not in info]
        assert not missing, f"step {t} missing {missing}"
        if term or trunc:
            break


def test_observation_stays_inside_its_declared_box(env):
    """SB3 does not check this; a silent out-of-box observation corrupts training."""
    obs, _ = env.reset(seed=2)
    rng = np.random.default_rng(2)
    for t in range(env.n_steps):
        obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, size=1).astype(np.float32))
        assert env.observation_space.contains(obs), f"step {t}: {obs}"
        if term or trunc:
            break


def test_p_set_prev_and_j_frac_are_distinct_channels(env):
    """They were byte-identical in the Day-0 stub: one of eight dimensions was dead,
    and the previous setpoint the contract promises was never actually supplied."""
    obs, _ = env.reset(seed=1)
    seen_different = False
    for t in range(env.n_steps):
        obs, r, term, trunc, info = env.step(np.array([0.2], dtype=np.float32))
        if obs[2] > 0 and abs(float(obs[2]) - float(obs[3])) > 1e-6:
            seen_different = True
            break
        if term or trunc:
            break
    assert seen_different, "obs[2] p_set_prev is still a copy of obs[3] j_frac"


def test_reset_observation_reports_the_real_renewable_power(cfg):
    """The first action of an episode must not be taken against a false observation."""
    env = PEMWEEnv(cfg, seed=5)
    obs, _ = env.reset(seed=5)
    assert float(obs[0]) == pytest.approx(env.profile[0] / env.p_rated, rel=1e-6)


def test_obs_channels_track_the_quantities_they_are_named_after(env):
    """The observation returned BY step t is the one the agent acts on at t+1, so its
    `p_set_prev` is the setpoint commanded at t -- i.e. info["p_set_w"] of this step."""
    obs, _ = env.reset(seed=4)
    for t in range(600):
        obs, r, term, trunc, info = env.step(np.array([0.35], dtype=np.float32))
        assert float(obs[2]) == pytest.approx(info["p_set_w"] / env.p_rated,
                                              rel=1e-5, abs=1e-6)
        assert float(obs[3]) == pytest.approx(info["j"] / env.stack.j_rated,
                                              rel=1e-5, abs=1e-6)
        assert float(obs[4]) == pytest.approx(info["dv_deg_total_uv"] / env.deg.dv_eol,
                                              rel=1e-4, abs=1e-9)
        if term or trunc:
            break


def test_reward_is_the_weighted_sum_of_the_logged_components(env):
    """C reads r_yield/r_deg/r_ramp UNWEIGHTED. They must reconstruct the total exactly,
    or the component logging cannot diagnose reward hacking."""
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(1)
    for _ in range(400):
        obs, reward, term, trunc, info = env.step(
            rng.uniform(-1, 1, size=1).astype(np.float32))
        expect = (env.w1 * info["r_yield"] - env.w2 * info["r_deg"]
                  - env.w3 * info["r_ramp"])
        assert reward == pytest.approx(expect, rel=1e-9, abs=1e-12)
        if term or trunc:
            break


def test_renewable_only_hard_cap_is_never_violated(env):
    """DECISIONS.md 3. If this ever fails, every yield number in the paper is inflated."""
    obs, _ = env.reset(seed=6)
    rng = np.random.default_rng(6)
    for t in range(env.n_steps):
        obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, size=1).astype(np.float32))
        assert info["p_set_w"] <= info["p_renew_w"] + 1e-6, f"step {t}"
        assert info["p_stack_w"] <= info["p_renew_w"] + 1e-6, f"step {t}"
        assert info["curtailed_w"] >= -1e-9
        if term or trunc:
            break


def test_episode_is_exactly_one_day(env):
    """DECISIONS.md 1-2: 1440 steps of 1 min. Truncation, never termination."""
    obs, _ = env.reset(seed=0)
    n = 0
    while True:
        obs, r, term, trunc, info = env.step(np.array([0.0], dtype=np.float32))
        n += 1
        assert not term, "the episode must truncate on time, not terminate"
        if trunc:
            break
    assert n == 1440
    assert env.dt_min == 1.0


def test_persist_degradation_flag(cfg):
    """False -> each episode starts fresh; True -> the 90-day rollout accumulates."""
    from pemwe.config import deepcopy_cfg
    fresh = PEMWEEnv(cfg, seed=0)
    fresh.reset(seed=0)
    for _ in range(50):
        fresh.step(np.array([0.5], dtype=np.float32))
    carried = fresh.dv_deg_uv
    fresh.reset(seed=0)
    assert fresh.dv_deg_uv == 0.0

    c2 = deepcopy_cfg(cfg)
    c2["env"]["persist_degradation"] = True
    keep = PEMWEEnv(c2, seed=0)
    keep.reset(seed=0)
    for _ in range(50):
        keep.step(np.array([0.5], dtype=np.float32))
    before = keep.dv_deg_uv
    keep.reset(seed=0)
    assert keep.dv_deg_uv == pytest.approx(before)
    assert before > 0
