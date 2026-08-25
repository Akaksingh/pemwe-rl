"""Baselines, training and evaluation. OWNER: Person B.

These exist because `evaluate.py` is frozen after Day 2 and Person C's entire plotting
library reads its output. A schema drift here breaks C silently, on Day 5, in a figure --
which is the worst possible time and place to find it.

The SB3 tests are skipped when torch is not installed, so A's and C's suites still run on
a machine without it.
"""

import json

import numpy as np
import pytest

from pemwe import PEMWEEnv, BASELINES, profiles
from pemwe.config import deepcopy_cfg
from pemwe import evaluate as ev

sb3 = pytest.importorskip  # used below, per-test


def _has_data():
    try:
        profiles.load_profiles()
        return bool(profiles.SPLITS.get("test"))
    except Exception:
        return False


needs_data = pytest.mark.skipif(not _has_data(), reason="processed profiles not built")


# --- baselines ---------------------------------------------------------------------

def test_both_baselines_expose_the_sb3_policy_interface(cfg):
    """One rollout code path for scripted and learned policies depends on this."""
    for name, cls in BASELINES.items():
        p = cls(cfg)
        assert hasattr(p, "reset") and hasattr(p, "act")
        a = p.act(np.zeros(8, dtype=np.float32))
        assert a.shape == (1,)
        assert -1.0 <= float(a[0]) <= 1.0


def test_naive_baseline_implements_the_published_rule(cfg):
    """P_set = P_renew above the threshold, P_idle below it (ref [8]), verbatim."""
    p = BASELINES["baseline_naive"](cfg)
    thr = cfg["baseline"]["p_idle_threshold_frac"]
    for frac in (0.0, thr / 2, thr + 0.01, 0.5, 1.0):
        obs = np.zeros(8, dtype=np.float32)
        obs[0] = frac
        got = 0.5 * (float(p.act(obs)[0]) + 1.0)
        want = frac if frac >= thr else cfg["baseline"]["p_idle_frac"]
        assert got == pytest.approx(want, abs=1e-6)


def test_ramp_limited_baseline_respects_its_slew_limit(cfg):
    p = BASELINES["baseline_ramplimited"](cfg)
    p.reset()
    lim = cfg["baseline"]["ramp_limit_frac_per_min"]
    obs = np.zeros(8, dtype=np.float32)
    obs[0] = 1.0
    prev = 0.0
    for _ in range(200):
        frac = 0.5 * (float(p.act(obs)[0]) + 1.0)
        # tolerance is float32 eps scaled to the action range: act() returns float32,
        # so the decoded fraction cannot be exact.
        assert abs(frac - prev) <= lim + 1e-6
        prev = frac


def test_ramp_limited_baseline_is_not_a_strawman(cfg):
    """It must buy real stack life without throwing away the hydrogen.

    A baseline that is merely worse than the naive rule on both axes would make any RL
    result meaningless. This is the test that stops the comparison being rigged.
    """
    prof = profiles.env_profiles(n_days=6, split="train")
    if prof is None:
        pytest.skip("processed profiles not built")

    def run(pol):
        env = PEMWEEnv(cfg, profiles=prof, seed=0)
        h2 = dv = hours = 0.0
        for d in range(len(prof)):
            obs, _ = env.reset(seed=d, options={"day_idx": d})
            pol.reset()
            while True:
                obs, r, te, tr, info = env.step(pol.act(obs))
                h2 += info["h2_kg"]
                dv += info["dv_deg_uv"]
                hours += env.dt_min / 60.0
                if te or tr:
                    break
        return h2, dv / hours

    h2_n, rate_n = run(BASELINES["baseline_naive"](cfg))
    h2_r, rate_r = run(BASELINES["baseline_ramplimited"](cfg))

    assert rate_r < rate_n, "the ramp-limited baseline must degrade less"
    assert h2_r > 0.90 * h2_n, (
        f"it gives up {(1 - h2_r / h2_n) * 100:.1f}% of yield -- that is a strawman, "
        f"not a strong baseline")


# --- evaluate.py: the frozen schema -------------------------------------------------

@needs_data
def test_evaluate_emits_the_frozen_schema(cfg):
    res = ev.evaluate(cfg, BASELINES["baseline_naive"](cfg), "baseline_naive",
                      "test_run", split="test", max_days=2)
    assert validate_clean(res)
    assert res["profile_set"] == "test"
    assert len(res["episodes"]) == 2


def validate_clean(res):
    problems = ev.validate(res)
    assert not problems, "\n".join(problems)
    return True


@needs_data
def test_evaluate_matches_person_c_reference_schema(cfg):
    """Key-for-key against scripts/fake_results.py, which C built the plots on."""
    ref = ev._reference()
    if ref is None:
        pytest.skip("run scripts/fake_results.py to create the reference")
    res = ev.evaluate(cfg, BASELINES["baseline_naive"](cfg), "baseline_naive",
                      "test_run", split="test", max_days=2)
    assert not ev.validate(res, ref)


@needs_data
def test_only_one_episode_carries_a_trajectory(cfg):
    """Full trajectories for every episode would blow up results/."""
    res = ev.evaluate(cfg, BASELINES["baseline_naive"](cfg), "baseline_naive",
                      "t", split="test", max_days=4)
    assert len(res["trajectory"]["t_min"]) == 1440
    lens = {k: len(v) for k, v in res["trajectory"].items()}
    assert len(set(lens.values())) == 1, lens


@needs_data
def test_evaluation_defaults_to_held_out_days(cfg):
    """If a number in the paper came from a training day, the paper is wrong."""
    train = set(profiles.SPLITS["train"])
    res = ev.evaluate(cfg, BASELINES["baseline_naive"](cfg), "baseline_naive",
                      "t", split="test", max_days=6)
    leaked = [e["date"] for e in res["episodes"] if e["date"] in train]
    assert not leaked, f"training days leaked into a test evaluation: {leaked}"


@needs_data
def test_train_and_test_splits_are_disjoint():
    assert not set(profiles.SPLITS["train"]) & set(profiles.SPLITS["test"])
    assert len(profiles.SPLITS["test"]) > 0


@needs_data
def test_aggregate_matches_the_episodes_it_summarises(cfg):
    res = ev.evaluate(cfg, BASELINES["baseline_naive"](cfg), "baseline_naive",
                      "t", split="test", max_days=5)
    h2 = [e["h2_kg"] for e in res["episodes"]]
    dv = [e["dv_deg_uv"] for e in res["episodes"]]
    a = res["aggregate"]
    assert a["h2_kg_mean"] == pytest.approx(float(np.mean(h2)))
    assert a["dv_deg_uv_mean"] == pytest.approx(float(np.mean(dv)))
    assert a["deg_rate_uv_per_h"] == pytest.approx(float(np.mean(dv)) / 24.0)


def test_validator_actually_rejects_a_broken_file():
    """A validator that never fails is not a validator."""
    good = {"run_id": "x", "policy": "p", "seed": 0,
            "weights": {"w1": 1.0, "w2": 1.0, "w3": 0.1}, "profile_set": "test",
            "episodes": [{k: 0 for k in ev.EPISODE_KEYS}],
            "aggregate": {k: 0.0 for k in ev.AGGREGATE_KEYS},
            "trajectory": {k: [] for k in ev.TRAJECTORY_KEYS}}
    assert not ev.validate(good)

    missing_key = json.loads(json.dumps(good))
    del missing_key["aggregate"]["deg_rate_uv_per_h"]
    assert ev.validate(missing_key)

    ragged = json.loads(json.dumps(good))
    ragged["trajectory"]["t_min"] = [1, 2, 3]
    assert ev.validate(ragged)


# --- train.py ------------------------------------------------------------------------

def test_train_uses_only_training_days(cfg):
    from pemwe.train import train_profiles
    if not _has_data():
        pytest.skip("processed profiles not built")
    prof = train_profiles(cfg, n_days=5)
    assert prof.shape == (5, cfg["env"]["steps_per_episode"])
    assert len(profiles.SPLITS["train"]) >= 5


def test_reward_component_callback_logs_all_three_unweighted():
    """The brief's one non-obvious requirement, enforced.

    Total reward cannot tell "learning" from "found a degenerate policy". If these three
    tags ever stop being written, the w2 sweep becomes unreadable.
    """
    pytest.importorskip("stable_baselines3")
    from pemwe.train import _make_callback_cls

    recorded = {}

    class FakeLogger:
        def record_mean(self, k, v):
            recorded[k] = v
        record = record_mean

    import types
    cb = _make_callback_cls()((1.0, 10.0, 0.1), log_every=2)
    # BaseCallback.logger is a read-only property delegating to self.model.logger
    cb.model = types.SimpleNamespace(logger=FakeLogger())
    cb.locals = {"infos": [{"r_yield": 4.0, "r_deg": 2.0, "r_ramp": 1.0,
                            "is_on": True, "cycled": False, "j": 1.0,
                            "h2_kg": 0.04, "eta_lhv": 0.7, "curtailed_w": 0.0}] * 3}
    cb._on_step()

    for k in ("components/r_yield", "components/r_deg", "components/r_ramp"):
        assert k in recorded, f"{k} not logged -- the sweep would be undiagnosable"
    # unweighted, exactly as CONTRACTS.md 1 requires
    assert recorded["components/r_deg"] == pytest.approx(2.0)
    # and the weighted view, which is what the gradient actually sees
    assert recorded["weighted/deg"] == pytest.approx(-20.0)
    assert "diag/parked_at_idle" in recorded
    assert "diag/pinned_at_rated" in recorded
