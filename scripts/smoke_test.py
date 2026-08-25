"""Day-0 smoke test: proves the skeleton runs end to end before anyone starts.

Runs the three Gate-G1 checks (PLAN.md). All three currently PASS. Each is averaged over
N_PROFILES days on the same basis as scripts/calibrate_degradation.py, so a gate never
turns on one lucky or unlucky profile.

Note the gates pass against the SYNTHETIC placeholder profiles in env.synthetic_day. They
must be re-checked once real data lands -- see DECISIONS.md section 5.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from pemwe import load_config, PEMWEEnv, BASELINES
from pemwe import profiles as _P

N_PROFILES = 8   # gates average over this many days; matches calibrate_degradation.py

# Real Kutch weather once data/processed/ exists, synthetic placeholder before that.
# Same basis as scripts/calibrate_degradation.py so the gates and the calibration can
# never disagree about what they were measured on.
REAL = _P.env_profiles(N_PROFILES, split="train", source="hybrid")
PROFILE_SOURCE = "real Kutch (train split)" if REAL is not None else "SYNTHETIC placeholder"


def rollout(env, policy, seed=0, options=None):
    obs, _ = env.reset(seed=seed, options=options)
    policy.reset()
    tot = {"h2_kg": 0.0, "dv_deg_uv": 0.0, "reward": 0.0, "cycles": 0, "curt_kwh": 0.0}
    while True:
        obs, r, term, trunc, info = env.step(policy.act(obs))
        tot["h2_kg"] += info["h2_kg"]
        tot["dv_deg_uv"] += info["dv_deg_uv"]
        tot["reward"] += r
        tot["cycles"] += int(info["cycled"])
        tot["curt_kwh"] += info["curtailed_w"] * env.dt_min / 60.0 / 1000.0
        if term or trunc:
            return tot


class Jittery:
    """Hand-scripted adversary for Gate G1 item 2."""
    def __init__(self, rng):
        self.rng = rng
    def reset(self):
        pass
    def act(self, obs, info=None):
        return np.array([self.rng.uniform(-1, 1)], dtype=np.float32)


class Smooth:
    def reset(self):
        pass
    def act(self, obs, info=None):
        return np.array([2 * min(float(obs[0]), 0.55) - 1], dtype=np.float32)


def main():
    cfg = load_config()
    env = PEMWEEnv(cfg, seed=0)

    print(f"P_rated = {env.p_rated/1e3:.1f} kW   ({cfg['stack']['n_cells']} cells)")
    print(f"profiles: {PROFILE_SOURCE}")

    # --- G1 item 1: efficiency must be non-monotonic ---
    j, v, eta, pf = env.stack.efficiency_curve()
    k = int(np.argmax(eta))
    print("\n[G1.1] efficiency curve")
    print(f"  V_cell at rated j={j[-1]:.2f}      : {v[-1]:.3f} V   (expect ~1.75-1.85)")
    print(f"  peak eta_LHV                   : {eta[k]*100:.1f}% at j={j[k]:.2f} A/cm^2")
    print(f"  eta at rated                   : {eta[-1]*100:.1f}%")
    print(f"  eta at min                     : {eta[0]*100:.1f}%")
    ok1 = 0.3 < j[k] < 1.2 and eta[k] > eta[-1] and eta[k] > eta[0]
    print(f"  -> non-monotonic, peak mid-range: {'PASS' if ok1 else 'FAIL  <-- A fixes this'}")

    # --- G1 item 2: jittery must degrade faster than smooth ---
    # Averaged over N_PROFILES days -- a gate must not turn on one lucky or unlucky
    # profile. Same basis as scripts/calibrate_degradation.py, so the two always agree.
    def mean_over_days(make_policy):
        acc = {"h2_kg": 0.0, "dv_deg_uv": 0.0, "cycles": 0.0, "curt_kwh": 0.0}
        for s in range(N_PROFILES):
            env_s = PEMWEEnv(cfg, profiles=REAL, seed=s)
            opts = {"day_idx": s % N_PROFILES} if REAL is not None else None
            t = rollout(env_s, make_policy(s), seed=s, options=opts)
            for key in acc:
                acc[key] += t[key] / N_PROFILES
        return acc

    sm = mean_over_days(lambda s: Smooth())
    ji = mean_over_days(lambda s: Jittery(np.random.default_rng(s)))
    ratio = ji["dv_deg_uv"] / max(sm["dv_deg_uv"], 1e-9)
    print(f"\n[G1.2] jittery vs smooth policy, mean of {N_PROFILES} days")
    print(f"  smooth : {sm['h2_kg']:7.1f} kg H2, {sm['dv_deg_uv']:7.2f} uV, {sm['cycles']:5.1f} cycles")
    print(f"  jittery: {ji['h2_kg']:7.1f} kg H2, {ji['dv_deg_uv']:7.2f} uV, {ji['cycles']:5.1f} cycles")
    print(f"  -> degradation ratio {ratio:.2f}x (target >= 3x): "
          f"{'PASS' if ratio >= 3 else 'FAIL  <-- scripts/calibrate_degradation.py --solve'}")

    # --- G1 item 3: baselines run, and the naive one is calibrated to literature ---
    print(f"\n[G1.3] baseline controllers, mean of {N_PROFILES} days")
    rates = {}
    for name, cls in BASELINES.items():
        t = mean_over_days(lambda s, c=cls: c(cfg))
        rate = t["dv_deg_uv"] / 24.0
        rates[name] = rate
        life = env.deg.projected_life_years(rate)
        print(f"  {name:24s} {t['h2_kg']:7.1f} kg  {rate:6.2f} uV/h  "
              f"-> {life:5.2f} yr  ({t['cycles']:.1f} cycles, {t['curt_kwh']:.0f} kWh curtailed)")
    ok3 = abs(rates["baseline_naive"] - 4.0) <= 0.5
    print(f"  -> baseline_naive calibrated to 4.0 +/- 0.5 uV/h (~5 yr, ref #7): "
          f"{'PASS' if ok3 else 'FAIL  <-- scripts/calibrate_degradation.py --solve'}")

    print("\nSkeleton runs end to end. Day 1 can start.")


if __name__ == "__main__":
    main()
