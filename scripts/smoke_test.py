"""Day-0 smoke test: proves the skeleton runs end to end before anyone starts.

Also prints a preview of the three Gate-G1 checks (PLAN.md), so Person A can see on Day 1
exactly what the stub physics gets right and wrong. It is expected to FAIL some checks
today -- making them pass is A's Day 1-2 job.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from pemwe import load_config, PEMWEEnv, BASELINES


def rollout(env, policy, seed=0):
    obs, _ = env.reset(seed=seed)
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
    rng = np.random.default_rng(0)
    sm = rollout(PEMWEEnv(cfg, seed=1), Smooth(), seed=1)
    ji = rollout(PEMWEEnv(cfg, seed=1), Jittery(rng), seed=1)
    ratio = ji["dv_deg_uv"] / max(sm["dv_deg_uv"], 1e-9)
    print("\n[G1.2] jittery vs smooth policy, same day")
    print(f"  smooth : {sm['h2_kg']:7.1f} kg H2, {sm['dv_deg_uv']:7.2f} uV, {sm['cycles']:4d} cycles")
    print(f"  jittery: {ji['h2_kg']:7.1f} kg H2, {ji['dv_deg_uv']:7.2f} uV, {ji['cycles']:4d} cycles")
    print(f"  -> degradation ratio {ratio:.2f}x (target >= 3x): "
          f"{'PASS' if ratio >= 3 else 'FAIL  <-- A tunes coefficients'}")

    # --- G1 item 3: baselines run and produce sane output ---
    print("\n[G1.3] baseline controllers")
    for name, cls in BASELINES.items():
        t = rollout(PEMWEEnv(cfg, seed=1), cls(cfg), seed=1)
        rate = t["dv_deg_uv"] / 24.0
        life = env.deg.projected_life_years(rate)
        print(f"  {name:24s} {t['h2_kg']:7.1f} kg  {rate:6.2f} uV/h  "
              f"-> {life:5.2f} yr  ({t['cycles']} cycles, {t['curt_kwh']:.0f} kWh curtailed)")
    print("  -> calibration target: 4.0 +/- 0.5 uV/h  (~5 yr, ref #7)")

    print("\nSkeleton runs end to end. Day 1 can start.")


if __name__ == "__main__":
    main()
