"""Degradation calibration. OWNER: Person A. DECISIONS.md section 5, brief item 5.

THE PROBLEM. Two targets pull against each other:

  ABSOLUTE  the rule-based baseline [8] must land at 4.0 +/- 0.5 uV/h, i.e. the ~5-year
            PEM stack life reported in [7] at a 10% (177 mV) end-of-life criterion.
  RATIO     a jittery policy must degrade >= 3x faster than a smooth one, or the
            degradation model cannot teach the agent anything and there is no paper.

Scaling every coefficient up fixes the ratio and breaks the absolute rate. Scaling down
does the reverse. They have to be solved together.

THE METHOD. The five terms are linear in their five coefficients (shape parameters are
held fixed by DECISIONS.md 5), so for any policy p,

    dv_total(p) = c . E(p),   E(p) = sum over steps of DegradationModel.basis(...)

One rollout per policy gives the exposure matrix E. The calibration is then a small
constrained program over c >= 0:

    equality    c . E(naive) / 24 h            = 4.0 uV/h
    inequality  c . E(jittery) / c . E(smooth) >= 3.3      (margin over the 3.0 gate)
    inequality  c . E(jittery) / 24 h          <= 50 uV/h  (worst-case bound, [4])
    bounds      each coefficient inside its published interval
    objective   stay as close as possible to the centre of those intervals, in log space,
                so the solution is the LEAST distorted parameter set that satisfies the
                data rather than whichever corner the solver reaches first.

That last line is the point. It makes the answer reproducible and reviewable: every
coefficient can be reported with the interval it came from and its position inside it.

USAGE
    python scripts/calibrate_degradation.py                  # synthetic profiles
    python scripts/calibrate_degradation.py --profiles data/processed/kutch_2019_1min.parquet
    python scripts/calibrate_degradation.py --write          # write back to the config

RE-RUN THIS when Person C's real Kutch profiles land. The procedure does not change; only
the exposure matrix does. That is the whole reason it is a script and not a hand-tune.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from scipy.optimize import minimize

from pemwe import load_config, PEMWEEnv, BASELINES
from pemwe.degradation import TERMS

# --- literature intervals -----------------------------------------------------------
# [n] are paper/REFERENCES.md. These are the HARD constraints: the solver may move inside
# them and nowhere else, so no coefficient can be justified after the fact.
BOUNDS = {
    "base":   (1.0, 10.0,  "steady-state PEMWE voltage rise, 1-10 uV/h [3]"),
    "stress": (5.0, 60.0,  "high-j Ir dissolution / membrane thinning, superlinear [1,5]"),
    "ramp":   (0.5, 40.0,  "thermal fatigue per unit |dj/dt| [4]"),
    "cycle":  (0.05, 5.0,  "voltage rise per ON/OFF transition; start/stop is the "
                           "dominant intermittency mechanism [4]"),
    "idle":   (0.5, 10.0,  "OCV hold, anode Ir dissolution [1,5]"),
}
CFG_KEYS = {"base": "r_base_uv_per_h", "stress": "k_j_uv_per_h", "ramp": "k_ramp_uv_per_h",
            "cycle": "dv_cycle_uv", "idle": "r_idle_uv_per_h"}

TARGET_RATE = 4.0        # uV/h, baseline  [7] via DECISIONS.md 5
RATIO_MIN = 3.3          # jittery/smooth; gate is 3.0, we solve with margin
WORST_CASE_MAX = 50.0    # uV/h ceiling for an aggressive policy [4]


# --- the probe policies -------------------------------------------------------------

class Jittery:
    """Hand-scripted adversary for Gate G1 item 2. Same definition as smoke_test.py."""
    name = "jittery"

    def __init__(self, cfg, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def act(self, obs, info=None):
        return np.array([self.rng.uniform(-1, 1)], dtype=np.float32)


class Smooth:
    """Deliberately gentle: follow the resource, capped well below rated."""
    name = "smooth"

    def __init__(self, cfg, seed=0):
        pass

    def reset(self):
        pass

    def act(self, obs, info=None):
        return np.array([2 * min(float(obs[0]), 0.55) - 1], dtype=np.float32)


def exposure(cfg, policy_cls, profiles=None, seeds=(0, 1, 2, 3, 4)):
    """Mean per-day exposure vector E and mean per-day hours, over several days."""
    tot = np.zeros(5)
    for sd in seeds:
        env = PEMWEEnv(cfg, profiles=profiles, seed=sd)
        obs, _ = env.reset(seed=sd)
        pol = policy_cls(cfg) if policy_cls in BASELINES.values() else policy_cls(cfg, sd)
        pol.reset()
        j_prev, was_on = 0.0, False
        while True:
            obs, r, term, trunc, info = env.step(pol.act(obs))
            tot += env.deg.basis(info["j"], j_prev, env.stack.j_rated,
                                 env.dt_min, info["is_on"], was_on)
            j_prev, was_on = info["j"], info["is_on"]
            if term or trunc:
                break
    return tot / len(seeds)


def solve(E, hours=24.0):
    """Constrained solve in log-space. Returns the coefficient vector in TERMS order."""
    lo = np.array([BOUNDS[t][0] for t in TERMS])
    hi = np.array([BOUNDS[t][1] for t in TERMS])
    centre = np.sqrt(lo * hi)                      # geometric centre of each interval

    def unpack(x):
        return np.exp(x)

    def obj(x):
        # squared log-distance from the interval centres: the least-distorted set
        return float(np.sum((x - np.log(centre)) ** 2))

    cons = [
        {"type": "eq",
         "fun": lambda x: unpack(x) @ E["naive"] / hours - TARGET_RATE},
        {"type": "ineq",
         "fun": lambda x: unpack(x) @ E["jittery"] - RATIO_MIN * (unpack(x) @ E["smooth"])},
        {"type": "ineq",
         "fun": lambda x: WORST_CASE_MAX - unpack(x) @ E["jittery"] / hours},
    ]
    res = minimize(obj, np.log(centre), method="SLSQP",
                   bounds=list(zip(np.log(lo), np.log(hi))),
                   constraints=cons, options={"maxiter": 800, "ftol": 1e-12})
    if not res.success:
        raise SystemExit(f"calibration did not converge: {res.message}")
    return unpack(res.x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default=None,
                    help="parquet from Person C; synthetic days if omitted")
    ap.add_argument("--write", action="store_true",
                    help="write the solved coefficients back into configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config()

    profiles = None
    if args.profiles:
        import pandas as pd
        df = pd.read_parquet(args.profiles)
        n = cfg["env"]["steps_per_episode"]
        col = "hybrid_w" if "hybrid_w" in df.columns else df.columns[0]
        v = df[col].to_numpy(float)
        profiles = v[: len(v) // n * n].reshape(-1, n)
        print(f"profiles: {args.profiles} -> {profiles.shape[0]} days, column {col!r}")
    else:
        print("profiles: SYNTHETIC (re-run with --profiles when C's Kutch parquet lands)")

    print("\n--- exposure matrix E (per day, coefficient-free) ---")
    print(f"{'policy':<24}" + "".join(f"{t:>12}" for t in TERMS))
    E = {}
    for name, cls in [("naive", BASELINES["baseline_naive"]),
                      ("ramplimited", BASELINES["baseline_ramplimited"]),
                      ("smooth", Smooth), ("jittery", Jittery)]:
        E[name] = exposure(cfg, cls, profiles)
        print(f"{name:<24}" + "".join(f"{v:12.4f}" for v in E[name]))

    c = solve(E)

    print("\n--- solved coefficients, with position inside the literature interval ---")
    for t, val in zip(TERMS, c):
        lo, hi, src = BOUNDS[t]
        pos = (np.log(val) - np.log(lo)) / (np.log(hi) - np.log(lo))
        print(f"  {CFG_KEYS[t]:<20} {val:8.3f}   [{lo:g}, {hi:g}] at {pos*100:4.0f}% "
              f"of the log-interval\n      {src}")

    print("\n--- resulting rates ---")
    rates = {k: c @ v / 24.0 for k, v in E.items()}
    dv_eol = cfg["degradation"]["dv_eol_uv"]
    for k, r in rates.items():
        print(f"  {k:<14} {r:7.2f} uV/h   -> {dv_eol / r / 8760.0:6.2f} yr")
    print(f"  jittery/smooth ratio = {rates['jittery'] / rates['smooth']:.2f}x "
          f"(gate >= 3.0, solved with margin to {RATIO_MIN})")

    ok = (abs(rates["naive"] - TARGET_RATE) <= 0.5
          and rates["jittery"] / rates["smooth"] >= 3.0
          and rates["jittery"] <= WORST_CASE_MAX)
    print(f"\n  all calibration targets met: {'YES' if ok else 'NO'}")

    if args.write:
        path = ROOT / "configs" / "default.yaml"
        text = path.read_text()
        for t, val in zip(TERMS, c):
            key = CFG_KEYS[t]
            out, done = [], False
            for line in text.splitlines():
                stripped = line.lstrip()
                if not done and stripped.startswith(key + ":"):
                    indent = line[: len(line) - len(stripped)]
                    _, _, rest = line.partition(":")
                    comment = rest.split("#", 1)
                    tail = ("  #" + comment[1]) if len(comment) > 1 else ""
                    out.append(f"{indent}{key}: {val:.4f}{tail}")
                    done = True
                else:
                    out.append(line)
            text = "\n".join(out) + "\n"
        path.write_text(text)
        print(f"\n  wrote solved coefficients to {path}")
    else:
        print("\n  (dry run -- pass --write to update configs/default.yaml)")


if __name__ == "__main__":
    main()
