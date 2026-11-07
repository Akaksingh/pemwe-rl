"""Rule-based yield-degradation families, for the question: can a simple rule reach the low-degradation
operating points the learned policy reaches?

    OMP_NUM_THREADS=1 python scripts/baseline_frontier.py --workers 14 [--split test] [--out results_hpc/baseline_frontier.csv]

Families, all evaluated on the same days and environment as everything else (original profile, nominal
coefficients, hybrid resource):
  slew_thr10   the tuned baseline's design: load following with the 10% idle threshold and a slew limit L
  slew_nothr   the same without the idle threshold
  cap          constant ceiling: follow the resource but never above a fraction C of rated power
               (idle threshold removed)
The slew limits and caps are a grid, not tuned on the evaluation days; no value is selected here.
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                    # noqa: E402

from pemwe.config import load_config                  # noqa: E402

SLEW = (0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.045, 0.06)
CAPS = (0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class Cap:
    """Follow the resource, never above `cap` of rated power; no idle threshold."""
    def __init__(self, cap):
        self.cap = cap

    def reset(self):
        pass

    def act(self, obs, info=None):
        from pemwe.baselines import _to_action
        p = float(obs[0])
        return _to_action(min(p, self.cap), p)


def unit(args):
    family, param, split = args
    from pemwe.env import PEMWEEnv
    from pemwe.baselines import BASELINES
    from pemwe.evaluate import rollout_episode
    from pemwe import profiles

    ov = []
    if family.startswith("slew"):
        ov.append(f"baseline.ramp_limit_frac_per_min={param}")
        if family == "slew_nothr":
            ov.append("baseline.p_idle_threshold_frac=0.0")
    cfg = load_config(overrides=ov)
    df = profiles.load_profiles(ROOT / cfg["data"]["profile_path"])
    dates = profiles.SPLITS[split]
    env = PEMWEEnv(cfg, profiles=profiles.profiles_array(df, dates, "hybrid"), seed=0)
    pol = Cap(param) if family == "cap" else BASELINES["baseline_ramplimited"](cfg)
    eps = [rollout_episode(env, pol, i, 0)[0] for i in range(len(dates))]
    return dict(family=family, param=param, split=split, days=len(eps),
                h2_kg_day=float(np.mean([e["h2_kg"] for e in eps])),
                dv_uv_day=float(np.mean([e["dv_deg_uv"] for e in eps])),
                transitions_day=float(np.mean([e["n_cycles"] for e in eps])),
                curtailed_kwh_day=float(np.mean([e["curtailed_kwh"] for e in eps])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--out", default=str(ROOT / "results_hpc" / "baseline_frontier.csv"))
    a = ap.parse_args()
    units = [(f, p, a.split) for f in ("slew_thr10", "slew_nothr") for p in SLEW] + [("cap", c, a.split) for c in CAPS]
    t0 = time.perf_counter()
    with mp.get_context("fork").Pool(a.workers) as pool:
        rows = pool.map(unit, units, chunksize=1)
    print(f"{len(units)} units in {time.perf_counter() - t0:.1f}s")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['family']:11s} {r['param']:>7}  H2 {r['h2_kg_day']:6.1f}  dV {r['dv_uv_day']:6.1f}  trans {r['transitions_day']:5.1f}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
