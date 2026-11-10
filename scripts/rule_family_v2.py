"""A smarter simple rule: hysteresis (minimum on/off confirmation), a power cap and a slew limit.

    python scripts/rule_family_v2.py --workers 6 [--split test] [--out results_hpc/rule_family_v2.csv]

Why: the one-dimensional rule families (slew limit, power cap) leave an obvious objection to the learned
controller's low-degradation advantage: "a smarter rule would close the gap". This rule avoids marginal
starts and stops explicitly, which is what the learned policy was found to do (policy_probe.csv).

Rule (all quantities are fractions of rated power):
  OFF -> ON   when the resource has been at least `p_on` for `k_up` consecutive minutes
  ON  -> OFF  when the resource has been below `p_off` for `k_down` consecutive minutes
  while ON    command min(resource, cap), moved by at most `slew` per minute from the previous command
              (a start therefore ramps up from zero at the slew limit)
The grid is reported in full; nothing is selected on the evaluation days. Runs on any OS (spawn pool).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                    # noqa: E402

P_ON = (0.08, 0.15)
K_UP = (5, 20)
K_DOWN = (5, 20)
SLEW = (0.001, 0.003, 0.01)
CAP = (0.5, 0.7, 1.0)
P_OFF = 0.03


def _to_action(target_frac_of_rated: float, p_renew_frac: float) -> np.ndarray:
    """Same conversion as pemwe.baselines._to_action (target power as a fraction of rated -> env action,
    which is a fraction of the AVAILABLE power). Copied so the rule can be unit-tested without the simulator."""
    u = 0.0 if p_renew_frac <= 1e-9 else float(np.clip(target_frac_of_rated / p_renew_frac, 0.0, 1.0))
    return np.array([2.0 * u - 1.0], dtype=np.float32)


class HysteresisRule:
    def __init__(self, p_on, k_up, k_down, slew, cap, p_off=P_OFF):
        self.p_on, self.k_up, self.k_down, self.slew, self.cap, self.p_off = p_on, k_up, k_down, slew, cap, p_off
        self.reset()

    def reset(self):
        self.on, self.up, self.down, self.prev = False, 0, 0, 0.0

    def act(self, obs, info=None):
        p = float(obs[0])
        if not self.on:
            self.up = self.up + 1 if p >= self.p_on else 0
            if self.up >= self.k_up:
                self.on, self.up, self.down, self.prev = True, 0, 0, 0.0
        else:
            self.down = self.down + 1 if p < self.p_off else 0
            if self.down >= self.k_down:
                self.on, self.down, self.prev = False, 0, 0.0
        if not self.on:
            return _to_action(0.0, p)
        target = min(p, self.cap)
        self.prev = float(np.clip(target, self.prev - self.slew, self.prev + self.slew))
        return _to_action(min(self.prev, p), p)


_CACHE = {}


def unit(args):
    params, split = args
    from pemwe.config import load_config
    from pemwe.env import PEMWEEnv
    from pemwe.evaluate import rollout_episode
    from pemwe import profiles
    cfg = load_config()
    if split not in _CACHE:
        df = profiles.load_profiles(ROOT / cfg["data"]["profile_path"])
        dates = profiles.SPLITS[split]
        _CACHE[split] = (dates, profiles.profiles_array(df, dates, "hybrid"))
    dates, prof = _CACHE[split]
    env = PEMWEEnv(cfg, profiles=prof, seed=0)
    pol = HysteresisRule(*params)
    eps = [rollout_episode(env, pol, i, 0)[0] for i in range(len(dates))]
    p_on, k_up, k_down, slew, cap = params
    return dict(p_on=p_on, k_up=k_up, k_down=k_down, slew=slew, cap=cap, split=split, days=len(eps),
                h2_kg_day=float(np.mean([e["h2_kg"] for e in eps])),
                dv_uv_day=float(np.mean([e["dv_deg_uv"] for e in eps])),
                transitions_day=float(np.mean([e["n_cycles"] for e in eps])),
                curtailed_kwh_day=float(np.mean([e["curtailed_kwh"] for e in eps])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--out", default=str(ROOT / "results_hpc" / "rule_family_v2.csv"))
    ap.add_argument("--quick", action="store_true", help="2 grid points (smoke test)")
    a = ap.parse_args()
    grid = list(itertools.product(P_ON, K_UP, K_DOWN, SLEW, CAP))
    if a.quick:
        grid = grid[:2]
    units = [(g, a.split) for g in grid]
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        rows = pool.map(unit, units, chunksize=1)
    print(f"{len(units)} units in {time.perf_counter() - t0:.1f}s")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    best = sorted(rows, key=lambda r: -r["h2_kg_day"])
    print("lowest-degradation points:")
    for r in sorted(rows, key=lambda r: r["dv_uv_day"])[:6]:
        print(f"  dV {r['dv_uv_day']:6.1f}  H2 {r['h2_kg_day']:6.1f}  trans {r['transitions_day']:4.1f}  "
              f"p_on={r['p_on']} k_up={r['k_up']} k_down={r['k_down']} slew={r['slew']} cap={r['cap']}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
