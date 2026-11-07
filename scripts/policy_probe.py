"""What does a trained policy actually do? Action versus available power, from rollouts.

    python scripts/policy_probe.py --models models/e3_utd1_s0.zip models/e5_w2-20.0_s0.zip [--split test]

For every step with enough resource to run (available power above the plant's minimum), records the
fraction u of available power the policy commands and whether the plant is on, and bins them by
available power (fraction of rated). The constant-action-+1 controller is included as the reference
(u = 1 everywhere). This is a descriptive probe of a few checkpoints on a day subset, not a statistic.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                    # noqa: E402

from pemwe.config import load_config                  # noqa: E402

BINS = [(0.035, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, 0.70), (0.70, 1.0)]


def probe(policy, cfg, split, stride):
    from pemwe.env import PEMWEEnv
    from pemwe import profiles
    df = profiles.load_profiles(ROOT / cfg["data"]["profile_path"])
    dates = profiles.SPLITS[split][::stride]
    env = PEMWEEnv(cfg, profiles=profiles.profiles_array(df, dates, "hybrid"), seed=0)
    avail, u, on = [], [], []
    for i in range(len(dates)):
        obs, _ = env.reset(seed=0, options={"day_idx": i})
        policy.reset()
        for _ in range(env.n_steps):
            a = policy.act(obs)
            p = float(obs[0])
            obs, _, _, trunc, info = env.step(a)
            avail.append(p)
            u.append(0.5 * (float(a[0]) + 1.0))
            on.append(bool(info["is_on"]))
            if trunc:
                break
    return np.array(avail), np.array(u), np.array(on), len(dates)


def report(name, avail, u, on, days, rows=None):
    pmin = 0.035
    m = avail > pmin
    print(f"\n{name}  ({days} days)   steps with resource above plant minimum: {m.sum()}")
    print(f"  mean commanded fraction u = {u[m].mean():.3f};  share of steps with u < 0.95: {(u[m] < 0.95).mean():.3f};"
          f"  plant OFF although resource available: {(~on[m]).mean():.3f}")
    print(f"  {'available power':>18} {'steps':>8} {'mean u':>8} {'u<0.95':>8} {'OFF':>7}")
    for lo, hi in BINS:
        s = (avail > lo) & (avail <= hi)
        if s.sum():
            print(f"  {f'{lo:.3f}-{hi:.2f} of rated':>18} {s.sum():8d} {u[s].mean():8.3f} {(u[s] < 0.95).mean():8.3f} {(~on[s]).mean():7.3f}")
            if rows is not None:
                rows.append(dict(policy=name, bin_lo=lo, bin_hi=hi, steps=int(s.sum()), mean_u=float(u[s].mean()),
                                 share_u_below_095=float((u[s] < 0.95).mean()), share_off=float((~on[s]).mean()), days=days))
    if rows is not None:
        rows.append(dict(policy=name, bin_lo=pmin, bin_hi=1.0, steps=int(m.sum()), mean_u=float(u[m].mean()),
                         share_u_below_095=float((u[m] < 0.95).mean()), share_off=float((~on[m]).mean()), days=days))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "results_hpc" / "policy_probe.csv"))
    a = ap.parse_args()
    from pemwe.baselines import BASELINES
    from pemwe.evaluate import SB3Policy
    cfg0 = load_config(overrides=["baseline.p_idle_threshold_frac=0.0"])
    rows = []
    report("constant +1 (reference)", *probe(BASELINES["baseline_naive"](cfg0), load_config(), a.split, a.stride), rows=rows)
    for m in a.models:
        report(Path(m).stem, *probe(SB3Policy(m, algo="sac"), load_config(), a.split, a.stride), rows=rows)
    import csv
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
