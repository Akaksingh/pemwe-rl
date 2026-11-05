"""Paired comparison of controllers against a reference, from run_scenario_grid.py output.

    python scripts/summarize_pilots.py --csv results_pilot/grid_round2.csv --w2 0.1 [--ref no_threshold]

Reward is recomputed from the logged, unweighted components with the ORIGINAL scales and the
stated weights, so every controller is judged by the same objective regardless of how it was
trained (for example with a rescaled reward). The unit of analysis is the day: per-day reward
differences against the reference, with a bootstrap interval over days for the mean. Days are
correlated through the weather, so the interval describes this set of days only; it is not a
statement about other weather or about seed-to-seed variation.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--w1", type=float, default=1.0)
    ap.add_argument("--w2", type=float, default=0.1)
    ap.add_argument("--w3", type=float, default=0.1)
    ap.add_argument("--ref", default="no_threshold")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    d = pd.read_csv(a.csv)
    d = d[(d.scenario == "original") & (d.variant == "nominal")].copy()
    d["reward"] = a.w1 * d.r_yield - a.w2 * d.r_deg - a.w3 * d.r_ramp
    ref = d[d.controller == a.ref].set_index("date")
    rng = np.random.default_rng(a.seed)
    rows = []
    for c, g in d.groupby("controller"):
        g = g.set_index("date")
        diff = (g.reward - ref.reward).dropna().to_numpy()
        boots = np.array([rng.choice(diff, len(diff)).mean() for _ in range(4000)])
        rows.append(dict(controller=c, days=len(g), h2_kg=g.h2_kg.mean(), dv_uv_day=g.dv_deg_uv.mean(),
                         transitions=g.n_cycles.mean(), curtailed_kwh=g.curtailed_kwh.mean(),
                         d_reward=diff.mean(), ci_lo=np.quantile(boots, 0.025), ci_hi=np.quantile(boots, 0.975),
                         pct_days_better=float((diff > 0).mean() * 100)))
    t = pd.DataFrame(rows).sort_values("d_reward", ascending=False)
    pd.set_option("display.width", 220)
    print(f"w=({a.w1},{a.w2},{a.w3}); reference = {a.ref}; split/days as in the CSV")
    print(t.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
