"""Experiment E3: learning quality versus update-to-data ratio (UTD), from checkpoints.

    python scripts/analyze_e3.py --grid results_pilot/e3_grid.csv --curves results_e3 [--w2 0.1] [--png]

Inputs
  --grid    run_scenario_grid.py output for every checkpoint (one row per controller, day) plus the
            constant-action-+1 reference, all on the same day subset
  --curves  directory with one progress CSV per run (timesteps, wall_s, n_updates, ep_rew_mean)
Run ids must look like  e3_utd<UTD>_s<seed>  and checkpoints like  e3_utd<UTD>_s<seed>_<steps>_steps.

Quality = per-day reward of the policy minus the reward of the constant-action-+1 controller on the
same days, divided by that controller's mean reward (so 0 means "as good as the simple controller" and
negative is worse). Reward is recomputed from the logged unweighted components at the stated weights.
Wall-clock comes from the training logs and was measured with several runs sharing the machine, so
it is comparable across the runs of this experiment but not a single-run speed.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import numpy as np                                      # noqa: E402
import pandas as pd                                     # noqa: E402

import analyze_hpc as H                                 # noqa: E402

# ordinal single-hue ramp (blue steps 250-600 of the reference palette): UTD is ordered, not nominal
RAMP = {"0.03125": "#86b6ef", "0.125": "#5598e7", "0.5": "#2a78d6", "1": "#184f95"}
MARK = {"0.03125": "o", "0.125": "s", "0.5": "^", "1": "D"}
NAME = {"0.03125": "UTD 1/32", "0.125": "UTD 1/8", "0.5": "UTD 1/2", "1": "UTD 1"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--curves", required=True)
    ap.add_argument("--w1", type=float, default=1.0)
    ap.add_argument("--w2", type=float, default=0.1)
    ap.add_argument("--w3", type=float, default=0.1)
    ap.add_argument("--ref", default="no_threshold")
    ap.add_argument("--fig-dir", default=str(ROOT / "results" / "figures"))
    ap.add_argument("--out-csv", default=str(ROOT / "results_hpc" / "e3_quality.csv"))
    ap.add_argument("--png", action="store_true")
    a = ap.parse_args()
    H.PNG = a.png
    H.setup()

    g = pd.read_csv(a.grid)
    g["reward"] = a.w1 * g.r_yield - a.w2 * g.r_deg - a.w3 * g.r_ramp
    ref = g[g.controller == a.ref].set_index("date").reward
    ref_mean = ref.mean()
    rows = []
    for c, d in g.groupby("controller"):
        m = re.match(r"e3_utd([\d.]+)_s(\d+)_(\d+)_steps", c)
        if not m:
            continue
        utd, seed, steps = m.group(1), int(m.group(2)), int(m.group(3))
        diff = (d.set_index("date").reward - ref).dropna()
        rows.append(dict(utd=utd, seed=seed, steps=steps, gap_pct=100.0 * diff.mean() / ref_mean,
                         h2_kg=d.h2_kg.mean(), dv_uv=d.dv_deg_uv.mean(), transitions=d.n_cycles.mean()))
    q = pd.DataFrame(rows)
    if q.empty:
        sys.exit("no e3 checkpoints found in the grid file")

    # attach wall-clock and update count from the progress logs (first logged step at or after the checkpoint)
    wall, upd = [], []
    for r in q.itertuples():
        p = Path(a.curves) / f"e3_utd{r.utd}_s{r.seed}.csv"
        if p.exists():
            t = pd.read_csv(p)
            t = t[t.timesteps >= r.steps]
            wall.append(float(t.wall_s.iloc[0]) if len(t) else np.nan)
            upd.append(float(t.n_updates.iloc[0]) if len(t) else np.nan)
        else:
            wall.append(np.nan)
            upd.append(np.nan)
    q["wall_min"] = np.array(wall) / 60.0
    q["updates"] = upd
    Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
    q.to_csv(a.out_csv, index=False)

    agg = q.groupby(["utd", "steps"]).agg(gap=("gap_pct", "mean"), gap_min=("gap_pct", "min"), gap_max=("gap_pct", "max"),
                                          wall=("wall_min", "mean"), updates=("updates", "mean"), n=("seed", "count")).reset_index()
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.6))
    for k, (xcol, xl) in enumerate((("steps", "environment steps (thousands)"), ("updates", "gradient updates (thousands)"),
                                    ("wall", "wall-clock (minutes, shared GPU)"))):
        for utd in RAMP:
            d = agg[agg.utd == utd].sort_values(xcol)
            if d.empty:
                continue
            x = d[xcol] / (1000.0 if xcol != "wall" else 1.0)
            ax[k].plot(x, d.gap, color=RAMP[utd], marker=MARK[utd], ms=5, lw=1.5, mec=H.SURFACE, mew=0.8, label=NAME[utd])
            ax[k].fill_between(x, d.gap_min, d.gap_max, color=RAMP[utd], alpha=0.18, lw=0)
        ax[k].axhline(0, color=H.MUTED, lw=1.0, ls=(0, (4, 3)))
        ax[k].set_xlabel(xl)
        ax[k].set_ylabel("reward vs constant +1 (%)" if k == 0 else "")
        ax[k].set_title(["(a) vs env steps", "(b) vs updates", "(c) vs wall-clock"][k], loc="left", color=H.INK)
    ax[0].legend(frameon=False, loc="lower right", handlelength=2.2, fontsize=7)
    fig.tight_layout()
    out = Path(a.fig_dir) / "fig_e3_utd.pdf"
    H._save(fig, out)
    pd.set_option("display.width", 200)
    print(agg.round(2).to_string(index=False))
    print("wrote", out, "and", a.out_csv)


if __name__ == "__main__":
    main()
