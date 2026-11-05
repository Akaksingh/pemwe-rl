"""Yield-degradation frontier: corrected SAC (E5) against the legacy SAC runs and the rule-based baselines.

    python scripts/analyze_e5.py [--e5 results_e5/grid_e5_test.csv] [--base results_hpc/degradation_sensitivity.csv]
                                 [--legacy-dir results] [--png]

All three sources are evaluated on the same 72 held-out test days with the same environment:
  baselines  rows of degradation_sensitivity.csv (variant = nominal, original profile) from run_scenario_grid.py
  legacy     results/sac_w2-*_seed*.json: the earlier SAC runs (update-to-data ratio 0.032, gamma 0.999,
             2M steps), evaluated by pemwe.evaluate (same env, same days)
  corrected  E5 runs (update-to-data ratio 1, gamma from the pilots, 300k steps) from the grid CSV

The test split is used here once, for the final evaluation of policies whose settings were chosen
from in-sample training-day pilots. Axes: lower degradation (x) and higher hydrogen (y) are better.
Each corrected point is the mean over seeds; the faint dots are individual seeds. A point is reported as
"dominated" when some other controller has at least as much hydrogen AND no more degradation.
"""
from __future__ import annotations

import argparse
import glob
import json
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

BASE_STYLE = {"naive_as_published": ("o", "published rule"), "no_threshold": ("s", "threshold removed"),
              "ramp_limited": ("^", "tuned slew limit")}


def legacy_points(d: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(d / "sac_w2-*_seed*.json"))):
        m = re.search(r"w2-([\d.]+)_seed(\d+)", f)
        a = json.load(open(f))["aggregate"]
        rows.append(dict(w2=float(m[1]), seed=int(m[2]), h2=a["h2_kg_mean"], dv=a["dv_deg_uv_mean"]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e5", default=str(ROOT / "results_e5" / "grid_e5_test.csv"))
    ap.add_argument("--base", default=str(ROOT / "results_hpc" / "degradation_sensitivity.csv"))
    ap.add_argument("--legacy-dir", default=str(ROOT / "results"))
    ap.add_argument("--fig-dir", default=str(ROOT / "results" / "figures"))
    ap.add_argument("--out-csv", default=str(ROOT / "results_hpc" / "e5_frontier.csv"))
    ap.add_argument("--png", action="store_true")
    a = ap.parse_args()
    H.PNG = a.png
    H.setup()

    b = pd.read_csv(a.base)
    b = b[(b.variant == "nominal") & (b.scenario == "original") & (b.realisation == 0)].set_index("controller")
    leg = legacy_points(Path(a.legacy_dir))

    new = pd.DataFrame()
    if Path(a.e5).exists():
        g = pd.read_csv(a.e5).drop_duplicates(subset=["controller", "scenario", "realisation", "variant", "date"])
        g = g[(g.scenario == "original") & (g.variant == "nominal") & (g.realisation == 0)]
        rows = []
        for c, d in g.groupby("controller"):
            m = re.match(r"e5_w2-([\d.]+)_s(\d+)", c)
            if m:
                rows.append(dict(w2=float(m[1]), seed=int(m[2]), h2=d.h2_kg.mean(), dv=d.dv_deg_uv.mean(),
                                 transitions=d.n_cycles.mean(), curtailed=d.curtailed_kwh.mean()))
        new = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    # legacy SAC: mean over seeds per w2, a line
    lm = leg.groupby("w2").agg(h2=("h2", "mean"), dv=("dv", "mean")).reset_index().sort_values("w2")
    ax.plot(lm.dv, lm.h2, color=H.AQUA, marker="o", ms=4, lw=1.3, mec=H.SURFACE, mew=0.6, zorder=3,
            label="SAC, earlier runs (UTD 0.03, 2M steps)")
    if len(new):
        nm = new.groupby("w2").agg(h2=("h2", "mean"), dv=("dv", "mean")).reset_index().sort_values("w2")
        ax.scatter(new.dv, new.h2, s=10, color=H.BLUE, alpha=0.35, linewidths=0, zorder=3)
        ax.plot(nm.dv, nm.h2, color=H.BLUE, marker="D", ms=5, lw=1.5, mec=H.SURFACE, mew=0.8, zorder=4,
                label="SAC, corrected (UTD 1, 300k steps)")
        for r in nm.itertuples():
            ax.annotate(f"{r.w2:g}", (r.dv, r.h2), textcoords="offset points", xytext=(4, -9), fontsize=6.3, color=H.INK2)
    for c, (mk, lab) in BASE_STYLE.items():
        if c in b.index:
            ax.plot(b.loc[c, "dv_uv_day"], b.loc[c, "h2_kg_day"], marker=mk, ms=7, mfc=H.SURFACE, mec=H.INK, mew=1.3,
                    ls="", zorder=5)
            ax.annotate(lab, (b.loc[c, "dv_uv_day"], b.loc[c, "h2_kg_day"]), textcoords="offset points",
                        xytext=(5, 3), fontsize=6.8, color=H.INK)
    ax.set_xlabel("degradation (µV / day, lower is better)")
    ax.set_ylabel("hydrogen (kg / day)")
    ax.invert_xaxis()
    ax.grid(axis="x", visible=True)
    ax.legend(frameon=False, loc="lower left", fontsize=6.6, handlelength=1.8)
    fig.tight_layout()
    out = Path(a.fig_dir) / "fig_frontier.pdf"
    H._save(fig, out)

    # ---- dominance report
    allc = [(f"baseline:{c}", b.loc[c, "dv_uv_day"], b.loc[c, "h2_kg_day"]) for c in b.index]
    allc += [(f"legacy w2={r.w2:g}", r.dv, r.h2) for r in lm.itertuples()]
    if len(new):
        allc += [(f"corrected w2={r.w2:g}", r.dv, r.h2) for r in nm.itertuples()]
    print(f"{'point':26s}{'H2 kg/d':>9}{'dV uV/d':>9}   dominated by")
    for n, dv, h2 in allc:
        dom = [m for m, dv2, h22 in allc if m != n and h22 >= h2 and dv2 <= dv and (h22 > h2 or dv2 < dv)]
        print(f"{n:26s}{h2:9.1f}{dv:9.1f}   {', '.join(dom[:3]) + (' ...' if len(dom) > 3 else '') if dom else '-'}")
    if len(new):
        new.to_csv(a.out_csv, index=False)
        print(new.sort_values(["w2", "seed"]).round(2).to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
