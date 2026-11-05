"""Experiments A and B: controller behaviour under degradation-coefficient and resource-variability uncertainty.

    python scripts/analyze_e6.py --csv results_e6/grid_baselines_test.csv [--out-dir results_hpc] [--png]

Input is the per-(controller, scenario, variant, day) table written by scripts/run_scenario_grid.py.
Outputs (all computed from that table, nothing typed in by hand):
  degradation_sensitivity.csv      controller x coefficient variant (original profile)
  renewable_scenarios_results.csv  controller x scenario x noise realisation (nominal coefficients)
  controller_comparison.csv        nominal case, paired differences against a reference, bootstrap over days
  fig_sensitivity.pdf              projected life by coefficient variant and by scenario
  rank_stability (printed)         whether the ordering of controllers changes across rows

Projected life uses the repository convention dv_eol / (mean daily rise / 24 h) with dv_eol = 177,000 uV.
That literal is 10% of 1.77 V, not of the model's own rated cell voltage (1.846 V); we keep it for
comparability with the rest of the study (audit finding C3) and the figure's absolute values are
therefore on a calibrated, not physical, scale. The unit of analysis for intervals is the day;
days share weather, so intervals describe this day set only.
"""
from __future__ import annotations

import argparse
import itertools
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

DV_EOL = 177000.0
VARIANTS = ["nominal", "base_low", "base_high", "stress_low", "stress_high", "ramp_low", "ramp_high",
            "cycle_low", "cycle_high", "idle_low", "idle_high", "all_low", "all_high"]
SCEN = ["low_variability", "original", "high_solar", "high_wind", "combined_high"]
STYLE = {"naive_as_published": (H.BLUE, "o", "published rule (10% threshold)"),
         "no_threshold": (H.ORANGE, "s", "threshold removed"),
         "ramp_limited": (H.AQUA, "^", "tuned slew limit")}


def summarise(d: pd.DataFrame) -> pd.DataFrame:
    a = d.groupby(["controller", "scenario", "realisation", "variant"]).agg(
        h2_kg_day=("h2_kg", "mean"), dv_uv_day=("dv_deg_uv", "mean"), transitions_day=("n_cycles", "mean"),
        curtailed_kwh_day=("curtailed_kwh", "mean"), days=("date", "nunique")).reset_index()
    a["life_yr"] = DV_EOL / (a.dv_uv_day / 24.0) / 8760.0
    return a


def paired_table(d: pd.DataFrame, ref: str, rng) -> pd.DataFrame:
    n = d[(d.scenario == "original") & (d.variant == "nominal") & (d.realisation == 0)]
    r = n[n.controller == ref].set_index("date")
    rows = []
    for c, g in n.groupby("controller"):
        g = g.set_index("date")
        for col, nm in (("h2_kg", "h2_kg_day"), ("dv_deg_uv", "dv_uv_day"), ("n_cycles", "transitions_day"),
                        ("curtailed_kwh", "curtailed_kwh_day")):
            diff = (g[col] - r[col]).dropna().to_numpy()
            m, lo, hi = H.boot_ci(diff, rng=rng)
            rows.append(dict(controller=c, reference=ref, metric=nm, mean_diff=m, ci_lo=lo, ci_hi=hi,
                             pct_days_lower=float((diff < 0).mean() * 100), days=len(diff)))
    return pd.DataFrame(rows)


def rank_stability(a: pd.DataFrame, key: str, by: str, label: str):
    orders = {}
    for k, g in a.groupby(key):
        g = g.groupby("controller")[by].mean().sort_values(ascending=False)
        orders[k] = tuple(g.index)
    uniq = sorted(set(orders.values()))
    print(f"  {label}: {len(uniq)} distinct ordering(s) across {len(orders)} rows")
    for u in uniq:
        rows = [k for k, v in orders.items() if v == u]
        print(f"    {' > '.join(u)}   [{len(rows)} rows: {', '.join(map(str, rows[:6]))}{'...' if len(rows) > 6 else ''}]")


def dot_panel(ax, rows, order, getter, title):
    ys = np.arange(len(order))[::-1]
    allx = []
    dodge = {c: o for c, o in zip(STYLE, (0.22, 0.0, -0.22))}      # keeps tied values visible
    for y, key in zip(ys, order):
        vals = []
        for c, (col, mk, lab) in STYLE.items():
            v = getter(c, key)
            if v is None:
                continue
            vals.append(v)
            ax.plot(v, y + dodge[c], marker=mk, color=col, ms=5.5, mec=H.SURFACE, mew=0.8, ls="", zorder=3,
                    label=lab if y == ys[0] else None)
        if len(vals) > 1:
            ax.plot([min(vals), max(vals)], [y, y], color=H.BASE, lw=0.9, zorder=2)
        allx += vals
    ax.set_yticks(ys, [o.replace("_", " ") for o in order])
    ax.set_title(title, loc="left", color=H.INK)
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    return allx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", default=str(ROOT / "results_hpc"))
    ap.add_argument("--fig-dir", default=str(ROOT / "results" / "figures"))
    ap.add_argument("--ref", default="naive_as_published")
    ap.add_argument("--png", action="store_true")
    a = ap.parse_args()
    H.PNG = a.png
    H.setup()
    rng = np.random.default_rng(0)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(a.csv)
    # The scenario sweep's `original` realisation 0 is the same series as the coefficient sweep's
    # nominal run on the shipped profile; keep one copy so no day is counted twice.
    n0 = len(d)
    d = d.drop_duplicates(subset=["controller", "scenario", "realisation", "variant", "date"])
    print(f"rows: {n0} -> {len(d)} after removing exact duplicate (controller, scenario, realisation, variant, date) keys")
    agg = summarise(d)

    sens = agg[(agg.scenario == "original") & (agg.realisation == 0)]
    sens.to_csv(out / "degradation_sensitivity.csv", index=False)
    scen = agg[agg.variant == "nominal"]
    scen.to_csv(out / "renewable_scenarios_results.csv", index=False)
    cmp_ = paired_table(d, a.ref, rng)
    cmp_.to_csv(out / "controller_comparison.csv", index=False)

    pd.set_option("display.width", 220)
    print("== nominal, original profile: mean per day over the test days (paired vs", a.ref, ") ==")
    print(cmp_.round(2).to_string(index=False))
    print("\n== rank stability ==")
    rank_stability(sens, "variant", "life_yr", "coefficient variants, ordered by projected life")
    rank_stability(sens, "variant", "h2_kg_day", "coefficient variants, ordered by hydrogen")
    sc = scen.groupby(["controller", "scenario"]).agg(life_yr=("life_yr", "mean"), h2_kg_day=("h2_kg_day", "mean")).reset_index()
    rank_stability(sc, "scenario", "life_yr", "scenarios, ordered by projected life")
    rank_stability(sc, "scenario", "h2_kg_day", "scenarios, ordered by hydrogen")

    # ---- figure
    sv = sens.set_index(["controller", "variant"]).life_yr
    sg = scen.groupby(["controller", "scenario"]).life_yr.agg(["mean", "min", "max"])
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 3.3), gridspec_kw={"width_ratios": [1.15, 1]})
    xa = dot_panel(ax[0], None, VARIANTS, lambda c, k: sv.get((c, k)), "(a) Degradation coefficients")
    xb = dot_panel(ax[1], None, SCEN, lambda c, k: (sg.loc[(c, k), "mean"] if (c, k) in sg.index else None),
                   "(b) Renewable variability")
    lo, hi = min(xa + xb) * 0.85, max(xa + xb) * 1.15
    for a_ in ax:
        a_.set_xscale("log")
        a_.set_xlim(lo, hi)
        a_.set_xticks([t for t in (1, 2, 3, 5, 10, 20) if lo <= t <= hi])
        a_.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        a_.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        a_.set_xlabel("projected life (years, calibrated scale)")
    h, lab = ax[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=7.2, handletextpad=0.3)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path = Path(a.fig_dir) / "fig_sensitivity.pdf"
    H._save(fig, path)
    print("wrote", path)


if __name__ == "__main__":
    main()
