"""Does the learned policies' lifetime advantage survive coefficient and resource uncertainty?

    python scripts/analyze_sac_sensitivity.py [--csv results_e5/grid_e5_test.csv] [--png]

Input: run_scenario_grid.py output with the three rule-based controllers and the E5 SAC checkpoints,
72 held-out test days, 13 degradation-coefficient variants (original profile) and 5 resource scenarios
x 3 noise realisations (nominal coefficients). The SAC policies were trained ONCE at nominal
coefficients on the original profile and are evaluated unchanged elsewhere: this is a robustness test
under distribution shift, not retraining. Seeds are averaged within each w2.

Reported per row: projected life (calibrated scale, dv_eol / mean daily rise) and hydrogen. The
comparison reference is the tuned slew-limited rule. "SAC outlives the rule" counts rows where the SAC
mean projected life exceeds the rule's; the hydrogen cost is reported next to it because a longer life
bought with much less hydrogen is not an improvement by itself.
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
from analyze_e6 import summarise, VARIANTS, SCEN        # noqa: E402

SHOW = [("ramp_limited", H.INK, "^", "tuned slew limit", True),
        ("SAC w2=5.32", H.BLUE, "o", "SAC, w2 = 5.32", False),
        ("SAC w2=12.9", H.ORANGE, "s", "SAC, w2 = 12.9", False),
        ("SAC w2=20", H.AQUA, "D", "SAC, w2 = 20", False)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "results_e5" / "grid_e5_test.csv"))
    ap.add_argument("--out-dir", default=str(ROOT / "results_hpc"))
    ap.add_argument("--fig-dir", default=str(ROOT / "results" / "figures"))
    ap.add_argument("--png", action="store_true")
    a = ap.parse_args()
    H.PNG = a.png
    H.setup()

    keys = ["controller", "scenario", "realisation", "variant", "date"]
    d = pd.read_csv(a.csv).drop_duplicates(subset=keys)
    agg = summarise(d)
    # slew-limited rules whose limit was chosen to match a SAC policy's degradation level (a fair reference)
    for name in ("grid_slew_matched.csv", "grid_slew_matched2.csv"):
        f = Path(a.csv).parent / name
        if f.exists():
            agg = pd.concat([agg, summarise(pd.read_csv(f).drop_duplicates(subset=keys))], ignore_index=True)

    def series(c):
        m = re.match(r"e5_w2-([\d.]+)_s\d+", c)
        return f"SAC w2={float(m[1]):g}" if m else c
    agg["series"] = agg.controller.map(series)
    # mean over seeds within a series (rows are controller x scenario x realisation x variant)
    g = agg.groupby(["series", "scenario", "realisation", "variant"]).agg(
        life_yr=("life_yr", "mean"), h2=("h2_kg_day", "mean"), dv=("dv_uv_day", "mean"), trans=("transitions_day", "mean")).reset_index()

    coef = g[(g.scenario == "original") & (g.realisation == 0)].copy()
    coef["kind"] = "coefficient variant"
    coef["key"] = coef.variant
    sc = g[g.variant == "nominal"].groupby(["series", "scenario"]).agg(
        life_yr=("life_yr", "mean"), h2=("h2", "mean"), dv=("dv", "mean"), trans=("trans", "mean")).reset_index()
    sc["kind"] = "resource scenario"
    sc["key"] = sc.scenario
    out = pd.concat([coef[["series", "kind", "key", "life_yr", "h2", "dv", "trans"]],
                     sc[["series", "kind", "key", "life_yr", "h2", "dv", "trans"]]], ignore_index=True)
    out.to_csv(Path(a.out_dir) / "sac_sensitivity.csv", index=False)

    ref = "ramp_limited"
    print("== SAC vs tuned slew limit: rows where SAC outlives the rule (projected life), with hydrogen cost ==")
    rows = []
    for s in [x for x in out.series.unique() if x.startswith("SAC")]:
        for kind, keys in (("coefficient variant", VARIANTS), ("resource scenario", SCEN)):
            t = out[(out.kind == kind)].pivot(index="key", columns="series", values="life_yr").loc[keys]
            h = out[(out.kind == kind)].pivot(index="key", columns="series", values="h2").loc[keys]
            win = int((t[s] > t[ref]).sum())
            ratio = (t[s] / t[ref])
            rows.append(dict(series=s, kind=kind, rows=len(keys), sac_outlives_rule=win,
                             life_ratio_min=ratio.min(), life_ratio_median=ratio.median(), life_ratio_max=ratio.max(),
                             h2_vs_rule_pct_mean=float(((h[s] - h[ref]) / h[ref] * 100).mean())))
    summ = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(summ.round(2).to_string(index=False))
    summ.to_csv(Path(a.out_dir) / "sac_sensitivity_summary.csv", index=False)
    print("\n== matched comparison: SAC against slew-limited rules at similar nominal degradation ==")
    nom = out[(out.kind == "coefficient variant") & (out.key == "nominal")].set_index("series")
    mrows = []
    for s, r in (("SAC w2=12.9", "slew_0.0015"), ("SAC w2=12.9", "slew_0.001"),
                 ("SAC w2=20", "slew_0.001"), ("SAC w2=20", "slew_0.0005")):
        if r not in out.series.unique():
            continue
        for kind, keys_ in (("coefficient variant", VARIANTS), ("resource scenario", SCEN)):
            t = out[out.kind == kind].pivot(index="key", columns="series", values="life_yr").loc[keys_]
            h = out[out.kind == kind].pivot(index="key", columns="series", values="h2").loc[keys_]
            mrows.append(dict(sac=s, rule=r, kind=kind, rows=len(keys_), sac_outlives=int((t[s] > t[r]).sum()),
                              life_ratio_min=(t[s] / t[r]).min(), life_ratio_median=(t[s] / t[r]).median(),
                              h2_vs_rule_pct=float(((h[s] - h[r]) / h[r] * 100).mean()),
                              nominal_sac_h2=nom.loc[s, "h2"], nominal_sac_dv=nom.loc[s, "dv"],
                              nominal_rule_h2=nom.loc[r, "h2"], nominal_rule_dv=nom.loc[r, "dv"]))
    mt = pd.DataFrame(mrows)
    if len(mt):
        print(mt.round(2).to_string(index=False))
        mt.to_csv(Path(a.out_dir) / "sac_matched_summary.csv", index=False)
    loses = coef[(coef.series == "SAC w2=12.9")].set_index("key").life_yr < coef[coef.series == ref].set_index("key").life_yr
    print("\ncoefficient variants where SAC w2=12.9 does NOT outlive the tuned slew limit:",
          [k for k, v in loses.items() if v] or "none")

    # ---- figure
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 3.4), gridspec_kw={"width_ratios": [1.15, 1]})
    allx = []
    for a_, kind, keys, title in ((ax[0], "coefficient variant", VARIANTS, "(a) Degradation coefficients"),
                                  (ax[1], "resource scenario", SCEN, "(b) Renewable variability")):
        ys = np.arange(len(keys))[::-1]
        dodge = {s[0]: o for s, o in zip(SHOW, (0.27, 0.09, -0.09, -0.27))}
        t = out[out.kind == kind].pivot(index="key", columns="series", values="life_yr")
        for y, k in zip(ys, keys):
            vals = []
            for name, col, mk, lab, hollow in SHOW:
                if name not in t.columns:
                    continue
                v = t.loc[k, name]
                vals.append(v)
                a_.plot(v, y + dodge[name], marker=mk, ms=5.2, color=col, mfc=H.SURFACE if hollow else col,
                        mec=col if hollow else H.SURFACE, mew=1.2 if hollow else 0.7, ls="", zorder=3,
                        label=lab if y == ys[0] else None)
            a_.plot([min(vals), max(vals)], [y, y], color=H.BASE, lw=0.8, zorder=2)
            allx += vals
        a_.set_yticks(ys, [k.replace("_", " ") for k in keys])
        a_.set_title(title, loc="left", color=H.INK)
        a_.set_xscale("log")
        a_.grid(axis="x", visible=True)
        a_.grid(axis="y", visible=False)
    lo, hi = min(allx) * 0.85, max(allx) * 1.15
    for a_ in ax:
        a_.set_xlim(lo, hi)
        a_.set_xticks([t for t in (1, 2, 3, 5, 10, 20, 40) if lo <= t <= hi])
        a_.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        a_.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        a_.set_xlabel("projected life (years, calibrated scale)")
    h, lab = ax[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, fontsize=7, handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path = Path(a.fig_dir) / "fig_sensitivity_sac.pdf"
    H._save(fig, path)
    print("wrote", path)


if __name__ == "__main__":
    main()
