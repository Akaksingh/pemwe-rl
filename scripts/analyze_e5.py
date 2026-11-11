"""Yield-degradation frontier: corrected SAC (E5) against the earlier SAC runs and the rule-based families.

    python scripts/analyze_e5.py [--png]

All sources are evaluated on the same 72 held-out test days with the same environment:
  rules      baseline_frontier.csv: slew-limit grids (with and without the idle threshold) and power-cap
             grid, plus the three named baselines from degradation_sensitivity.csv (nominal)
  legacy     results/sac_w2-*_seed*.json: the earlier SAC runs (update-to-data ratio 0.032, gamma 0.999,
             2M steps), evaluated by pemwe.evaluate
  corrected  E5 runs (update-to-data ratio 1, gamma 0.95, 300k steps) from the grid CSV

The test split is used here once, for the final evaluation of policies whose settings were chosen from
in-sample training-day pilots. The rule grids are not tuned on these days: every grid point is shown.
"Best simple rule" is the non-dominated envelope of all rule points (hydrogen higher AND degradation
lower is better). For each corrected SAC point the envelope's hydrogen at the same degradation is found
by linear interpolation along the envelope; points below the envelope's lowest-degradation point are
reported against that point and flagged.
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


def envelope(points: pd.DataFrame) -> pd.DataFrame:
    """Non-dominated set: no other point has >= hydrogen and <= degradation (one strictly)."""
    keep = []
    for i, r in points.iterrows():
        dom = ((points.h2 >= r.h2) & (points.dv <= r.dv) & ((points.h2 > r.h2) | (points.dv < r.dv))).any()
        if not dom:
            keep.append(i)
    return points.loc[keep].sort_values("dv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e5", default=str(ROOT / "results_e5" / "grid_e5_test.csv"))
    ap.add_argument("--base", default=str(ROOT / "results_hpc" / "degradation_sensitivity.csv"))
    ap.add_argument("--families", default=str(ROOT / "results_hpc" / "baseline_frontier.csv"))
    ap.add_argument("--legacy-dir", default=str(ROOT / "results"))
    ap.add_argument("--fig-dir", default=str(ROOT / "results" / "figures"))
    ap.add_argument("--out-csv", default=str(ROOT / "results_hpc" / "e5_frontier.csv"))
    ap.add_argument("--vs-csv", default=str(ROOT / "results_hpc" / "e5_vs_rules.csv"))
    ap.add_argument("--png", action="store_true")
    a = ap.parse_args()
    H.PNG = a.png
    H.setup()

    b = pd.read_csv(a.base)
    b = b[(b.variant == "nominal") & (b.scenario == "original") & (b.realisation == 0)].set_index("controller")
    fam = pd.read_csv(a.families)
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

    rules = pd.concat([
        pd.DataFrame({"name": [f"{r.family} {r.param:g}" for r in fam.itertuples()], "h2": fam.h2_kg_day, "dv": fam.dv_uv_day}),
        pd.DataFrame({"name": [f"named {c}" for c in b.index], "h2": b.h2_kg_day.values, "dv": b.dv_uv_day.values})],
        ignore_index=True)
    env0 = envelope(rules)                                     # slew-limit / power-cap rules only (earlier envelope)
    rf2 = ROOT / "results_hpc" / "rule_family_v2.csv"          # hysteresis + cap + slew rules (Kaggle run), optional
    if rf2.exists():
        v2 = pd.read_csv(rf2)
        rules = pd.concat([rules, pd.DataFrame({
            "name": [f"hyst p_on={r.p_on} k_up={r.k_up} k_down={r.k_down} slew={r.slew} cap={r.cap}" for r in v2.itertuples()],
            "h2": v2.h2_kg_day, "dv": v2.dv_uv_day})], ignore_index=True)
        print(f"including {len(v2)} hysteresis-rule grid points in the rule envelope")
    env = envelope(rules)

    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    ax.scatter(rules.dv, rules.h2, s=7, color=H.BASE, linewidths=0, zorder=2)
    ax.plot(env.dv, env.h2, color=H.INK2, lw=1.1, ls=(0, (4, 2)), zorder=3, label="best simple rule (envelope)")
    lm = leg.groupby("w2").agg(h2=("h2", "mean"), dv=("dv", "mean")).reset_index().sort_values("w2")
    ax.plot(lm.dv, lm.h2, color=H.AQUA, marker="o", ms=3.5, lw=1.2, mec=H.SURFACE, mew=0.6, zorder=4,
            label="SAC, earlier runs (UTD 0.03, 2M steps)")
    if len(new):
        nm = new.groupby("w2").agg(h2=("h2", "mean"), dv=("dv", "mean")).reset_index().sort_values("w2")
        ax.scatter(new.dv, new.h2, s=9, color=H.BLUE, alpha=0.35, linewidths=0, zorder=4)
        ax.plot(nm.dv, nm.h2, color=H.BLUE, marker="D", ms=5, lw=1.5, mec=H.SURFACE, mew=0.8, zorder=5,
                label="SAC, corrected (UTD 1, 300k steps)")
        for r in nm.itertuples():
            ax.annotate(f"{r.w2:g}", (r.dv, r.h2), textcoords="offset points", xytext=(-3, 6), fontsize=6.3,
                        color=H.INK2, ha="center")
    label_at = {"naive_as_published": (97.0, 178.0), "ramp_limited": (89.0, 162.0), "no_threshold": (85.0, 208.5)}
    for c, (mk, lab) in BASE_STYLE.items():
        if c in b.index:
            ax.plot(b.loc[c, "dv_uv_day"], b.loc[c, "h2_kg_day"], marker=mk, ms=6.5, mfc=H.SURFACE, mec=H.INK, mew=1.2,
                    ls="", zorder=6)
            ax.annotate(lab, (b.loc[c, "dv_uv_day"], b.loc[c, "h2_kg_day"]), xytext=label_at[c], textcoords="data",
                        fontsize=6.5, color=H.INK, ha="left", va="center",
                        arrowprops=dict(arrowstyle="-", color=H.MUTED, lw=0.7, shrinkA=1, shrinkB=4))
    ax.set_xlabel("degradation (µV / day, lower is better)")
    ax.set_ylabel("hydrogen (kg / day)")
    ax.set_xlim(100, 44)
    ax.set_ylim(60, 212)
    ax.grid(axis="x", visible=True)
    ax.legend(frameon=False, loc="lower left", fontsize=6.3, handlelength=2.0)
    fig.tight_layout()
    out = Path(a.fig_dir) / "fig_frontier.pdf"
    try:
        H._save(fig, out)
    except ImportError as e:        # e.g. a locked-down machine that blocks matplotlib's PDF backend
        print(f"WARNING: figure NOT regenerated ({e}); the CSVs below are still written. "
              "Rerun this script on a machine where matplotlib's PDF backend loads.")

    # ---- advantage over the best simple rule at the same degradation
    vs = []
    if len(new):
        for r in nm.itertuples():
            below = r.dv < env.dv.min()
            rule_h2 = float(np.interp(r.dv, env.dv.to_numpy(), env.h2.to_numpy())) if not below else float(env.h2.iloc[0])
            old_h2 = float(np.interp(r.dv, env0.dv.to_numpy(), env0.h2.to_numpy())) if r.dv >= env0.dv.min() else float(env0.h2.iloc[0])
            vs.append(dict(w2=r.w2, sac_h2=r.h2, sac_dv=r.dv, rule_h2_at_same_dv=rule_h2,
                           advantage_kg=r.h2 - rule_h2, advantage_pct=100 * (r.h2 - rule_h2) / rule_h2,
                           advantage_pct_slewcap_only=100 * (r.h2 - old_h2) / old_h2,
                           below_envelope_range=bool(below), n_seeds=int((new.w2 == r.w2).sum())))
        v = pd.DataFrame(vs)
        v.to_csv(a.vs_csv, index=False)
        new.to_csv(a.out_csv, index=False)
        pd.set_option("display.width", 200)
        print("best-simple-rule envelope:")
        print(env.round(1).to_string(index=False))
        print("\nSAC (corrected, mean of seeds) against the envelope at the same degradation:")
        print(v.round(2).to_string(index=False))
        print("\nper seed:")
        print(new.sort_values(["w2", "seed"]).round(2).to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
