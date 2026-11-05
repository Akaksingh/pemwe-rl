"""Figure and table for the renewable-variability scenarios (Experiment B inputs).

    python scripts/analyze_scenarios.py [--csv results_hpc/renewable_scenarios.csv] [--fig-dir results/figures] [--png]

Reads the metrics written by scripts/make_scenarios.py. Bars are the mean of the independent noise
realisations; dots are the individual realisations, so realisation-to-realisation spread is visible
rather than hidden in an error bar. The `original` scenario is the shipped profile (realisation 0
reproduces it exactly); no measured one-minute data exists to validate any scenario.
"""
from __future__ import annotations

import argparse
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

import analyze_hpc as H                                 # noqa: E402  (shared figure style)

ORDER = ["low_variability", "original", "high_solar", "high_wind", "combined_high"]
LABELS = ["low\nvariability", "original", "high\nsolar", "high\nwind", "combined\nhigh"]
PANELS = [("ramp1_p99", 100.0, "(a) 1-min change, 99th pct", "% of 1 MW", "{:.1f}"),
          ("major_events_per_day", 1.0, "(b) major fluctuations", "events / day", "{:.0f}"),
          ("low_intervals_per_day", 1.0, "(c) low-power intervals", "intervals / day", "{:.1f}")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "results_hpc" / "renewable_scenarios.csv"))
    ap.add_argument("--fig-dir", default=str(ROOT / "results" / "figures"))
    ap.add_argument("--png", action="store_true")
    a = ap.parse_args()
    H.PNG = a.png
    H.setup()

    d = pd.read_csv(a.csv)
    d = d[d.scenario != "shipped_file"]
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.6))
    x = np.arange(len(ORDER))
    for a_, (col, k, title, ylab, fmt) in zip(ax, PANELS):
        means, pts = [], []
        for s in ORDER:
            v = d[d.scenario == s][col].to_numpy() * k
            means.append(v.mean())
            pts.append(v)
        a_.bar(x, means, width=0.58, color=H.BLUE, zorder=3)
        for xi, v in zip(x, pts):
            a_.scatter(np.full(len(v), xi), v, s=11, color=H.INK, zorder=4, linewidths=0)
        for xi, m in zip(x, means):
            a_.text(xi, m * 1.04 + 0.01 * max(means), fmt.format(m), ha="center", va="bottom", fontsize=7, color=H.INK2)
        a_.set_xticks(x, LABELS)
        a_.tick_params(axis="x", labelsize=6.3)
        a_.set_ylabel(ylab)
        a_.set_title(title, loc="left", color=H.INK, fontsize=8)
        a_.set_ylim(0, max(means) * 1.18)
    fig.tight_layout()
    out = Path(a.fig_dir) / "fig_scenarios.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    H._save(fig, out)

    t = (d.groupby("scenario")[["cf_of_1MW", "ramp1_p99", "ramp15_p95", "major_events_per_day", "cv_within_hour_mean",
                                "frac_time_below_pmin", "low_intervals_per_day", "low_mean_duration_min",
                                "acf_fluct_lag5"]].mean().loc[ORDER])
    pd.set_option("display.width", 220)
    print(t.round(3).to_string())
    print("wrote", out)


if __name__ == "__main__":
    main()
