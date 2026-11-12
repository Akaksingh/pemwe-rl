"""Analyse the batched-simulator benchmark: throughput against batch size, a Hockney model, and the grid speedup.

    python scripts/analyze_batch.py [--png]

Input : results_batch/bench_batch.csv (+ .meta.json), written by scripts/bench_batch.py on the Kaggle CPU kernel
Output: results/figures/fig_batch.pdf, results_batch/batch_summary.json

Model (Hockney, 1991): one lockstep step over B episodes costs  t(B) = a + b*B,  a fixed per-call overhead plus
work per episode. Then throughput r(B) = B / t(B) = r_inf * B / (B + n_half), with r_inf = 1/b (asymptotic rate) and
n_half = a/b, the batch size at which half of r_inf is reached. The fit is a relative-error weighted line through
(B, seconds per step), so small and large batches count equally.
All timings are same-machine ratios: the scalar environment and the batched simulator ran back to back on one machine.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import numpy as np                                      # noqa: E402
import pandas as pd                                     # noqa: E402

import analyze_hpc as H                                 # noqa: E402

N_STEPS = 1440


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "results_batch" / "bench_batch.csv"))
    ap.add_argument("--fig-dir", default=str(ROOT / "results" / "figures"))
    ap.add_argument("--png", action="store_true")
    a = ap.parse_args()
    H.PNG = a.png
    H.setup()

    d = pd.read_csv(a.csv)
    meta = json.loads(Path(a.csv).with_suffix(".meta.json").read_text())
    one = d[d.method == "scalar_1proc"].iloc[0]
    pool = d[d.method == "grid_scalar_pool"].iloc[0]
    bat = d[d.method == "batched"].sort_values("size")
    grid = d[d.method == "grid_batched"].iloc[0]

    B = bat["size"].to_numpy(float)
    t_step = bat.seconds.to_numpy() / N_STEPS                       # seconds per lockstep step
    b, a0 = np.polyfit(B, t_step, 1, w=1.0 / t_step)                # t = a0 + b*B, relative-error weights
    r_inf, n_half = 1.0 / b, a0 / b
    pred = a0 + b * B
    rel_err = np.abs(pred - t_step) / t_step

    summary = dict(
        platform=meta["platform"], logical_cpus=meta["logical_cpus"], scalar_pool_workers=meta["scalar_pool_workers"],
        real_profiles=meta["real_profiles"], validation_episodes=meta["validation_episodes"],
        validation_max_rel_diff=meta["validation_max_rel_diff"],
        scalar_one_steps_per_s=float(one.steps_per_s), scalar_pool_steps_per_s=float(pool.steps_per_s),
        scalar_pool_seconds=float(pool.seconds), grid_episodes=int(grid.episodes),
        grid_batched_seconds=float(grid.seconds), grid_batched_steps_per_s=float(grid.steps_per_s),
        grid_speedup_vs_pool=float(pool.seconds / grid.seconds),
        speedup_vs_one_process=float(grid.steps_per_s / one.steps_per_s),
        peak_steps_per_s=float(bat.steps_per_s.max()), peak_batch=int(bat.loc[bat.steps_per_s.idxmax(), "size"]),
        hockney_overhead_ms=a0 * 1e3, hockney_us_per_episode_step=b * 1e6, hockney_r_inf=r_inf, hockney_n_half=n_half,
        hockney_max_rel_err=float(rel_err.max()),
        pool_over_one=float(pool.steps_per_s / one.steps_per_s),
    )
    Path(ROOT / "results_batch" / "batch_summary.json").write_text(json.dumps(summary, indent=2))
    for k, v in summary.items():
        print(f"{k:30s} {v}")

    # ---- figure: throughput against batch size, log-log
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    xs = np.logspace(0, np.log10(B.max() * 1.3), 200)
    ax.plot(xs, xs / (a0 + b * xs), color=H.MUTED, lw=1.0, ls=(0, (4, 2)), zorder=2, label="overhead + work model")
    ax.plot(B, B / t_step, color=H.BLUE, marker="o", ms=4.5, lw=0, mec=H.SURFACE, mew=0.7, zorder=4,
            label="batched, one process")
    ax.axhline(one.steps_per_s, color=H.ORANGE, lw=1.3, ls="--", zorder=3)
    ax.axhline(pool.steps_per_s, color=H.AQUA, lw=1.3, ls=":", zorder=3)
    ax.axvline(n_half, color=H.BASE, lw=0.8, zorder=1)
    xr = B.max() * 1.25
    ax.annotate(f"scalar, 1 process: {one.steps_per_s:,.0f}", (xr, one.steps_per_s), xytext=(0, -3),
                textcoords="offset points", fontsize=6.3, color=H.INK2, ha="right", va="top")
    ax.annotate(f"scalar, {int(pool['size'])}-worker pool: {pool.steps_per_s:,.0f}", (xr, pool.steps_per_s),
                xytext=(0, 3), textcoords="offset points", fontsize=6.3, color=H.INK2, ha="right", va="bottom")
    ax.annotate(f"half of peak at\n$B\\approx${n_half:,.0f}", (n_half, r_inf / 20), xytext=(5, 0), textcoords="offset points",
                fontsize=6.3, color=H.INK2, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("episodes per batch, $B$")
    ax.set_ylabel("environment steps / s")
    ax.set_ylim(1e3, 6e6)
    ax.grid(axis="x", visible=True)
    ax.legend(frameon=False, loc="upper left", fontsize=6.3, handlelength=2.0)
    fig.tight_layout()
    out = Path(a.fig_dir) / "fig_batch.pdf"
    H._save(fig, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
