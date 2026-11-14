"""Turn raw HPC benchmark CSVs into summary tables and paper figures.

    python scripts/analyze_hpc.py [--dir results_hpc] [--fig-dir results/figures] [--seed 0]

Inputs  (all written by the benchmark scripts; nothing is typed in by hand)
    hpc_env.csv        scripts/bench_hpc_env.py     simulator strong/weak scaling, SB3 vec-env
    hpc_learner*.csv   first-pass learner benchmark (GPU batch sweep, CPU threads)
    hpc_multi.csv      scripts/bench_hpc_multi.py   aggregate learner throughput, k concurrent runs
Outputs
    hpc_summary.csv    speedup S(n)=T(1)/T(n), efficiency E(n)=S(n)/n, throughput, with 95% intervals
    fig_hpc_sim.pdf    simulator speedup and parallel efficiency
    fig_hpc_learner.pdf  learner throughput: batch size, device/threads, concurrency

Statistics: every case has a handful of repeats (default 5, not independent samples of a
population -- repeats of one machine state). Intervals are bootstrap percentile intervals over
repeats and are indicative only; they say nothing about other machines.
Strong-scaling speedup uses fixed total work, so S(n) = T(1)/T(n) = throughput(n)/throughput(1).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import numpy as np                                      # noqa: E402
import pandas as pd                                     # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PNG = False

# ---- palette: dataviz reference instance, categorical slots 1-3 (documented as validated for
# all pairs in light mode), light surface, plus ink/grid tokens. Marker shape and line style are
# a second encoding so figures survive grayscale printing and colour-vision differences.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#ffffff"

STYLE = {  # series -> (colour, marker, linestyle)
    "unpinned":  (BLUE, "o", "-"),
    "physical":  (ORANGE, "s", "--"),
    "smt_pairs": (AQUA, "^", ":"),
}
LABEL = {"unpinned": "OS-placed", "physical": "one worker per physical core", "smt_pairs": "packed on hyper-thread pairs"}


def setup():
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.grid.axis": "y",
        "pdf.fonttype": 42, "ps.fonttype": 42, "lines.solid_capstyle": "round",
    })


def boot_ci(x, f=np.mean, n=4000, rng=None, alpha=0.05):
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(f(x)), np.nan, np.nan
    b = np.array([f(rng.choice(x, len(x), replace=True)) for _ in range(n)])
    return float(f(x)), float(np.quantile(b, alpha / 2)), float(np.quantile(b, 1 - alpha / 2))


def boot_ratio_ci(num, den, n=4000, rng=None, alpha=0.05):
    """CI of mean(num)/mean(den), resampling each set of repeats independently."""
    rng = rng or np.random.default_rng(0)
    num, den = np.asarray(num, float), np.asarray(den, float)
    est = num.mean() / den.mean()
    if len(num) < 2 or len(den) < 2:
        return est, np.nan, np.nan
    r = np.array([rng.choice(num, len(num)).mean() / rng.choice(den, len(den)).mean() for _ in range(n)])
    return float(est), float(np.quantile(r, alpha / 2)), float(np.quantile(r, 1 - alpha / 2))


def scaling_table(env: pd.DataFrame, rng) -> pd.DataFrame:
    rows = []
    for (exp, backend), g in env[env.experiment.str.startswith("raw_env")].groupby(["experiment", "backend"]):
        # n = 1 reference: the same placement's own single worker when it exists (so each series
        # starts at S = 1); otherwise (SMT-pair placement has no single-worker case) the
        # OS-placed single worker of the same experiment
        own = env[(env.experiment == exp) & (env.backend == backend) & (env.n_workers == 1)].steps_per_s
        ref = own if len(own) else env[(env.experiment == exp) & (env.backend == "unpinned")
                                       & (env.n_workers == 1)].steps_per_s
        for n, h in g.groupby("n_workers"):
            tp = h.steps_per_s.to_numpy()
            m, lo, hi = boot_ci(tp, rng=rng)
            if exp.endswith("strong_scaling") and len(ref):
                s, slo, shi = boot_ratio_ci(tp, ref.to_numpy(), rng=rng)
            else:
                s = slo = shi = np.nan
            rows.append(dict(experiment=exp, placement=backend, n_workers=int(n), repeats=len(tp),
                             steps_per_s=m, steps_per_s_lo=lo, steps_per_s_hi=hi,
                             speedup=s, speedup_lo=slo, speedup_hi=shi,
                             efficiency=s / n if n else np.nan, efficiency_lo=slo / n, efficiency_hi=shi / n,
                             per_worker_steps_per_s=m / n,
                             sim_days_per_wall_s=m * 60.0 / 86400.0))
    return pd.DataFrame(rows).sort_values(["experiment", "placement", "n_workers"])


def fig_sim(tab: pd.DataFrame, path: Path):
    st = tab[tab.experiment == "raw_env_strong_scaling"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ns = np.array([1, 2, 4, 8, 12, 16])
    for k, a in enumerate(ax):
        a.set_xlim(0, 17.5)
        a.set_xticks([1, 2, 4, 8, 12, 16])
        a.set_xlabel("Worker processes")
        a.axvline(8, color=BASE, lw=0.9, ls=(0, (2, 2)), zorder=1)
    ax[0].plot([0, 16], [0, 16], color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax[0].text(11.3, 14.2, "ideal", color=MUTED, fontsize=7.5)
    ax[1].axhline(1.0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ends = {}
    for pl, (col, mk, ls) in STYLE.items():
        d = st[st.placement == pl]
        if d.empty:
            continue
        for a, y, lo, hi in ((ax[0], "speedup", "speedup_lo", "speedup_hi"),
                             (ax[1], "efficiency", "efficiency_lo", "efficiency_hi")):
            yerr = np.vstack([(d[y] - d[lo]).clip(lower=0), (d[hi] - d[y]).clip(lower=0)])
            a.errorbar(d.n_workers, d[y], yerr=yerr, color=col, marker=mk, ms=5.5, lw=1.5, ls=ls,
                       mec=SURFACE, mew=0.8, capsize=2, elinewidth=0.9, label=LABEL[pl], zorder=3)
        ends[pl] = d.iloc[-1]
    ax[0].set_ylabel("Speedup, $S(n)=T(1)/T(n)$")
    ax[1].set_ylabel("Parallel efficiency, $E(n)=S(n)/n$")
    ax[0].set_ylim(0, 16.5)
    ax[1].set_ylim(0, 1.15)
    ax[0].set_title("(a) Simulator speedup, fixed work", loc="left", color=INK)
    ax[1].set_title("(b) Parallel efficiency", loc="left", color=INK)
    ax[0].text(8.15, 0.6, "8 physical cores", color=MUTED, fontsize=7)
    ax[0].legend(frameon=False, loc="upper left", handlelength=2.6)
    fig.tight_layout()
    _save(fig, path)


def _save(fig, path: Path):
    fig.savefig(path)
    if PNG:
        fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def fig_learner(learn: pd.DataFrame, multi: pd.DataFrame | None, path: Path, rng):
    panels = 3 if multi is not None and len(multi) else 2
    fig, ax = plt.subplots(1, panels, figsize=(7.0 if panels == 2 else 7.16, 2.7),
                           gridspec_kw={"width_ratios": [1, 1, 1.6][:panels]})
    # (a) GPU updates/s vs batch size
    g = learn[(learn.backend == "cuda")].copy()
    g["batch"] = g.note.str.extract(r"batch=(\d+)").astype(int)
    xs = sorted(g.batch.unique())
    m = [boot_ci(g[g.batch == b].steps_per_s, rng=rng) for b in xs]
    ax[0].errorbar(range(len(xs)), [t[0] for t in m],
                   yerr=[[t[0] - t[1] for t in m], [t[2] - t[0] for t in m]],
                   color=BLUE, marker="o", ms=5.5, lw=1.5, mec=SURFACE, mew=0.8, capsize=2, elinewidth=0.9)
    ax[0].set_xticks(range(len(xs)), [str(x) for x in xs])
    ax[0].set_xlabel("Batch size (samples per update)")
    ax[0].set_ylabel("Learner updates per second")
    ax[0].set_ylim(0, 215)
    ax[0].set_title("(a) GPU: flat in batch size", loc="left", color=INK)
    ax[0].text(0.0, 28, "8 times more samples per update,\nsame update rate", color=INK2, fontsize=7)

    # (b) single learner, device/threads at batch 512
    sel = learn[(learn.note == "batch=512")]
    cats, vals, cols = [], [], []
    for lab, mask, col in (("GPU", sel.backend == "cuda", BLUE),
                           ("CPU 4\nthreads", (sel.backend == "cpu") & (sel.n_workers == 4), ORANGE),
                           ("CPU 8\nthreads", (sel.backend == "cpu") & (sel.n_workers == 8), ORANGE),
                           ("CPU 16\nthreads", (sel.backend == "cpu") & (sel.n_workers == 16), ORANGE)):
        v = sel[mask].steps_per_s
        if len(v):
            cats.append(lab)
            vals.append(boot_ci(v, rng=rng))
            cols.append(col)
    xb = np.arange(len(cats))
    ax[1].bar(xb, [v[0] for v in vals], width=0.5, color=cols, zorder=3)
    ax[1].errorbar(xb, [v[0] for v in vals], yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
                   fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2, zorder=4)
    for x, v in zip(xb, vals):
        ax[1].text(x, v[0] + 6, f"{v[0]:.0f}", ha="center", fontsize=7.5, color=INK2)
    ax[1].set_xticks(xb, cats)
    ax[1].set_ylim(0, 215)
    ax[1].set_ylabel("Learner updates per second")
    ax[1].set_title("(b) One learner, batch 512", loc="left", color=INK)

    # (c) aggregate throughput vs concurrency
    if panels == 3:
        order = ["gpu_x1", "gpu_x2", "gpu_x4", "cpu4t_x1", "cpu4t_x2", "cpu4t_x4", "cpu2t_x8", "mixed_gpu2_cpu4t_x3"]
        nice = {"gpu_x1": "GPU\n×1", "gpu_x2": "GPU\n×2", "gpu_x4": "GPU\n×4", "cpu4t_x1": "CPU\n×1",
                "cpu4t_x2": "CPU\n×2", "cpu4t_x4": "CPU\n×4", "cpu2t_x8": "CPU\n×8", "mixed_gpu2_cpu4t_x3": "2 GPU\n+3 CPU"}
        colr = {"gpu": BLUE, "cpu": ORANGE, "mix": AQUA}
        xs2, ys, cs, labs = [], [], [], []
        for i, c in enumerate([c for c in order if c in set(multi.case)]):
            v = multi[multi.case == c].aggregate_updates_per_s
            ys.append(boot_ci(v, rng=rng))
            cs.append(colr["mix" if c.startswith("mixed") else ("gpu" if c.startswith("gpu") else "cpu")])
            labs.append(nice[c])
            xs2.append(i)
        ax[2].bar(xs2, [y[0] for y in ys], width=0.55, color=cs, zorder=3)
        ax[2].errorbar(xs2, [y[0] for y in ys],
                       yerr=[[y[0] - y[1] for y in ys], [y[2] - y[0] for y in ys]],
                       fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2, zorder=4)
        for x, y in zip(xs2, ys):
            ax[2].text(x, y[0] + 8, f"{y[0]:.0f}", ha="center", fontsize=7, color=INK2)
        ax[2].set_xticks(xs2, labs)
        ax[2].tick_params(axis="x", labelsize=6.5)
        ax[2].set_ylabel("Aggregate updates per second")
        ax[2].set_title("(c) Concurrent runs, total rate", loc="left", color=INK)
    fig.tight_layout()
    _save(fig, path)


def diag_table(diag: pd.DataFrame, rng) -> pd.DataFrame:
    rows = []
    for (mode, n), g in diag.groupby(["mode", "n_workers"]):
        m, lo, hi = boot_ci(g.steps_per_s_worker.to_numpy(), rng=rng)
        f = g.freq_mhz.mean()
        t = g.groupby("repeat").elapsed_s.agg(["max", "min"])
        rows.append(dict(mode=mode, n_workers=int(n), per_worker_steps_per_s=m, lo=lo, hi=hi,
                         mean_freq_mhz=f, steps_per_ghz_s=m / (f / 1000.0),
                         slow_over_fast=float((t["max"] / t["min"]).mean())))
    return pd.DataFrame(rows).sort_values(["mode", "n_workers"])


def fig_diag(tab: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.5))
    sty = {"different_work": (BLUE, "o", "-", "different days per worker"),
           "same_work": (ORANGE, "s", "--", "identical work per worker")}
    cols = [("per_worker_steps_per_s", "steps / s per worker", "(a) Per-worker rate", (0, 15500)),
            ("mean_freq_mhz", "core clock (MHz)", "(b) Sampled clock", (0, 5600)),
            ("steps_per_ghz_s", "steps per GHz·s per worker", "(c) Rate per GHz", (0, 3300))]
    for a_, (c, yl, ti, yl_) in zip(ax, cols):
        for mode, (col, mk, ls, lab) in sty.items():
            d = tab[tab["mode"] == mode]
            a_.plot(d.n_workers, d[c], color=col, marker=mk, ms=5.5, lw=1.5, ls=ls, mec=SURFACE, mew=0.8, label=lab)
        a_.set_xlabel("workers (one per physical core)")
        a_.set_ylabel(yl)
        a_.set_ylim(*yl_)
        a_.set_xticks([1, 2, 3, 4, 5, 6, 8])
        a_.set_title(ti, loc="left", color=INK)
    ax[0].legend(frameon=False, loc="lower left", handlelength=2.6, fontsize=7)
    fig.tight_layout()
    _save(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "results_hpc"))
    ap.add_argument("--fig-dir", default=str(ROOT / "results" / "figures"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--png", action="store_true", help="also write a PNG preview next to each PDF")
    a = ap.parse_args()
    global PNG
    PNG = a.png
    d, fd = Path(a.dir), Path(a.fig_dir)
    fd.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    setup()

    env = pd.read_csv(d / "hpc_env.csv") if (d / "hpc_env.csv").exists() else None
    if env is not None:
        tab = scaling_table(env, rng)
        tab.to_csv(d / "hpc_summary.csv", index=False)
        fig_sim(tab, fd / "fig_hpc_sim.pdf")
        pd.set_option("display.width", 200)
        print(tab[tab.experiment == "raw_env_strong_scaling"][
            ["placement", "n_workers", "repeats", "steps_per_s", "speedup", "speedup_lo", "speedup_hi",
             "efficiency", "sim_days_per_wall_s"]].round(3).to_string(index=False))
    if (d / "hpc_diag.csv").exists():
        dt = diag_table(pd.read_csv(d / "hpc_diag.csv"), rng)
        dt.to_csv(d / "hpc_diag_summary.csv", index=False)
        fig_diag(dt, fd / "fig_hpc_diag.pdf")
        print(dt.round(1).to_string(index=False))
    lp = sorted(d.glob("hpc_learner*.csv"))
    if lp:
        learn = pd.concat([pd.read_csv(p) for p in lp], ignore_index=True)
        multi = pd.read_csv(d / "hpc_multi.csv") if (d / "hpc_multi.csv").exists() else None
        fig_learner(learn, multi, fd / "fig_hpc_learner.pdf", rng)
        print("wrote", fd / "fig_hpc_learner.pdf")


if __name__ == "__main__":
    main()
