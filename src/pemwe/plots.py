"""The six paper figures. OWNER: Person C.

Reads the frozen results schema (CONTRACTS.md section 3) and writes vector PDFs into
results/figures/. Built and debugged against scripts/fake_results.py before any real
result existed -- that is what buys the team its parallelism, and it means nothing here
changes when B's real numbers land, only the values in them.

DESIGN NOTES -- these are print figures, not screen figures
-----------------------------------------------------------
Palette is validated, not chosen by eye: every pair passes CVD separation, the
normal-vision floor, and >= 3:1 contrast on white under an ALL-PAIRS check (not merely
adjacent pairs -- on a Pareto plot every series sits next to every other).

Four fully colour-distinguishable categorical hues do not exist at that bar, so the
series count is cut rather than the standard: the Pareto figure carries THREE identities
(the RL family as one frontier, plus the two baselines), and the SAC-vs-PPO comparison is
faceted into its own figure instead of being overlaid. Cutting series beats weakening the
palette.

Colour is never the only channel. Every series also carries a distinct marker and dash
pattern, so the figures survive greyscale printing and photocopying.

Text wears ink colours, never the series colour. Grid and spines are recessive.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "results" / "figures"
RESULTS = ROOT / "results"

# --- IEEE geometry -----------------------------------------------------------
COL1, COL2 = 3.5, 7.16          # inches: single and double column

# --- validated palette (see module docstring) --------------------------------
C_RL = "#2a78d6"     # RL family / SAC
C_NAIVE = "#c25316"  # baseline_naive (ref [8])
C_RAMP = "#4a3aa7"   # baseline_ramplimited
C_PPO = "#c25316"    # only ever shown against SAC alone, in its own facet

INK = "#111111"
INK2 = "#444444"
MUTED = "#8a8a8a"
GRID = "#d8d8d8"

STYLE = {
    "sac":                  dict(color=C_RL,    marker="o", ls="-",  label="SAC (RL)"),
    "ppo":                  dict(color=C_PPO,   marker="s", ls="--", label="PPO (RL)"),
    "baseline_naive":       dict(color=C_NAIVE, marker="^", ls="--", label="Load-following [8]"),
    "baseline_ramplimited": dict(color=C_RAMP,  marker="D", ls=":",  label="Ramp-limited"),
}


def _rc():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 1.4,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "grid.color": GRID, "axes.grid": True, "grid.alpha": 0.9,
        "axes.axisbelow": True, "legend.frameon": False,
        "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,           # embed real fonts, not Type-3
    })


def _finish(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    out = FIGDIR / f"{name}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")
    return out


def load_results(path: str | Path = RESULTS) -> list[dict]:
    """Every results/<run_id>.json (CONTRACTS.md section 3)."""
    return [json.loads(p.read_text()) for p in sorted(Path(path).glob("*.json"))]


def _agg(rs, policy):
    return [r for r in rs if r["policy"] == policy]


def _mean_std(runs, key):
    """Mean over seeds, and the std ACROSS seeds -- what the paper reports."""
    v = np.array([r["aggregate"][key] for r in runs], dtype=float)
    return v.mean(), v.std()


# =============================================================================
# 1. HEADLINE -- yield vs degradation Pareto frontier
# =============================================================================

def fig_pareto(results, name="fig_pareto"):
    """The paper's headline (DECISIONS.md section 8).

    Deliberately NOT a %-gain bar chart. RL is not expected to beat the baseline on
    hydrogen yield -- the load-following baseline already harvests nearly all available
    energy. Plotted as a frontier, the same data says something stronger and more robust:
    the learned policy reaches operating points the rule-based controller cannot.
    """
    _rc()
    fig, ax = plt.subplots(figsize=(COL1, 2.7))

    rl = _agg(results, "sac")
    by_w2 = {}
    for r in rl:
        by_w2.setdefault(r["weights"]["w2"], []).append(r)

    w2s = sorted(by_w2)
    xs, ys, xe, ye = [], [], [], []
    for w in w2s:
        h, hs = _mean_std(by_w2[w], "dv_deg_uv_mean")
        y, ys_ = _mean_std(by_w2[w], "h2_kg_mean")
        xs.append(h); ys.append(y); xe.append(hs); ye.append(ys_)

    ax.errorbar(xs, ys, xerr=xe, yerr=ye, color=C_RL, marker="o", ms=3.5, lw=1.4,
                capsize=1.5, elinewidth=0.6, zorder=3,
                markeredgecolor="white", markeredgewidth=0.5,
                label=f"SAC frontier ({len(w2s)} $w_2$ values)")

    # annotate only the ends -- never a label on every point
    if xs:
        ax.annotate(f"$w_2$={w2s[0]:g}", (xs[0], ys[0]), textcoords="offset points",
                    xytext=(4, -8), fontsize=6.5, color=INK2)
        ax.annotate(f"$w_2$={w2s[-1]:g}", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(-6, 8), ha="right", fontsize=6.5, color=INK2)

    for pol in ("baseline_naive", "baseline_ramplimited"):
        runs = _agg(results, pol)
        if not runs:
            continue
        st = STYLE[pol]
        x, xs_ = _mean_std(runs, "dv_deg_uv_mean")
        y, ys_ = _mean_std(runs, "h2_kg_mean")
        ax.errorbar([x], [y], xerr=[xs_], yerr=[ys_], color=st["color"],
                    marker=st["marker"], ms=6, lw=0, capsize=1.5, elinewidth=0.6,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=4,
                    label=st["label"])

    ax.set_xlabel(r"Cumulative degradation  $\Delta V_{\mathrm{deg}}$  [$\mu$V/day]")
    ax.set_ylabel(r"H$_2$ yield  [kg/day]")
    ax.legend(loc="lower right", handletextpad=0.5, borderpad=0.2)
    ax.margins(0.12)
    return _finish(fig, name)


# =============================================================================
# 2. Example-day trajectory -- shows WHY
# =============================================================================

def fig_trajectory(rl_result, baseline_result=None, name="fig_trajectory"):
    """One day: renewable input, RL setpoint, baseline setpoint.

    This is the figure that explains the Pareto result rather than asserting it -- the
    reader sees the learned policy declining to chase transients the baseline follows.
    """
    _rc()
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(COL1, 3.4), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1]})

    t = np.asarray(rl_result["trajectory"]["t_min"], dtype=float) / 60.0
    pr = np.asarray(rl_result["trajectory"]["p_renew_w"], dtype=float) / 1e3
    ps = np.asarray(rl_result["trajectory"]["p_set_w"], dtype=float) / 1e3

    ax.fill_between(t, 0, pr, color=MUTED, alpha=0.25, lw=0, zorder=1,
                    label="Available renewable")
    ax.plot(t, ps, color=C_RL, lw=1.4, zorder=3, label="SAC setpoint")
    if baseline_result is not None:
        pb = np.asarray(baseline_result["trajectory"]["p_set_w"], dtype=float) / 1e3
        ax.plot(t, pb, color=C_NAIVE, lw=1.1, ls="--", zorder=2,
                label="Load-following [8]")
    ax.set_ylabel("Power [kW]")
    # legend above the axes: the midday peak fills the upper-left corner, so an inset
    # legend sits on top of the data it is meant to explain
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
              handletextpad=0.4, columnspacing=1.1, borderpad=0.2)

    dv = np.asarray(rl_result["trajectory"]["dv_deg_total_uv"], dtype=float)
    ax2.plot(t, dv, color=C_RL, lw=1.4)
    if baseline_result is not None:
        dvb = np.asarray(baseline_result["trajectory"]["dv_deg_total_uv"], dtype=float)
        ax2.plot(t, dvb, color=C_NAIVE, lw=1.1, ls="--")
    ax2.set_ylabel(r"Cum. $\Delta V_{\mathrm{deg}}$ [$\mu$V]")
    ax2.set_xlabel("Hour of day")
    ax2.set_xlim(0, 24)
    ax2.set_xticks([0, 6, 12, 18, 24])
    return _finish(fig, name)


# =============================================================================
# 3. Training curves -- components separately, never just total reward
# =============================================================================

def fig_training_curves(curves: dict, name="fig_training_curves"):
    """Total reward AND the three components, mean +/- std across seeds.

    Three panels, not one axis with three scales: a dual-axis chart is the single most
    common chart mistake and the components have genuinely different units.

    `curves` maps panel title -> {"steps": [...], "runs": [[...], ...]} with one inner
    list per seed.
    """
    _rc()
    keys = list(curves)
    fig, axes = plt.subplots(1, len(keys), figsize=(COL2, 1.9), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, k in zip(axes, keys):
        steps = np.asarray(curves[k]["steps"], dtype=float)
        runs = np.asarray(curves[k]["runs"], dtype=float)
        mu, sd = runs.mean(axis=0), runs.std(axis=0)
        ax.fill_between(steps / 1e6, mu - sd, mu + sd, color=C_RL, alpha=0.18, lw=0)
        ax.plot(steps / 1e6, mu, color=C_RL, lw=1.4)
        ax.set_title(k, color=INK)
        ax.set_xlabel("Steps [M]")
    axes[0].set_ylabel("Value")
    fig.text(0.995, 0.02, f"mean $\\pm$ 1 s.d. over {runs.shape[0]} seeds",
             ha="right", fontsize=6.5, color=MUTED)
    fig.tight_layout()
    return _finish(fig, name)


# =============================================================================
# 4. Ablation -- reward weight sweep
# =============================================================================

def fig_ablation(results, name="fig_ablation"):
    """How w2 moves the two reward components. Two panels sharing an x axis."""
    _rc()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL2, 2.0), sharex=True)

    rl = _agg(results, "sac")
    by = {}
    for r in rl:
        by.setdefault(r["weights"]["w2"], []).append(r)
    w = sorted(by)

    for ax, key, lab, col in (
            (a1, "h2_kg_mean", r"H$_2$ yield [kg/day]", C_RL),
            (a2, "dv_deg_uv_mean", r"Degradation [$\mu$V/day]", C_RAMP)):
        m = np.array([_mean_std(by[x], key)[0] for x in w])
        s = np.array([_mean_std(by[x], key)[1] for x in w])
        ax.fill_between(w, m - s, m + s, color=col, alpha=0.18, lw=0)
        ax.plot(w, m, color=col, marker="o", ms=3, lw=1.4,
                markeredgecolor="white", markeredgewidth=0.5)
        ax.set_xscale("log")
        ax.set_xlabel(r"Degradation weight  $w_2$")
        ax.set_ylabel(lab)
    fig.tight_layout()
    return _finish(fig, name)


# =============================================================================
# 5. Long-horizon degradation
# =============================================================================

def fig_longhorizon(series: dict, dv_eol_uv=177000.0, name="fig_longhorizon"):
    """90-day cumulative degradation with the end-of-life line.

    `series` maps policy key -> array of cumulative uV, one point per day.
    """
    _rc()
    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    for pol, dv in series.items():
        st = STYLE.get(pol, STYLE["sac"])
        d = np.asarray(dv, dtype=float)
        ax.plot(np.arange(len(d)), d / 1000.0, color=st["color"], ls=st["ls"],
                lw=1.4, label=st["label"])
    ax.axhline(dv_eol_uv / 1000.0, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.annotate("end of life (10% $V$ rise)", (0.02, dv_eol_uv / 1000.0),
                xycoords=("axes fraction", "data"), va="bottom",
                fontsize=6.5, color=MUTED)
    ax.set_xlabel("Day")
    ax.set_ylabel(r"Cumulative $\Delta V_{\mathrm{deg}}$ [mV]")
    ax.legend(loc="upper left", handletextpad=0.5, borderpad=0.2)
    return _finish(fig, name)


# =============================================================================
# 6. Plant model validation (Methodology figure)
# =============================================================================

def fig_validation(env, name="fig_validation"):
    """Polarization and efficiency curves. Gate G1.1 evidence, and the figure that shows
    why the control problem is non-trivial: peak efficiency is mid-range, so maximum H2
    RATE and maximum H2 PER JOULE are at different setpoints."""
    _rc()
    j, v, eta, pf = env.stack.efficiency_curve()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL2, 2.1))

    a1.plot(j, v, color=C_RL, lw=1.6)
    a1.set_xlabel(r"Current density $j$ [A cm$^{-2}$]")
    a1.set_ylabel(r"Cell voltage $V_{\mathrm{cell}}$ [V]")

    a2.plot(pf * 100, np.asarray(eta) * 100, color=C_RAMP, lw=1.6)
    k = int(np.argmax(eta))
    a2.plot([pf[k] * 100], [eta[k] * 100], marker="o", ms=5, color=C_RAMP,
            markeredgecolor="white", markeredgewidth=0.7, zorder=4)
    a2.annotate(f"peak {eta[k]*100:.1f}%\nat {pf[k]*100:.0f}% rated",
                (pf[k] * 100, eta[k] * 100), textcoords="offset points",
                xytext=(6, -14), fontsize=6.5, color=INK2)
    a2.set_xlabel("Power setpoint [% of rated]")
    a2.set_ylabel(r"LHV efficiency $\eta$ [%]")
    fig.tight_layout()
    return _finish(fig, name)


def build_all(results=None):
    """Regenerate every figure. Safe to run against fake results."""
    results = results if results is not None else load_results()
    print(f"building figures from {len(results)} result files")
    outs = []
    if results:
        outs.append(fig_pareto(results))
        outs.append(fig_ablation(results))
        rl = [r for r in results if r["policy"] == "sac" and r.get("trajectory")]
        bl = [r for r in results if r["policy"] == "baseline_naive" and r.get("trajectory")]
        if rl:
            outs.append(fig_trajectory(rl[0], bl[0] if bl else None))
    return outs
