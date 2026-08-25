"""Plant-model validation. OWNER: Person A. Brief item 2, Gate G1 item 1.

Produces two things:

  1. results/figures/fig_validation.pdf  -- VECTOR, IEEE two-column width. This is the
     Methodology plant-model figure and the Day-2 gate evidence. Hand it to Person C.
     Panel (a) polarization with the loss breakdown, so a reader can see WHERE the
     voltage goes; panel (b) system LHV efficiency, which must be non-monotonic.

  2. The constants validation table on stdout, i.e. the answer to "where did this number
     come from?". Mirrored into notes/A_constants.md.

    python scripts/validate_physics.py

This does NOT live in pemwe.plots -- that module is Person C's. A hands over the PDF.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pemwe import load_config, StackModel

OUT = ROOT / "results" / "figures"

# --- the acceptance bands, all from DECISIONS.md / briefs/PERSON_A.md ----------------
CHECKS = [
    ("V_cell at rated",        "v_cell_rated_v",      1.75, 1.85,  "V"),
    ("peak eta_LHV",           "eta_peak",            0.65, 0.75,  "-"),
    ("j at peak eta",          "j_at_peak",           0.50, 0.80,  "A/cm2"),
    ("peak as frac of rated",  "peak_frac_of_rated",  0.25, 0.40,  "-"),
    ("eta_LHV at rated",       "eta_at_rated",        0.65, 0.70,  "-"),
    ("P_rated",                "p_rated_w",           0.9e6, 1.2e6, "W"),
    ("Tafel slope, anode",     "tafel_anode_mv_dec",  40.0, 60.0,  "mV/dec"),
    ("Tafel slope, cathode",   "tafel_cathode_mv_dec", 25.0, 40.0, "mV/dec"),
    ("ASR",                    "asr_ohm_cm2",         0.05, 0.25,  "ohm cm2"),
    ("j_crossover",            "j_crossover_a_cm2",   0.005, 0.05, "A/cm2"),
]


def figure(stack: StackModel, path: Path):
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
        "pdf.fonttype": 42, "savefig.bbox": "tight",
    })
    j, v, eta, pf = stack.efficiency_curve(600)
    e_rev = np.full_like(j, stack.e_rev())
    v_act = stack.v_activation(j)
    v_ohm = stack.v_ohmic(j)
    v_con = stack.v_concentration(j)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))

    # (a) polarization, with the loss budget stacked underneath the curve
    ax1.fill_between(j, 0, e_rev, color="#c9d6e8", lw=0, label="$E_{rev}(T,p)$")
    ax1.fill_between(j, e_rev, e_rev + v_act, color="#f2c9a0", lw=0, label="activation")
    ax1.fill_between(j, e_rev + v_act, e_rev + v_act + v_ohm, color="#bcd9bc", lw=0,
                     label="ohmic")
    ax1.fill_between(j, e_rev + v_act + v_ohm, v, color="#e3bcd4", lw=0,
                     label="concentration")
    ax1.plot(j, v, color="k", lw=1.2, label="$V_{cell}$")
    ax1.axvline(stack.j_rated, color="0.35", ls="--", lw=0.7)
    ax1.annotate(f"rated\n{float(stack.v_cell(stack.j_rated)):.3f} V",
                 xy=(stack.j_rated, float(stack.v_cell(stack.j_rated))),
                 xytext=(-4, -26), textcoords="offset points", ha="right", fontsize=7)
    ax1.set_xlabel("current density $j$  (A cm$^{-2}$)")
    ax1.set_ylabel("cell voltage  (V)")
    ax1.set_xlim(0, stack.j_rated)
    ax1.set_ylim(1.0, 2.0)
    ax1.legend(loc="upper left", frameon=False, ncol=1, handlelength=1.2)
    ax1.set_title("(a) polarization and loss budget", fontsize=8, loc="left")

    # (b) system efficiency -- the non-monotonic curve, plotted against power fraction
    k = int(np.argmax(eta))
    ax2.plot(pf * 100, eta * 100, color="#1f4e79", lw=1.4)
    ax2.plot(pf[k] * 100, eta[k] * 100, "o", ms=4, color="#c0392b", zorder=5)
    ax2.annotate(f"peak {eta[k]*100:.1f}%\n$j$ = {j[k]:.2f} A cm$^{{-2}}$\n"
                 f"({pf[k]*100:.0f}% of rated power)",
                 xy=(pf[k] * 100, eta[k] * 100), xytext=(10, -34),
                 textcoords="offset points", fontsize=7,
                 arrowprops=dict(arrowstyle="-", lw=0.6, color="#c0392b"))
    ax2.annotate(f"{eta[-1]*100:.1f}% at rated", xy=(100, eta[-1] * 100),
                 xytext=(-6, 10), textcoords="offset points", ha="right", fontsize=7)
    ax2.annotate("H$_2$ crossover\ndominates", xy=(pf[0] * 100 + 1, eta[0] * 100 + 2),
                 fontsize=7, color="0.35")
    ax2.set_xlabel("plant power  (% of rated)")
    ax2.set_ylabel("system efficiency $\\eta_{LHV}$  (%)")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(25, 78)
    ax2.set_title("(b) efficiency is non-monotonic", fontsize=8, loc="left")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main():
    cfg = load_config()
    stack = StackModel(cfg)
    s = stack.summary()

    print("=" * 78)
    print("PLANT MODEL VALIDATION -- Person A, Gate G1 item 1")
    print("=" * 78)
    print(f"\nstack: {cfg['stack']['n_cells']} cells x {cfg['stack']['area_cm2']:.0f} cm2, "
          f"j_rated = {stack.j_rated} A/cm2, T = {stack.T - 273.15:.0f} C, "
          f"{stack.p_cat_bar:.0f} bar cathode / {stack.p_an_bar:.0f} bar anode")
    print(f"derived: ASR = {stack.asr:.4f} ohm cm2 "
          f"(= {stack.delta_mem*1e4:.0f} um / {stack.sigma_mem} S/cm + "
          f"{stack.r_contact} contact)")
    print(f"derived: j_crossover = {stack.j_cross:.5f} A/cm2 "
          f"(Fick: 2F x {stack.perm_h2:g} x {stack.p_cat_bar:.0f} bar / {stack.delta_mem} cm)")
    print(f"derived: P_rated = {stack.p_rated/1e3:.1f} kW total, of which "
          f"{stack.p_bop/1e3:.1f} kW is the BoP floor")
    print(f"derived: minimum ON power = {stack.p_min_on/1e3:.1f} kW "
          f"({stack.p_min_on/stack.p_rated*100:.1f}% of rated)")

    print(f"\n{'quantity':<26}{'value':>12}   {'acceptance band':<20}{'verdict'}")
    print("-" * 78)
    allok = True
    for name, key, lo, hi, unit in CHECKS:
        val = s[key]
        ok = lo <= val <= hi
        allok &= ok
        vs = f"{val:,.4g}"
        print(f"{name:<26}{vs:>12}   [{lo:g}, {hi:g}] {unit:<9}{'PASS' if ok else 'FAIL'}")
    print("-" * 78)
    print(f"{'ALL PLANT-MODEL CHECKS':<26}{'':>12}   {'':<20}{'PASS' if allok else 'FAIL'}")

    j, v, eta, pf = stack.efficiency_curve(600)
    print("\npolarization samples (compare by eye against a published PEMWE curve at 60 C):")
    print(f"  {'j':>6}  {'V_cell':>8}  {'eta_LHV':>8}  {'P/P_rated':>10}")
    for jj in [0.05, 0.1, 0.2, 0.4, 0.73, 1.0, 1.5, 2.0]:
        i = int(np.argmin(abs(j - jj)))
        print(f"  {j[i]:6.2f}  {v[i]:8.4f}  {eta[i]*100:7.2f}%  {pf[i]*100:9.1f}%")

    out = OUT / "fig_validation.pdf"
    figure(stack, out)
    print(f"\nwrote {out}  (vector PDF, 7.0 x 2.7 in, IEEE two-column width)")
    print("-> hand this to Person C as the Methodology plant-model figure")


if __name__ == "__main__":
    main()
