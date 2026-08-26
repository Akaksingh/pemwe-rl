"""PEM electrolyzer stack model: polarization curve, efficiency, H2 rate.

OWNER: Person A.

STATUS: VALIDATED Day 1. Every constant in configs/default.yaml::stack is either
confirmed against a published PEMWE source or corrected, with the citation left beside
the value. See notes/A_constants.md for the validation table and notes/A_methodology.md
for the prose that goes into the paper.

Amphlett-form cell voltage:

    V_cell(j) = E_rev(T, p) + V_act(j) + V_ohm(j) + V_conc(j) + dV_deg

with the reversible voltage carrying a Nernst pressure correction, because the stack is
operated in DIFFERENTIAL-PRESSURE mode (cathode at pressure_bar, anode near ambient).
That pressure is not decorative: it is the same number that sets H2 crossover below, so
it appears in both the voltage cost and the faradaic loss, as it does physically.

The piece that makes this a real control problem is faradaic_efficiency. H2 crossover
through the membrane is a roughly current-independent leak, so it consumes a LARGER
FRACTION of the product the lower the current density is. Efficiency therefore collapses
at low j, while voltage efficiency collapses at high j. Peak efficiency sits mid-range,
and the setpoint that maximises H2 *rate* is not the one that maximises H2 *per joule*.
That is Gate G1 item 1, and it is the entire reason a controller has something to decide.

MEMBRANE THICKNESS IS THE SHARED PHYSICAL CAUSE. A thinner membrane lowers ohmic loss
(good at high j) and raises crossover (bad at low j). Both are computed from the single
membrane_thickness_cm, so the two ends of the efficiency curve cannot be tuned
independently to flatter the result -- a reviewer can check that the trade-off is real.
"""

from __future__ import annotations

import numpy as np

R_GAS = 8.314462          # J/(mol K)
F_FARADAY = 96485.33      # C/mol
M_H2 = 2.01588e-3         # kg/mol
LHV_VOLTAGE = 1.253       # V, LHV of H2 (241.8 kJ/mol) expressed as a cell voltage
HHV_VOLTAGE = 1.481       # V, HHV of H2 (285.8 kJ/mol)


class StackModel:
    def __init__(self, cfg: dict):
        s = cfg["stack"]
        self.n_cells = s["n_cells"]
        self.area = s["area_cm2"]
        self.j_rated = s["j_rated_a_cm2"]
        self.j_min = s["j_min_a_cm2"]
        self.T = s["temp_k"]
        self.p_cat_bar = s["pressure_bar"]
        self.p_an_bar = s["pressure_anode_bar"]
        self.e_rev_ref = s["e_rev_ref_v"]
        self.e_rev_dt = s["e_rev_dtemp_v_per_k"]
        self.alpha_a = s["alpha_anode"]
        self.j0_a = s["j0_anode_a_cm2"]
        self.alpha_c = s["alpha_cathode"]
        self.j0_c = s["j0_cathode_a_cm2"]
        self.j_lim = s["j_lim_a_cm2"]
        self.conc_coeff = s["conc_coeff_v"]
        self.bop_frac = s["bop_frac_of_rated"]

        # --- membrane: one thickness drives BOTH ohmic loss and crossover -------
        self.delta_mem = s["membrane_thickness_cm"]
        self.sigma_mem = s["membrane_conductivity_s_cm"]
        self.r_contact = s["r_contact_ohm_cm2"]
        self.perm_h2 = s["h2_permeability_mol_cm_s_bar"]

        # area-specific resistance = membrane + everything else in series
        self.asr = self.delta_mem / self.sigma_mem + self.r_contact
        # Fick's law across the membrane, as an equivalent current density:
        #   j_cross = 2F * P_H2 * p_cathode / delta
        self.j_cross = 2.0 * F_FARADAY * self.perm_h2 * self.p_cat_bar / self.delta_mem

        # --- rated point ------------------------------------------------------
        # P_rated is the TOTAL plant draw at j_rated, parasitics included, so that a
        # commanded power fraction of 1.0 reaches exactly j_rated. Defining it as the
        # stack-only power (as the Day-0 stub did) makes power_w(j_rated) = 1.02*P_rated,
        # and the agent can then never reach rated current.
        p_stack_rated = self.n_cells * self.area * self.j_rated * self.v_cell(self.j_rated)
        self.p_rated = float(p_stack_rated / (1.0 - self.bop_frac))
        self.p_bop = float(self.bop_frac * self.p_rated)

        # Minimum power the plant can draw while ON: the stack at j_min PLUS the
        # parasitic floor. Anything less and it cannot run at all. Turning on below this
        # would draw more than the renewable input supplies, breaking the hard cap of
        # DECISIONS.md 3, so `j_from_power` gates on this value and not on stack power.
        self.p_min_on = float(
            self.n_cells * self.area * self.j_min * self.v_cell(self.j_min) + self.p_bop)

        # inversion table for j_from_power -- built once, see that method
        self._build_inversion_table()

    # --- polarization ---------------------------------------------------

    def e_rev(self) -> float:
        """Reversible cell voltage at operating temperature AND pressure.

        Standard linear temperature correction, plus a Nernst term for the cost of
        producing gas against the differential cathode pressure. Liquid water is taken
        at unit activity.
        """
        e_t = self.e_rev_ref + self.e_rev_dt * (self.T - 298.15)
        nernst = (R_GAS * self.T / (2.0 * F_FARADAY)) * np.log(
            self.p_cat_bar * np.sqrt(self.p_an_bar))
        return float(e_t + nernst)

    def v_activation(self, j):
        """Symmetric Butler-Volmer, inverted: eta = (RT / alpha F) asinh(j / 2 j0)."""
        j = np.maximum(j, 1e-9)
        an = (R_GAS * self.T / (self.alpha_a * F_FARADAY)) * np.arcsinh(j / (2 * self.j0_a))
        ca = (R_GAS * self.T / (self.alpha_c * F_FARADAY)) * np.arcsinh(j / (2 * self.j0_c))
        return an + ca

    def tafel_slopes_mv_per_dec(self):
        """Diagnostic: b = 2.303 RT / (alpha F). A validation target, not a model input."""
        k = 2.303 * R_GAS * self.T / F_FARADAY * 1000.0
        return k / self.alpha_a, k / self.alpha_c

    def v_ohmic(self, j):
        return j * self.asr

    def v_concentration(self, j):
        ratio = np.clip(j / self.j_lim, 0.0, 0.999)
        return -self.conc_coeff * np.log(1.0 - ratio)

    def v_cell(self, j, dv_deg_v: float = 0.0):
        """Single-cell voltage at current density j (A/cm^2), including degradation.

        dv_deg_v is the lumped degradation state from DegradationModel, in VOLTS. It
        enters here and nowhere else, which is what makes accumulated degradation cost
        future yield rather than only reward.
        """
        return (self.e_rev() + self.v_activation(j) + self.v_ohmic(j)
                + self.v_concentration(j) + dv_deg_v)

    # --- efficiency and yield -------------------------------------------

    def faradaic_efficiency(self, j):
        """H2 crossover loss: a near-constant leak, so it hurts most at low j.

        This is what makes idling inefficient, and it is the left-hand half of the
        non-monotonic efficiency curve.
        """
        j = np.maximum(j, 1e-9)
        return np.clip(1.0 - self.j_cross / j, 0.0, 1.0)

    def power_w(self, j, dv_deg_v: float = 0.0):
        """Total electrical power draw including balance-of-plant parasitics."""
        stack = self.n_cells * self.area * j * self.v_cell(j, dv_deg_v)
        bop = np.where(j >= self.j_min, self.p_bop, 0.0)
        return stack + bop

    def _build_inversion_table(self, n: int = 4096):
        """Precompute power(j) at zero degradation, once, for the inversion below."""
        self._j_grid = np.linspace(self.j_min, self.j_rated, n)
        stack = self.n_cells * self.area * self._j_grid * self.v_cell(self._j_grid)
        self._p_grid = stack + self.p_bop          # power_w at dv_deg = 0
        self._c_ja = self.n_cells * self.area      # d(power)/d(dv_deg) = n*A*j

    def j_from_power(self, p_w: float, dv_deg_v: float = 0.0) -> float:
        """Invert power -> j.

        Bisection here was the single hottest path in the whole project: 60 iterations per
        environment step, each evaluating the full polarization curve, which put ~83% of
        runtime in this one function and capped the environment at ~4k steps/s. That in
        turn made training env-bound rather than GPU-bound, so faster hardware bought
        nothing.

        Degradation enters power_w linearly --
            power(j, dv) = power(j, 0) + n_cells * area * j * dv
        -- so a single precomputed table of power(j, 0) inverts the whole family. Take an
        interpolated guess, then correct: the dv term is at most ~1e-4 V against a ~1.6 V
        cell, so one fixed-point pass is already at machine precision for our purposes.
        Numerically identical to the bisection to well within the tolerance it converged
        to, and roughly 30x cheaper.
        """
        if p_w < self.p_min_on:
            return 0.0
        if p_w >= self.power_w(self.j_rated, dv_deg_v):
            return self.j_rated
        j = float(np.interp(p_w, self._p_grid, self._j_grid))
        if dv_deg_v:
            for _ in range(2):
                j = float(np.interp(p_w - self._c_ja * j * dv_deg_v,
                                    self._p_grid, self._j_grid))
        return j

    def h2_rate_kg_s(self, j, dv_deg_v: float = 0.0):
        """Faraday's law, derated by faradaic efficiency. Independent of dv_deg."""
        i_total = j * self.area
        n_dot = self.n_cells * i_total * self.faradaic_efficiency(j) / (2 * F_FARADAY)
        return n_dot * M_H2

    def eta_lhv(self, j, dv_deg_v: float = 0.0):
        """SYSTEM LHV efficiency: H2 chemical power out / total electrical power in."""
        p = self.power_w(j, dv_deg_v)
        h2_w = self.h2_rate_kg_s(j, dv_deg_v) / M_H2 * LHV_VOLTAGE * 2 * F_FARADAY
        return np.where(p > 0, h2_w / np.maximum(p, 1e-9), 0.0)

    # --- validation ------------------------------------------------------

    def efficiency_curve(self, n: int = 400):
        """Gate G1 item 1: this must be NON-MONOTONIC, peaking around j = 0.5-0.8.

        Swept over the ON branch only, i.e. strictly above j_min. Starting the sweep AT
        j_min would evaluate `j > j_min` as False and silently drop the BoP parasitic
        from the first point, reporting a low-load efficiency that the plant can never
        actually achieve.
        """
        j = np.linspace(np.nextafter(self.j_min, self.j_rated), self.j_rated, n)
        return j, self.v_cell(j), self.eta_lhv(j), self.power_w(j) / self.p_rated

    def summary(self) -> dict:
        """The validation numbers, in one dict, for the Day-1 constants table."""
        j, v, eta, pf = self.efficiency_curve()
        k = int(np.argmax(eta))
        b_a, b_c = self.tafel_slopes_mv_per_dec()
        return {
            "asr_ohm_cm2": self.asr,
            "j_crossover_a_cm2": self.j_cross,
            "e_rev_v": self.e_rev(),
            "tafel_anode_mv_dec": b_a,
            "tafel_cathode_mv_dec": b_c,
            "v_cell_rated_v": float(self.v_cell(self.j_rated)),
            "p_rated_w": self.p_rated,
            "p_bop_w": self.p_bop,
            "eta_peak": float(eta[k]),
            "j_at_peak": float(j[k]),
            "peak_frac_of_rated": float(j[k] / self.j_rated),
            "eta_at_rated": float(eta[-1]),
            "eta_at_min": float(eta[0]),
        }
