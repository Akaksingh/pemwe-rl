"""PEM electrolyzer stack model: polarization curve, efficiency, H2 rate.

OWNER: Person A.

STATUS: first-draft physics, NOT yet validated. Day 1 job is to check every constant
against the literature and confirm the shape of the curves. Day 2 gate G1 item 1 is
proved from `efficiency_curve()` below.

Amphlett-form cell voltage:
    V_cell(j) = E_rev(T) + V_act(j) + V_ohm(j) + V_conc(j) + dV_deg

The piece that makes this a real control problem is `faradaic_efficiency`: H2 crossover
through the membrane means efficiency COLLAPSES at low current density, while voltage
efficiency collapses at high current density. So peak efficiency sits mid-range, and the
setpoint that maximises H2 *rate* is not the one that maximises H2 *per joule*.
"""

from __future__ import annotations

import numpy as np

R_GAS = 8.314462          # J/(mol K)
F_FARADAY = 96485.33      # C/mol
M_H2 = 2.01588e-3         # kg/mol
LHV_VOLTAGE = 1.253       # V, thermoneutral-equivalent LHV voltage
HHV_VOLTAGE = 1.481       # V


class StackModel:
    def __init__(self, cfg: dict):
        s = cfg["stack"]
        self.n_cells = s["n_cells"]
        self.area = s["area_cm2"]
        self.j_rated = s["j_rated_a_cm2"]
        self.j_min = s["j_min_a_cm2"]
        self.T = s["temp_k"]
        self.e_rev_ref = s["e_rev_ref_v"]
        self.e_rev_dt = s["e_rev_dtemp_v_per_k"]
        self.alpha_a = s["alpha_anode"]
        self.j0_a = s["j0_anode_a_cm2"]
        self.alpha_c = s["alpha_cathode"]
        self.j0_c = s["j0_cathode_a_cm2"]
        self.asr = s["asr_ohm_cm2"]
        self.j_lim = s["j_lim_a_cm2"]
        self.conc_coeff = s["conc_coeff_v"]
        self.j_cross = s["j_crossover_a_cm2"]
        self.bop_frac = s["bop_frac_of_rated"]
        self.p_rated = self.power_w(self.j_rated, dv_deg_v=0.0)

    # --- polarization ---------------------------------------------------

    def e_rev(self) -> float:
        return self.e_rev_ref + self.e_rev_dt * (self.T - 298.15)

    def v_activation(self, j: np.ndarray | float) -> np.ndarray | float:
        j = np.maximum(j, 1e-9)
        an = (R_GAS * self.T / (self.alpha_a * F_FARADAY)) * np.arcsinh(j / (2 * self.j0_a))
        ca = (R_GAS * self.T / (self.alpha_c * F_FARADAY)) * np.arcsinh(j / (2 * self.j0_c))
        return an + ca

    def v_ohmic(self, j):
        return j * self.asr

    def v_concentration(self, j):
        ratio = np.clip(j / self.j_lim, 0.0, 0.999)
        return -self.conc_coeff * np.log(1.0 - ratio)

    def v_cell(self, j, dv_deg_v: float = 0.0):
        """Single-cell voltage at current density j (A/cm^2)."""
        return (self.e_rev() + self.v_activation(j) + self.v_ohmic(j)
                + self.v_concentration(j) + dv_deg_v)

    # --- efficiency and yield -------------------------------------------

    def faradaic_efficiency(self, j):
        """H2 crossover loss. Dominant at low j -- this is what makes idling inefficient."""
        j = np.maximum(j, 1e-9)
        return np.clip(1.0 - self.j_cross / j, 0.0, 1.0)

    def power_w(self, j, dv_deg_v: float = 0.0):
        """Total electrical power draw incl. balance-of-plant parasitics."""
        stack = self.n_cells * self.area * j * self.v_cell(j, dv_deg_v)
        bop = np.where(j > self.j_min, self.bop_frac * getattr(self, "p_rated", 0.0), 0.0)
        return stack + bop

    def j_from_power(self, p_w: float, dv_deg_v: float = 0.0) -> float:
        """Invert power->j by bisection. Monotonic in j, so this is safe."""
        if p_w <= self.power_w(self.j_min, dv_deg_v):
            return 0.0
        lo, hi = self.j_min, self.j_rated
        if p_w >= self.power_w(hi, dv_deg_v):
            return hi
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if self.power_w(mid, dv_deg_v) < p_w:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def h2_rate_kg_s(self, j, dv_deg_v: float = 0.0):
        """Faraday's law, derated by faradaic efficiency."""
        i_total = j * self.area
        n_dot = self.n_cells * i_total * self.faradaic_efficiency(j) / (2 * F_FARADAY)
        return n_dot * M_H2

    def eta_lhv(self, j, dv_deg_v: float = 0.0):
        p = self.power_w(j, dv_deg_v)
        if np.all(p <= 0):
            return 0.0
        h2_w = self.h2_rate_kg_s(j, dv_deg_v) / M_H2 * LHV_VOLTAGE * 2 * F_FARADAY
        return np.where(p > 0, h2_w / np.maximum(p, 1e-9), 0.0)

    # --- validation ------------------------------------------------------

    def efficiency_curve(self, n: int = 200):
        """Gate G1 item 1: this must be NON-MONOTONIC, peaking around j=0.5-0.8."""
        j = np.linspace(self.j_min, self.j_rated, n)
        return j, self.v_cell(j), self.eta_lhv(j), self.power_w(j) / self.p_rated
