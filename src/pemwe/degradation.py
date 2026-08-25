"""Degradation model. OWNER: Person A.

Five terms, lumped into one scalar `dv_deg_uv` = cumulative cell-voltage rise in microvolts.
That scalar feeds straight back into StackModel.v_cell, so degradation genuinely costs
future yield -- which is what forces the agent to trade short-term output against
long-term capability.

Term-by-term justification and the Day-2 calibration procedure: DECISIONS.md section 5.

The idle term is load-bearing. Delete it and the reward-optimal degenerate policy becomes
"sit at P_idle forever": zero degradation, near-zero yield. It is also physically correct
(anode at open-circuit potential drives Ir dissolution, refs #1/#5).

CALIBRATION HOOK -- `basis()`
----------------------------
The five terms are LINEAR in their five coefficients: the shape parameters
(`j_stress_threshold`, `j_stress_exponent`) are held fixed by DECISIONS.md 5, so

    dv_total(policy) = coeffs . sum_over_steps basis(step)

exactly. `basis()` returns that per-step exposure vector with the coefficients stripped
out. A single rollout of each policy therefore reduces the whole calibration to a small
linear program: solve for the coefficients that put the rule-based baseline at
4.0 uV/h AND the jittery policy at >= 3x the smooth one, instead of hand-tuning against
two targets that pull in opposite directions. See scripts/calibrate_degradation.py.
"""

from __future__ import annotations

import numpy as np

# Order is fixed and shared with the calibration script and configs/default.yaml.
TERMS = ("base", "stress", "ramp", "cycle", "idle")


class DegradationModel:
    def __init__(self, cfg: dict):
        d = cfg["degradation"]
        self.r_base = d["r_base_uv_per_h"]
        self.k_j = d["k_j_uv_per_h"]
        self.j_thr = d["j_stress_threshold"]
        self.j_exp = d["j_stress_exponent"]
        self.k_ramp = d["k_ramp_uv_per_h"]
        self.dv_cycle = d["dv_cycle_uv"]
        self.r_idle = d["r_idle_uv_per_h"]
        self.dv_eol = d["dv_eol_uv"]

    @property
    def coeffs(self) -> np.ndarray:
        """The five scale coefficients, in TERMS order."""
        return np.array([self.r_base, self.k_j, self.k_ramp, self.dv_cycle, self.r_idle],
                        dtype=float)

    def basis(self, j, j_prev, j_rated, dt_min, is_on, was_on) -> np.ndarray:
        """Per-step exposure vector, coefficients stripped. dv = coeffs . basis.

        Depends only on the trajectory and on the FIXED shape parameters, never on the
        coefficients being calibrated. That is what makes the calibration linear.
        """
        dt_h = dt_min / 60.0
        jf = j / j_rated
        return np.array([
            dt_h if is_on else 0.0,                                        # base
            (max(0.0, jf - self.j_thr) ** self.j_exp * dt_h) if is_on else 0.0,  # stress
            abs(j - j_prev) / dt_min * dt_h,                               # ramp
            1.0 if (is_on != was_on) else 0.0,                             # cycle
            0.0 if is_on else dt_h,                                        # idle
        ], dtype=float)

    def step_uv(self, j, j_prev, j_rated, dt_min, is_on, was_on):
        """Degradation added this step, in microvolts. Returns (total, components)."""
        e = self.basis(j, j_prev, j_rated, dt_min, is_on, was_on)
        parts = self.coeffs * e
        return float(parts.sum()), dict(zip(TERMS, (float(x) for x in parts)))

    def projected_life_years(self, mean_rate_uv_per_h: float) -> float:
        """End of life = 10% cell-voltage rise. Calibration target: baseline ~5 years."""
        if mean_rate_uv_per_h <= 0:
            return float("inf")
        return self.dv_eol / mean_rate_uv_per_h / 8760.0
