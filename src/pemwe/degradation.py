"""Degradation model. OWNER: Person A.

Five terms, lumped into one scalar `dv_deg_uv` = cumulative cell-voltage rise in microvolts.
That scalar feeds straight back into StackModel.v_cell, so degradation genuinely costs
future yield -- which is what forces the agent to trade short-term output against
long-term capability.

Term-by-term justification and the Day-2 calibration procedure: DECISIONS.md section 5.

The idle term is load-bearing. Delete it and the reward-optimal degenerate policy becomes
"sit at P_idle forever": zero degradation, near-zero yield. It is also physically correct
(anode at open-circuit potential drives Ir dissolution, refs #1/#5).
"""

from __future__ import annotations


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

    def step_uv(self, j, j_prev, j_rated, dt_min, is_on, was_on):
        """Degradation added this step, in microvolts. Returns (total, components)."""
        dt_h = dt_min / 60.0
        jf, jf_prev = j / j_rated, j_prev / j_rated

        base = self.r_base * dt_h if is_on else 0.0
        stress = (self.k_j * max(0.0, jf - self.j_thr) ** self.j_exp * dt_h) if is_on else 0.0
        ramp = self.k_ramp * abs(j - j_prev) / dt_min * dt_h
        cycle = self.dv_cycle if (is_on != was_on) else 0.0
        idle = self.r_idle * dt_h if not is_on else 0.0

        total = base + stress + ramp + cycle + idle
        return total, {"base": base, "stress": stress, "ramp": ramp,
                       "cycle": cycle, "idle": idle}

    def projected_life_years(self, mean_rate_uv_per_h: float) -> float:
        """End of life = 10% cell-voltage rise. Calibration target: baseline ~5 years."""
        if mean_rate_uv_per_h <= 0:
            return float("inf")
        return self.dv_eol / mean_rate_uv_per_h / 8760.0
