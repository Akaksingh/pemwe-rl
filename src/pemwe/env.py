"""Gymnasium environment. OWNER: Person A.

This runs END TO END right now. That is the whole point: Person B trains against it on
Day 1 hour 1 while Person A is still replacing the physics underneath. The observation
space, action space and `info` keys are FROZEN (CONTRACTS.md section 1) -- change the
physics freely, never the signatures.

Day-2 status: physics validated, degradation calibrated, Gate G1 passing. Two contract
BUGS were fixed here without touching the shape or the ordering of anything frozen:

  * obs[2] `p_set_prev` and obs[3] `j_frac` were byte-identical, because the stub wrote
    `self.j_prev, self.j = j, j`. One of eight observation dimensions was wasted and the
    previous SETPOINT the contract promises the agent was never actually supplied.
  * reset() reported p_renew = 0 regardless of the profile, so the first action of every
    episode was taken against a false observation.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .stack import StackModel
from .degradation import DegradationModel


def synthetic_day(rng: np.random.Generator, n: int, p_rated: float,
                  cloudiness: float = 0.4) -> np.ndarray:
    """Placeholder profile so A and B are not blocked on C's data pipeline.

    REPLACE with pemwe.profiles.get_day() once data/processed/ lands (Day 1 PM).
    """
    t = np.linspace(0, 24, n)
    clear = np.clip(np.sin((t - 6) / 12 * np.pi), 0, None) ** 1.3
    # crude OU-ish cloud transients; C's real version is turbulence-calibrated
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.92 * noise[i - 1] + rng.normal(0, cloudiness)
    return np.clip(clear * (1 + 0.25 * noise), 0, 1) * p_rated


class PEMWEEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: dict, profiles: np.ndarray | None = None, seed: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.stack = StackModel(cfg)
        self.deg = DegradationModel(cfg)

        e = cfg["env"]
        self.dt_min = e["dt_min"]
        self.n_steps = e["steps_per_episode"]
        self.persist_deg = e["persist_degradation"]

        r = cfg["reward"]
        self.w1, self.w2, self.w3 = r["w1"], r["w2"], r["w3"]
        self.h2_scale = r["h2_scale_kg"]
        self.deg_scale = r["deg_scale_uv"]

        self.p_rated = self.stack.p_rated
        self.profiles = profiles  # (n_days, n_steps) in W, from Person C

        self.observation_space = spaces.Box(-1.0, 2.0, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self.dv_deg_uv = 0.0

    # --- gym api ---------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self.profiles is not None:
            idx = (options or {}).get("day_idx", self._rng.integers(len(self.profiles)))
            self.profile = np.asarray(self.profiles[idx], dtype=float)
        else:
            self.profile = synthetic_day(self._rng, self.n_steps, self.p_rated)

        self.t = 0
        self.j = 0.0
        self.j_prev = 0.0
        self.p_set_prev = 0.0
        self.is_on = False
        self.h2_total = 0.0
        if not self.persist_deg:
            self.dv_deg_uv = 0.0
        self._renew_hist = [float(self.profile[0])] * 15
        # The agent's first action must see the renewable power actually available at
        # t = 0, not a hard-coded zero.
        return self._obs(float(self.profile[0]), 0.0), {}

    def step(self, action):
        a = float(np.clip(action, -1.0, 1.0)[0])
        u = 0.5 * (a + 1.0)                            # [-1,1] -> [0,1]
        p_renew = float(self.profile[self.t])
        # ACTION = FRACTION OF AVAILABLE POWER, not of rated.
        #
        # The obvious parameterisation, p_set = min(u * P_rated, P_renew), is degenerate
        # here: the resource sits at ~0.31 of rated on average, so every action above
        # ~0.31 produces an IDENTICAL setpoint and ~69% of the action range is a flat
        # plateau with no gradient. Trained against it, PPO saturated at maximum power and
        # produced byte-identical policies across the whole w2 sweep AND across seeds --
        # no frontier at all, because the agent could not express curtailment cheaply.
        #
        # As a fraction of what is available, every action has a distinct effect whenever
        # the resource is non-zero, and deliberate curtailment is directly expressible.
        # The renewable-only hard cap of DECISIONS.md 3 still holds by construction:
        # u <= 1 means p_set <= p_renew, always.
        p_set = min(u * p_renew, self.p_rated)

        dv_v = self.dv_deg_uv * 1e-6
        j = self.stack.j_from_power(p_set, dv_v)
        was_on, self.is_on = self.is_on, j >= self.stack.j_min
        if not self.is_on:
            j = 0.0
            p_set = 0.0            # cannot part-load below the ON threshold; drawing the
                                   # commanded power while OFF would break the hard cap

        h2_kg = self.stack.h2_rate_kg_s(j, dv_v) * self.dt_min * 60.0 if self.is_on else 0.0
        dv_step, dv_parts = self.deg.step_uv(j, self.j_prev, self.stack.j_rated,
                                             self.dt_min, self.is_on, was_on)
        self.dv_deg_uv += dv_step
        self.h2_total += h2_kg

        # UNWEIGHTED components -- CONTRACTS.md section 1. Never collapse these into one number.
        r_yield = h2_kg / self.h2_scale
        r_deg = dv_step / self.deg_scale
        r_ramp = abs(j - self.j_prev) / self.stack.j_rated
        reward = self.w1 * r_yield - self.w2 * r_deg - self.w3 * r_ramp

        p_stack = self.stack.power_w(j, dv_v) if self.is_on else 0.0
        info = {
            "p_renew_w": p_renew, "p_set_w": p_set, "p_stack_w": p_stack,
            "j": j, "v_cell": float(self.stack.v_cell(max(j, 1e-9), dv_v)),
            "h2_kg": h2_kg, "eta_lhv": float(self.stack.eta_lhv(max(j, 1e-9), dv_v)) if self.is_on else 0.0,
            "dv_deg_uv": dv_step, "dv_deg_total_uv": self.dv_deg_uv,
            "dv_parts": dv_parts, "is_on": self.is_on, "cycled": was_on != self.is_on,
            "curtailed_w": max(0.0, p_renew - p_set),
            "r_yield": r_yield, "r_deg": r_deg, "r_ramp": r_ramp,
        }

        # advance state AFTER info is built, so info describes the step just taken
        self.j_prev = j
        self.j = j
        self.p_set_prev = p_set
        self._renew_hist.append(p_renew)
        self._renew_hist = self._renew_hist[-15:]
        self.t += 1
        truncated = self.t >= self.n_steps

        next_renew = float(self.profile[min(self.t, self.n_steps - 1)])
        return self._obs(next_renew, h2_kg), float(reward), False, truncated, info

    # --- observation -----------------------------------------------------

    def _obs(self, p_renew, h2_kg):
        """CONTRACTS.md section 1, eight fields, this order. FROZEN."""
        hour = (self.t * self.dt_min / 60.0) % 24
        h2_ref = self.stack.h2_rate_kg_s(self.stack.j_rated) * self.dt_min * 60.0
        return np.array([
            p_renew / self.p_rated,                        # 0 p_renew
            float(np.mean(self._renew_hist)) / self.p_rated,  # 1 p_renew_ma15
            self.p_set_prev / self.p_rated,                # 2 p_set_prev  (was a duplicate of 3)
            self.j / self.stack.j_rated,                   # 3 j_frac
            self.dv_deg_uv / self.deg.dv_eol,              # 4 deg_norm
            h2_kg / max(h2_ref, 1e-12),                    # 5 h2_rate_norm
            np.sin(2 * np.pi * hour / 24),                 # 6 tod_sin
            np.cos(2 * np.pi * hour / 24),                 # 7 tod_cos
        ], dtype=np.float32)
