"""Episode-parallel (batched) simulator: B independent days stepped in lockstep with NumPy.

WHY
---
`PEMWEEnv.step` costs about 80 microseconds, almost all of it Python and NumPy call overhead on scalars,
and the process-parallel harness (one episode per worker) multiplies that cost by the number of cores.
Episodes never interact, so the same arithmetic can be applied to an array of B episodes at once: the
per-call overhead is paid once per step for all of them.

WHAT IT IS (AND IS NOT)
-----------------------
A re-implementation of `PEMWEEnv.step` that REUSES the physics (`StackModel`, `DegradationModel` constants)
and mirrors the scalar code line by line, including the float32 rounding of the action and of the
observation that scripted rules see. It does not change any model. `tests/test_batch.py` checks it against
the scalar environment step by step (same actions, same profiles) and episode by episode for the scripted
rules. Not supported: `persist_degradation: true` (the long-horizon rollout) and the gymnasium API.

USAGE
-----
    sim = BatchSim(cfg, profiles)                    # profiles: (B, T) watts
    res = sim.rollout(BatchHysteresis(p_on=..., ...))  # one rule per episode (parameters broadcast to B)
    res["h2_kg"], res["dv_deg_uv"], ...               # (B,) per-episode metrics, same keys as evaluate.py

Learned policies can use `sim.reset()` / `sim.step(action)` / `sim.obs()` (a (B, 8) float32 observation with
the frozen contract of CONTRACTS.md section 1) and run one batched forward pass per step.
"""
from __future__ import annotations

import numpy as np

from .stack import StackModel
from .degradation import DegradationModel


def _f32(x):
    """Round to float32 and back, as the scalar environment does when a rule returns an action array."""
    return np.asarray(x, dtype=np.float64).astype(np.float32).astype(np.float64)


def _to_action(target_frac_of_rated, p_renew_frac):
    """Vector form of baselines._to_action: target power (fraction of rated) -> action in [-1, 1] (float32-rounded)."""
    p = np.asarray(p_renew_frac, dtype=np.float64)
    safe = np.where(p <= 1e-9, 1.0, p)
    u = np.where(p <= 1e-9, 0.0, np.clip(target_frac_of_rated / safe, 0.0, 1.0))
    return _f32(2.0 * u - 1.0)


class BatchSim:
    def __init__(self, cfg: dict, profiles: np.ndarray):
        self.cfg = cfg
        self.stack = StackModel(cfg)
        self.deg = DegradationModel(cfg)
        e = cfg["env"]
        if e["persist_degradation"]:
            raise NotImplementedError("persist_degradation=true (long-horizon rollout) is not supported by BatchSim")
        self.dt_min = e["dt_min"]
        self.n_steps = e["steps_per_episode"]
        r = cfg["reward"]
        self.w1, self.w2, self.w3 = r["w1"], r["w2"], r["w3"]
        self.h2_scale, self.deg_scale = r["h2_scale_kg"], r["deg_scale_uv"]
        self.p_rated = self.stack.p_rated
        prof = np.asarray(profiles, dtype=np.float64)
        if prof.ndim != 2 or prof.shape[1] < self.n_steps:
            raise ValueError(f"profiles must be (B, >= {self.n_steps}), got {prof.shape}")
        self.B = prof.shape[0]
        # time-major copy: row t is contiguous, so profile[:, t] for every episode is one cheap slice
        self.prof_t = np.ascontiguousarray(prof[:, : self.n_steps].T)
        # the scalar env keeps a list of the last 15 resource values, initialised to profile[0] and appended to
        # every step; before step t it holds ext[t : t + 15] with ext = [p0] * 15 + profile
        ext = np.concatenate([np.repeat(prof[:, :1], 15, axis=1), prof[:, : self.n_steps]], axis=1)
        cs = np.concatenate([np.zeros((self.B, 1)), np.cumsum(ext, axis=1)], axis=1)
        self._ma15_t = np.ascontiguousarray(((cs[:, 15:] - cs[:, :-15]) / 15.0).T)    # (T + 1, B): value before step t
        self._h2_ref = self.stack.h2_rate_kg_s(self.stack.j_rated) * self.dt_min * 60.0
        self._dt_h = self.dt_min / 60.0
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        B = self.B
        self.t = 0
        self.j = np.zeros(B)
        self.j_prev = np.zeros(B)
        self.p_set_prev = np.zeros(B)
        self.is_on = np.zeros(B, dtype=bool)
        self.dv_deg_uv = np.zeros(B)
        self.h2_last = np.zeros(B)
        # per-episode accumulators (same summation order as evaluate.rollout_episode)
        self.acc = {k: np.zeros(B) for k in ("h2_kg", "dv_deg_uv", "curtailed_kwh", "reward_total",
                                             "r_yield", "r_deg", "r_ramp", "eta_sum")}
        self.n_cycles = np.zeros(B, dtype=np.int64)
        self.n_on = np.zeros(B, dtype=np.int64)
        return self.obs()

    def p_renew_frac(self, t: int | None = None):
        """Observation field 0 as float64 (rounded through float32, as the scalar environment emits it)."""
        t = self.t if t is None else t
        return _f32(self.prof_t[min(t, self.n_steps - 1)] / self.p_rated)

    def obs(self):
        """(B, 8) float32, frozen contract (CONTRACTS.md section 1); the observation for the NEXT step."""
        t = self.t
        hour = (t * self.dt_min / 60.0) % 24
        p = self.prof_t[min(t, self.n_steps - 1)]
        o = np.empty((self.B, 8), dtype=np.float32)
        o[:, 0] = p / self.p_rated
        o[:, 1] = self._ma15_t[t] / self.p_rated
        o[:, 2] = self.p_set_prev / self.p_rated
        o[:, 3] = self.j / self.stack.j_rated
        o[:, 4] = self.dv_deg_uv / self.deg.dv_eol
        o[:, 5] = self.h2_last / max(self._h2_ref, 1e-12)
        o[:, 6] = np.sin(2 * np.pi * hour / 24)
        o[:, 7] = np.cos(2 * np.pi * hour / 24)
        return o

    # ------------------------------------------------------------------- step
    def _j_from_power(self, p, dv_v):
        st = self.stack
        j = np.interp(p, st._p_grid, st._j_grid)
        for _ in range(2):                         # the scalar code skips this only when dv == 0, where it is a no-op
            j = np.interp(p - st._c_ja * j * dv_v, st._p_grid, st._j_grid)
        j = np.where(p >= st.power_w(st.j_rated, dv_v), st.j_rated, j)
        return np.where(p < st.p_min_on, 0.0, j)

    def step(self, action):
        """Advance every episode one minute. `action`: (B,) in [-1, 1], a fraction of the AVAILABLE power."""
        st, dg = self.stack, self.deg
        t = self.t
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(self.B), -1.0, 1.0)
        u = 0.5 * (a + 1.0)
        p_renew = self.prof_t[t]
        p_set = np.minimum(u * p_renew, self.p_rated)

        dv_v = self.dv_deg_uv * 1e-6
        j = self._j_from_power(p_set, dv_v)
        was_on = self.is_on
        on = j >= st.j_min
        j = np.where(on, j, 0.0)
        p_set = np.where(on, p_set, 0.0)

        h2 = np.where(on, st.h2_rate_kg_s(j, dv_v) * self.dt_min * 60.0, 0.0)
        # degradation: five terms, summed in the order of degradation.TERMS
        dt_h, dt_min = self._dt_h, self.dt_min
        jf = j / st.j_rated
        c = dg.coeffs
        stress = np.maximum(0.0, jf - dg.j_thr) ** dg.j_exp * dt_h
        dv_step = ((((np.where(on, dt_h, 0.0) * c[0] + np.where(on, stress, 0.0) * c[1])
                     + (np.abs(j - self.j_prev) / dt_min * dt_h) * c[2])
                    + (on != was_on).astype(np.float64) * c[3])
                   + np.where(on, 0.0, dt_h) * c[4])
        self.dv_deg_uv = self.dv_deg_uv + dv_step

        r_yield = h2 / self.h2_scale
        r_deg = dv_step / self.deg_scale
        r_ramp = np.abs(j - self.j_prev) / st.j_rated
        reward = self.w1 * r_yield - self.w2 * r_deg - self.w3 * r_ramp

        acc = self.acc
        acc["h2_kg"] += h2
        acc["dv_deg_uv"] += dv_step
        acc["curtailed_kwh"] += np.maximum(0.0, p_renew - p_set) * dt_h / 1000.0
        acc["reward_total"] += reward
        acc["r_yield"] += r_yield
        acc["r_deg"] += r_deg
        acc["r_ramp"] += r_ramp
        eta = np.where(on, st.eta_lhv(np.maximum(j, 1e-9), dv_v), 0.0)
        acc["eta_sum"] += eta
        self.n_on += on
        self.n_cycles += (on != was_on)

        self.is_on, self.j_prev, self.j, self.p_set_prev, self.h2_last = on, j, j, p_set, h2
        self.t += 1
        return dict(h2_kg=h2, dv_step_uv=dv_step, j=j, p_set_w=p_set, is_on=on, reward=reward, p_renew_w=p_renew)

    # ---------------------------------------------------------------- rollout
    def metrics(self):
        """Per-episode metrics with the keys of evaluate.rollout_episode (arrays of length B)."""
        out = {k: v.copy() for k, v in self.acc.items() if k != "eta_sum"}
        n = self.t
        out["n_cycles"] = self.n_cycles.copy()
        out["mean_abs_ramp"] = self.acc["r_ramp"] / n
        out["mean_eta_lhv"] = np.where(self.n_on > 0, self.acc["eta_sum"] / np.maximum(self.n_on, 1), 0.0)
        return out

    def rollout(self, rule, keep_trajectory: bool = False):
        """Run every episode to the end of the day under a batched rule; returns metrics (and optionally j, p_set)."""
        self.reset()
        rule.reset(self.B)
        traj = {k: np.empty((self.n_steps, self.B)) for k in ("j", "p_set_w", "h2_kg", "dv_step_uv")} if keep_trajectory else None
        for t in range(self.n_steps):
            a = rule.act(self.p_renew_frac(t))
            s = self.step(a)
            if traj is not None:
                for k in traj:
                    traj[k][t] = s[k]
        m = self.metrics()
        if traj is not None:
            m["trajectory"] = traj
        return m


# ----------------------------------------------------------------------------- batched rules
class _Rule:
    def reset(self, B):                      # noqa: D401
        pass


class BatchNaive(_Rule):
    """baselines.NaiveLoadFollowing, with the threshold and idle fraction free to differ per episode."""

    def __init__(self, thr=0.10, idle=0.0):
        self.thr, self.idle = thr, idle

    def act(self, p):
        target = np.where(p >= self.thr, p, self.idle)
        return _to_action(target, p)


class BatchRampLimited(_Rule):
    """baselines.RampLimitedBaseline (threshold, idle fraction and slew limit free per episode)."""

    def __init__(self, thr=0.10, idle=0.0, limit=0.03):
        self.thr, self.idle, self.limit = thr, idle, limit

    def reset(self, B):
        self.prev = np.zeros(B)

    def act(self, p):
        target = np.minimum(np.where(p >= self.thr, p, self.idle), p)
        delta = np.clip(target - self.prev, -self.limit, self.limit)
        self.prev = np.clip(self.prev + delta, 0.0, 1.0)
        return _to_action(self.prev, p)


class BatchHysteresis(_Rule):
    """scripts/rule_family_v2.HysteresisRule: confirmed start/stop, power cap and slew limit; every parameter
    may be a scalar or a (B,) array, so a whole grid of rules is evaluated in one batch."""

    def __init__(self, p_on, k_up, k_down, slew, cap, p_off=0.03):
        self.p_on, self.k_up, self.k_down, self.slew, self.cap, self.p_off = p_on, k_up, k_down, slew, cap, p_off

    def reset(self, B):
        self.on = np.zeros(B, dtype=bool)
        self.up = np.zeros(B, dtype=np.int64)
        self.down = np.zeros(B, dtype=np.int64)
        self.prev = np.zeros(B)

    def act(self, p):
        off = ~self.on
        self.up = np.where(off, np.where(p >= self.p_on, self.up + 1, 0), self.up)
        turn_on = off & (self.up >= self.k_up)
        self.down = np.where(self.on, np.where(p < self.p_off, self.down + 1, 0), self.down)
        turn_off = self.on & (self.down >= self.k_down)
        self.on = (self.on | turn_on) & ~turn_off
        self.up = np.where(turn_on, 0, self.up)
        self.down = np.where(turn_on | turn_off, 0, self.down)
        self.prev = np.where(turn_on | turn_off, 0.0, self.prev)
        target = np.minimum(p, self.cap)
        stepped = np.clip(target, self.prev - self.slew, self.prev + self.slew)
        self.prev = np.where(self.on, stepped, self.prev)
        cmd = np.where(self.on, np.minimum(self.prev, p), 0.0)
        return _to_action(cmd, p)
