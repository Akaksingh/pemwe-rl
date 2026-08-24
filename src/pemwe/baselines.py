"""Baseline controllers. OWNER: Person B.

Two of them, deliberately:

  NaiveLoadFollowing  -- the rule from ref #8, verbatim. The literature baseline.
  RampLimitedBaseline -- the same rule plus a slew limit. This one must be GENUINELY GOOD.

Build the second one honestly. A strawman baseline is the fastest way to lose a reviewer,
and "RL beats a deliberately bad controller" is not a result. If RL only ties the
ramp-limited baseline on yield but wins on degradation, that is still the Pareto story
(DECISIONS.md section 8) and the paper stands.

Both expose the same `.act(obs, info) -> np.ndarray` as an SB3 policy, so evaluate.py
treats scripted and learned policies identically.
"""

from __future__ import annotations

import numpy as np


class NaiveLoadFollowing:
    """P_set = min(P_renew, P_rated) if P_renew >= threshold else P_idle.  (ref #8)"""

    name = "baseline_naive"

    def __init__(self, cfg: dict):
        b = cfg["baseline"]
        self.thr = b["p_idle_threshold_frac"]
        self.idle = b["p_idle_frac"]

    def reset(self):
        pass

    def act(self, obs, info=None):
        p_renew_frac = float(obs[0])
        frac = p_renew_frac if p_renew_frac >= self.thr else self.idle
        return np.array([2.0 * np.clip(frac, 0.0, 1.0) - 1.0], dtype=np.float32)


class RampLimitedBaseline(NaiveLoadFollowing):
    """Load-following with a slew-rate limit -- the strong, honest baseline."""

    name = "baseline_ramplimited"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.limit = cfg["baseline"]["ramp_limit_frac_per_min"]
        self.prev = 0.0

    def reset(self):
        self.prev = 0.0

    def act(self, obs, info=None):
        p_renew_frac = float(obs[0])
        target = p_renew_frac if p_renew_frac >= self.thr else self.idle
        delta = np.clip(target - self.prev, -self.limit, self.limit)
        self.prev = float(np.clip(self.prev + delta, 0.0, 1.0))
        return np.array([2.0 * self.prev - 1.0], dtype=np.float32)


BASELINES = {c.name: c for c in (NaiveLoadFollowing, RampLimitedBaseline)}
