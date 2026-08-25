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
    """Load-following with a slew-rate limit -- the strong, honest baseline.

    `ramp_limit_frac_per_min` is NOT a guess. It is chosen by `scripts/tune_baseline.py`
    at the KNEE of this controller's own yield-life frontier, measured over 24 TRAIN days:
    past that point further smoothing buys rapidly less stack life per kilogram of
    hydrogen given up. At 0.03/min it keeps 98.7 % of the naive rule's yield while adding
    +1.46 years of projected life.

    Two criteria were rejected, and the reasons matter for the paper:

      * argmax of reward at the default weights picks 0.06/min. At w2 = 1 the degradation
        term is ~1 % of the yield term, so reward is nearly flat across the whole grid and
        its argmax gives up almost all the life benefit -- a strawman on precisely the axis
        the paper's claim lives on.
      * argmax of projected life picks the tightest limit on the grid, a controller so
        sluggish it barely follows the resource. Also a strawman, in the other direction.

    `tests/test_experiments.py::test_ramp_limited_baseline_is_not_a_strawman` holds the
    line: this controller must degrade less than the naive rule while keeping >90 % of its
    hydrogen, or the suite fails.
    """

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
