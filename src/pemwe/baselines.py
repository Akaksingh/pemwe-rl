"""Baseline controllers. OWNER: Person B.

Two of them, deliberately:

  NaiveLoadFollowing  -- the rule from ref [8], verbatim. The literature baseline.
  RampLimitedBaseline -- the same rule plus a slew limit. This one must be GENUINELY GOOD.

Build the second one honestly. A strawman baseline is the fastest way to lose a reviewer,
and "RL beats a deliberately bad controller" is not a result. If RL only ties the
ramp-limited baseline on yield but wins on degradation, that is still the Pareto story
(DECISIONS.md section 8) and the paper stands.

Both expose the same `.act(obs, info) -> np.ndarray` as an SB3 policy, so evaluate.py
treats scripted and learned policies identically.

ACTION SEMANTICS: the environment reads the action as a fraction of the power CURRENTLY
AVAILABLE, not of rated (see env.step for why). These controllers are written in terms of
a target power fraction OF RATED, which is how the literature states them, and convert at
the end -- so the control laws below still read the way the papers write them.
"""

from __future__ import annotations

import numpy as np


def _to_action(target_frac_of_rated: float, p_renew_frac: float) -> np.ndarray:
    """Target power (as a fraction of rated) -> env action (fraction of available).

    Undefined when nothing is available; commanding zero is the only meaningful choice.
    """
    if p_renew_frac <= 1e-9:
        u = 0.0
    else:
        u = np.clip(target_frac_of_rated / p_renew_frac, 0.0, 1.0)
    return np.array([2.0 * u - 1.0], dtype=np.float32)


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
        target = p_renew_frac if p_renew_frac >= self.thr else self.idle
        return _to_action(target, p_renew_frac)


class RampLimitedBaseline(NaiveLoadFollowing):
    """Load-following with a slew-rate limit -- the strong, honest baseline.

    `ramp_limit_frac_per_min` is NOT a guess. It is chosen by `scripts/tune_baseline.py`
    at the KNEE of this controller's own yield-life frontier, measured over 24 TRAIN days:
    past that point further smoothing buys rapidly less stack life per kilogram of
    hydrogen given up.

    Two criteria were rejected, and the reasons matter for the paper:

      * argmax of reward at the default weights gives up almost all the life benefit --
        at w2 = 1 the degradation term is ~1 % of the yield term, so reward is nearly flat
        across the whole grid and its argmax is a strawman on precisely the axis the
        paper's claim lives on.
      * argmax of projected life picks the tightest limit on the grid, a controller so
        sluggish it barely follows the resource. Also a strawman, in the other direction.

    `tests/test_experiments.py::test_ramp_limited_baseline_is_not_a_strawman` holds the
    line: this controller must degrade less than the naive rule while keeping >90 % of its
    hydrogen, or the suite fails.

    The slew limit applies to POWER (as a fraction of rated), which is what the physical
    constraint is about -- not to the action, which is relative to a moving resource.
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
        # the resource itself caps what is reachable this step
        target = min(target, p_renew_frac)
        delta = np.clip(target - self.prev, -self.limit, self.limit)
        self.prev = float(np.clip(self.prev + delta, 0.0, 1.0))
        return _to_action(self.prev, p_renew_frac)


BASELINES = {c.name: c for c in (NaiveLoadFollowing, RampLimitedBaseline)}
