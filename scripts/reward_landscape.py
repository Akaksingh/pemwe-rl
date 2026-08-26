"""Does w2 actually move the optimum? Answered without training anything.

Person B's handoff flags that w2 barely moves the learned policy at 9% of the step budget,
and asks for a full-budget run to decide. There is a cheaper and more decisive test.

A Pareto frontier exists only if the reward's ARGMAX over policies changes as w2 rises.
That is a property of the reward function and the plant, not of the optimiser -- so it can
be measured by scoring a family of fixed, hand-written policies on real held-out days and
asking which one wins at each w2.

  - If the winner moves smoothly from aggressive to conservative as w2 rises, the frontier
    is real and Person B's shortfall is purely a training-budget problem.
  - If one policy wins at every w2, no amount of training produces a frontier, and the
    reward scaling (h2_scale_kg / deg_scale_uv) has to change before the matrix is run.

RESULT (12 held-out days, 16 fixed policies): a frontier EXISTS -- the optimum moves
capped_1.00 -> ramp_0.010 -> ramp_0.005 -> idle. Two consequences, both acted on:

  1. w2 >= ~20 selects IDLE. Running beats idling only while
     r_yield > w2 * (r_deg_run - r_deg_idle), i.e. 15790 > w2 * 780 -> w2 < 20.2.
     The sweep range was cut from 0.1-100 to 0.1-20 as a result.
  2. Every w2 from 0.1 to 5.62 picked the SAME optimum, so the old spacing spent 8 of 13
     points on one Pareto point.

CAVEAT: this is a LOWER bound on the frontier. The policy family here is crude (constant,
capped, slew-limited), so a learned policy may differentiate where these do not. Read it as
"the frontier is at least this rich", never as a ceiling on what RL can reach.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import yaml

from pemwe import load_config, PEMWEEnv, BASELINES
from pemwe import profiles as P

cfg = load_config()
W2 = cfg["sweep"]["w2"]
N_DAYS = 12


class Const:
    """Constant power fraction (clipped by the renewable cap)."""
    def __init__(self, f): self.f = f; self.name = f"const_{f:.2f}"
    def reset(self): pass
    def act(self, obs, info=None):
        return np.array([2 * self.f - 1], dtype=np.float32)


class Capped:
    """Follow the resource but never above `cap` -- the 'smooth' family."""
    def __init__(self, cap): self.cap = cap; self.name = f"capped_{cap:.2f}"
    def reset(self): pass
    def act(self, obs, info=None):
        return np.array([2 * min(float(obs[0]), self.cap) - 1], dtype=np.float32)


class Ramp:
    """Load-following with a slew limit -- the 'conservative' family."""
    def __init__(self, lim): self.lim = lim; self.name = f"ramp_{lim:.3f}"; self.prev = 0.0
    def reset(self): self.prev = 0.0
    def act(self, obs, info=None):
        tgt = float(obs[0])
        d = np.clip(tgt - self.prev, -self.lim, self.lim)
        self.prev = float(np.clip(self.prev + d, 0, 1))
        return np.array([2 * self.prev - 1], dtype=np.float32)


REAL = P.env_profiles(N_DAYS, split="test", source="hybrid")
assert REAL is not None, "needs data/processed/kutch_2019_1min.parquet"

class Idle:
    """Never run. The degenerate corner DECISIONS.md 5 warns about -- it must be in the
    family, or the scan cannot see the top of the w2 range choosing it."""
    name = "idle"
    def reset(self): pass
    def act(self, obs, info=None):
        return np.array([-1.0], dtype=np.float32)


pols = ([Idle()]
        + [Capped(c) for c in (0.05, 0.12, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0)]
        + [Ramp(l) for l in (0.005, 0.01, 0.02, 0.03, 0.06)]
        + [Const(f) for f in (0.3, 0.5)]
        + [BASELINES["baseline_naive"](cfg)])

rows = []
for pol in pols:
    name = getattr(pol, "name", pol.__class__.__name__)
    ry = rd = rr = h2 = dv = 0.0
    for d in range(N_DAYS):
        env = PEMWEEnv(cfg, profiles=REAL, seed=d)
        obs, _ = env.reset(seed=d, options={"day_idx": d})
        pol.reset()
        while True:
            obs, r, term, trunc, info = env.step(pol.act(obs))
            ry += info["r_yield"]; rd += info["r_deg"]; rr += info["r_ramp"]
            h2 += info["h2_kg"]; dv += info["dv_deg_uv"]
            if term or trunc:
                break
    rows.append(dict(name=name, ry=ry / N_DAYS, rd=rd / N_DAYS, rr=rr / N_DAYS,
                     h2=h2 / N_DAYS, dv=dv / N_DAYS))

w1, w3 = cfg["reward"]["w1"], cfg["reward"]["w3"]
print(f"{N_DAYS} held-out days, per-day means\n")
print(f"{'policy':<16}{'H2 kg':>9}{'dV uV':>9}{'r_yield':>10}{'r_deg':>9}{'r_ramp':>9}")
for r in rows:
    print(f"{r['name']:<16}{r['h2']:>9.1f}{r['dv']:>9.1f}{r['ry']:>10.0f}"
          f"{r['rd']:>9.0f}{r['rr']:>9.0f}")

print(f"\n{'w2':>8}  {'argmax policy':<16}{'H2 kg':>9}{'dV uV':>9}   deg share of objective")
winners = []
for w2 in W2:
    best = max(rows, key=lambda r: w1 * r["ry"] - w2 * r["rd"] - w3 * r["rr"])
    share = w2 * best["rd"] / max(w1 * best["ry"], 1e-9)
    winners.append(best["name"])
    print(f"{w2:>8.3g}  {best['name']:<16}{best['h2']:>9.1f}{best['dv']:>9.1f}"
          f"{share:>18.1%}")

uniq = sorted(set(winners), key=winners.index)
print(f"\ndistinct optima across the 13-point sweep: {len(uniq)}")
print(f"  {' -> '.join(uniq)}")
h2s = [next(r for r in rows if r["name"] == w)["h2"] for w in winners]
dvs = [next(r for r in rows if r["name"] == w)["dv"] for w in winners]
print(f"\nH2 spread across the frontier  : {(max(h2s)-min(h2s))/max(h2s):.1%}")
print(f"degradation spread             : {(max(dvs)-min(dvs))/max(dvs):.1%}")
print("\nVERDICT:", "FRONTIER EXISTS -- shortfall is training budget"
      if len(uniq) >= 3 else
      "NO FRONTIER -- reward scaling must change before the matrix is run")
