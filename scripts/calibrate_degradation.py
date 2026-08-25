"""Calibrate the degradation coefficients against two simultaneous targets.

Owner: Person A (DECISIONS.md 5). Gate G1.2 in PLAN.md.

The two targets pull against each other, which is why hand-tuning one coefficient at a
time does not converge:

  (1) SEPARATION  -- a jittery policy must degrade >= 3x faster than a smooth one, or the
      degradation model cannot teach the agent anything and there is no paper.
  (2) ABSOLUTE    -- the literature rule-based baseline must land at 4.0 +/- 0.5 uV/h, i.e.
      the ~5-year PEM stack lifetime of ref [7] at a 10% voltage-rise end-of-life.

Target (1) needs the POLICY-DEPENDENT terms (ramp, cycling, high-j stress) to dominate the
COMMON-MODE terms (base, idle) that both policies pay equally. Target (2) constrains the
total. Scaling everything up fixes (2) and leaves (1) untouched.

    python scripts/calibrate_degradation.py            # diagnose only
    python scripts/calibrate_degradation.py --solve    # search, then write the config

Physical bounds are enforced from the literature, so the search cannot buy the targets with
unphysical numbers:
  - r_base   1-10 uV/h        steady-state PEMWE degradation (ref [3])
  - worst-case aggressive total <= 50 uV/h   (ref [4])
  - r_idle   > 0              anode at OCV is not a free state (refs [1], [5])
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import yaml

from pemwe import load_config, PEMWEEnv, BASELINES

ROOT = Path(__file__).resolve().parents[1]
TERMS = ("base", "stress", "ramp", "cycle", "idle")
N_PROFILES = 8          # average over several days so one profile cannot flatter a policy


class Jittery:
    """Adversarial upper bound: chases every fluctuation, cycles constantly."""
    def __init__(self, seed): self.rng = np.random.default_rng(seed)
    def reset(self): pass
    def act(self, obs, info=None):
        return np.array([self.rng.uniform(-1, 1)], dtype=np.float32)


class Smooth:
    """Well-behaved reference: tracks renewable but caps mid-range and never chases."""
    def reset(self): pass
    def act(self, obs, info=None):
        return np.array([2 * min(float(obs[0]), 0.55) - 1], dtype=np.float32)


def rollout(cfg, policy, seed):
    env = PEMWEEnv(cfg, seed=seed)
    obs, _ = env.reset(seed=seed)
    policy.reset()
    parts = dict.fromkeys(TERMS, 0.0)
    h2 = cycles = 0.0
    while True:
        obs, _, term, trunc, info = env.step(policy.act(obs))
        for k in TERMS:
            parts[k] += info["dv_parts"][k]
        h2 += info["h2_kg"]
        cycles += int(info["cycled"])
        if term or trunc:
            return parts, h2, cycles


def measure(cfg, policy_factory):
    """Mean per-term uV over N_PROFILES days."""
    acc = dict.fromkeys(TERMS, 0.0)
    h2s, cyc = [], []
    for s in range(N_PROFILES):
        p, h2, c = rollout(cfg, policy_factory(s), s)
        for k in TERMS:
            acc[k] += p[k] / N_PROFILES
        h2s.append(h2)
        cyc.append(c)
    return acc, float(np.mean(h2s)), float(np.mean(cyc))


def total(parts):
    return sum(parts.values())


def report(cfg):
    pols = {
        "smooth": lambda s: Smooth(),
        "jittery": lambda s: Jittery(s),
        "baseline_naive": lambda s: BASELINES["baseline_naive"](cfg),
        "baseline_ramplimited": lambda s: BASELINES["baseline_ramplimited"](cfg),
    }
    out = {}
    print(f"{'policy':<22}" + "".join(f"{t:>9}" for t in TERMS)
          + f"{'total':>9}{'uV/h':>8}{'cycles':>8}")
    print("-" * 86)
    for name, f in pols.items():
        parts, h2, c = measure(cfg, f)
        out[name] = parts
        print(f"{name:<22}" + "".join(f"{parts[t]:>9.2f}" for t in TERMS)
              + f"{total(parts):>9.2f}{total(parts)/24:>8.2f}{c:>8.1f}")
    ratio = total(out["jittery"]) / max(total(out["smooth"]), 1e-9)
    base_rate = total(out["baseline_naive"]) / 24
    print(f"\n  separation  jittery/smooth = {ratio:5.2f}x   (target >= 3.00)  "
          f"{'PASS' if ratio >= 3 else 'FAIL'}")
    print(f"  absolute    baseline_naive = {base_rate:5.2f} uV/h  (target 4.0 +/- 0.5)  "
          f"{'PASS' if abs(base_rate-4.0) <= 0.5 else 'FAIL'}")

    common = sum(out["smooth"][t] for t in ("base", "idle"))
    print(f"\n  common-mode (base+idle) is {common/total(out['smooth'])*100:.0f}% of the "
          f"smooth policy's total —")
    print("  that fraction is what caps the achievable separation ratio.")
    return ratio, base_rate


def unit_integrals(cfg):
    """Every term is LINEAR in its coefficient, so measure each term's integral once with
    all coefficients set to 1.0. Any candidate coefficient set is then an exact dot product
    -- no re-simulation. (Degradation feeds back into v_cell, but over 24 h that is tens of
    microvolts on ~1.8 V, so the integrals are coefficient-invariant to ~1e-5. The AFTER
    report re-runs the real environment, which would expose it if that ever stopped holding.)"""
    import copy
    c = copy.deepcopy(cfg)
    c["degradation"].update(r_base_uv_per_h=1.0, k_j_uv_per_h=1.0, k_ramp_uv_per_h=1.0,
                            dv_cycle_uv=1.0, r_idle_uv_per_h=1.0)
    pols = {
        "smooth": lambda s: Smooth(),
        "jittery": lambda s: Jittery(s),
        "baseline_naive": lambda s: BASELINES["baseline_naive"](c),
    }
    return {n: measure(c, f)[0] for n, f in pols.items()}


COEFF_OF = {"base": "r_base_uv_per_h", "stress": "k_j_uv_per_h",
            "ramp": "k_ramp_uv_per_h", "cycle": "dv_cycle_uv", "idle": "r_idle_uv_per_h"}


def evaluate(units, c):
    """Exact total uV for a policy given coefficients c, from the unit integrals."""
    return sum(units[t] * c[COEFF_OF[t]] for t in TERMS)


def solve(cfg):
    """Search the five coefficients against both targets, inside literature bounds."""
    U = unit_integrals(cfg)
    print("unit integrals (uV per unit coefficient):")
    print(f"  {'policy':<22}" + "".join(f"{t:>10}" for t in TERMS))
    for n, u in U.items():
        print(f"  {n:<22}" + "".join(f"{u[t]:>10.3f}" for t in TERMS))
    print()

    # Literature-bounded grids. r_base >= 1.0 keeps steady-state degradation inside the
    # 1-10 uV/h band of ref [3]; r_idle >= 0.5 keeps the anode-at-OCV term meaningful
    # (refs [1],[5]) so "idle forever" stays a costly policy, per DECISIONS.md 5.
    import numpy as _np
    grids = {
        "r_base_uv_per_h":  _np.arange(1.0, 3.01, 0.25),
        "r_idle_uv_per_h":  _np.arange(0.5, 2.01, 0.25),
        "k_ramp_uv_per_h":  _np.arange(4.0, 40.01, 1.0),
        "dv_cycle_uv":      _np.arange(1.0, 12.01, 0.5),
        "k_j_uv_per_h":     _np.arange(20.0, 90.01, 5.0),
    }
    best, n_feasible = None, 0
    for rb in grids["r_base_uv_per_h"]:
      for ri in grids["r_idle_uv_per_h"]:
        for kr in grids["k_ramp_uv_per_h"]:
          for dc in grids["dv_cycle_uv"]:
            for kj in grids["k_j_uv_per_h"]:
                c = {"r_base_uv_per_h": float(rb), "r_idle_uv_per_h": float(ri),
                     "k_ramp_uv_per_h": float(kr), "dv_cycle_uv": float(dc),
                     "k_j_uv_per_h": float(kj)}
                S, J, N = (evaluate(U[k], c) for k in ("smooth", "jittery", "baseline_naive"))
                ratio, rate, agg = J / max(S, 1e-9), N / 24.0, J / 24.0
                if agg > 50.0:                       # ref [4] worst-case ceiling
                    continue
                if ratio < 3.0 or abs(rate - 4.0) > 0.5:
                    continue
                # The separation must come from BOTH intermittency mechanisms named in
                # ref [4] -- ramping AND on/off cycling. A solution where one term supplies
                # nearly all of it is a numerical fit, not a physical model.
                jr = U["jittery"]["ramp"] * kr
                jc = U["jittery"]["cycle"] * dc
                diff = jr + jc + U["jittery"]["stress"] * kj
                if min(jr, jc) / diff < 0.20:
                    continue
                n_feasible += 1
                # Prefer a comfortable but not extreme ratio, then closeness to 4.0 uV/h.
                score = (-abs(min(ratio, 8.0) - 4.5), -abs(rate - 4.0))
                if best is None or score > best[0]:
                    best = (score, c, ratio, rate, agg)
    print(f"feasible coefficient sets found: {n_feasible}")
    print()
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solve", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    print("=== BEFORE ===\n")
    report(cfg)

    if not args.solve:
        print("\nDiagnostic only. Re-run with --solve to search for coefficients.")
        return

    best = solve(cfg)
    if best is None:
        print("\nNo coefficient set satisfied both targets inside the physical bounds.")
        print("Widen the grid or revisit the bounds — do not relax the targets silently.")
        sys.exit(1)

    _, coeffs, ratio, rate, agg = best
    print(f"\n=== SOLUTION  ratio {ratio:.2f}x, baseline {rate:.2f} uV/h, "
          f"aggressive {agg:.1f} uV/h ===\n")
    for k, v in coeffs.items():
        print(f"  {k:<24} {v}")

    p = ROOT / "configs" / "default.yaml"
    txt = p.read_text(encoding="utf-8")
    for k, v in coeffs.items():
        import re
        txt = re.sub(rf"^(  {k}: )[0-9.]+", rf"\g<1>{v}", txt, flags=re.M)
    p.write_text(txt, encoding="utf-8")
    print(f"\nwrote {p}")

    cfg2 = load_config()
    print("\n=== AFTER ===\n")
    report(cfg2)


if __name__ == "__main__":
    main()
