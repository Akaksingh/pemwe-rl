"""90-day persistent-degradation rollout. OWNER: Person A (brief, Day 3 item 1).

This is where the life-extension headline number comes from. Training uses independent
24 h episodes (DECISIONS.md 1), so within an episode the accumulated dV_deg is ~100 uV
against a 1.85 V cell -- far too small to change the physics. It is the REWARD that
teaches the policy to care. Over 90 chained days with `env.persist_degradation: true`,
the accumulated voltage rise does start to feed measurably back through the polarization
curve and cost real yield, and that is the regime the lifetime claim is made in.

Writes one results/<run_id>.json per policy, in the CONTRACTS.md section 3 schema with
profile_set = "longhorizon_90d", so Person C's fig_longhorizon() reads it unchanged.
Per that schema `trajectory` carries ONE representative day; the 90-day cumulative curve
is the cumulative sum of episodes[].dv_deg_uv.

    python scripts/longhorizon_rollout.py
    python scripts/longhorizon_rollout.py --days 90 --profiles data/processed/kutch_2019_1min.parquet
    python scripts/longhorizon_rollout.py --model models/sac_w2-1.0_seed0.zip --policy sac
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from pemwe import load_config, PEMWEEnv, BASELINES
from pemwe.config import deepcopy_cfg

OUT = ROOT / "results"


class SB3Policy:
    """Wrap a trained SB3 model in the same .reset()/.act() shape as the baselines."""

    def __init__(self, path):
        from stable_baselines3 import SAC, PPO
        cls = SAC if "sac" in Path(path).name.lower() else PPO
        self.model = cls.load(path, device="cpu")

    def reset(self):
        pass

    def act(self, obs, info=None):
        a, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(a, dtype=np.float32).reshape(1)


def run(cfg, policy, n_days, profiles, run_id, policy_name, seed=0, keep_day=1):
    """Chain n_days episodes with degradation persisting across resets."""
    c = deepcopy_cfg(cfg)
    c["env"]["persist_degradation"] = True
    env = PEMWEEnv(c, profiles=profiles, seed=seed)
    env.dv_deg_uv = 0.0

    episodes, traj = [], None
    dt_h = env.dt_min / 60.0

    for d in range(n_days):
        opts = {"day_idx": d % len(profiles)} if profiles is not None else None
        obs, _ = env.reset(seed=seed + d, options=opts)
        policy.reset()
        acc = {"h2": 0.0, "dv": 0.0, "eta": [], "curt": 0.0, "cyc": 0,
               "ramp": [], "rew": 0.0, "ry": 0.0, "rd": 0.0, "rr": 0.0}
        rows = {"t_min": [], "p_renew_w": [], "p_set_w": [], "j": [], "dv_deg_total_uv": []}

        for t in range(env.n_steps):
            obs, r, term, trunc, info = env.step(policy.act(obs))
            acc["h2"] += info["h2_kg"]
            acc["dv"] += info["dv_deg_uv"]
            acc["curt"] += info["curtailed_w"] * dt_h / 1000.0
            acc["cyc"] += int(info["cycled"])
            acc["ramp"].append(info["r_ramp"])
            acc["rew"] += r
            acc["ry"] += info["r_yield"]
            acc["rd"] += info["r_deg"]
            acc["rr"] += info["r_ramp"]
            if info["is_on"]:
                acc["eta"].append(info["eta_lhv"])
            if d == keep_day:
                rows["t_min"].append(t)
                rows["p_renew_w"].append(info["p_renew_w"])
                rows["p_set_w"].append(info["p_set_w"])
                rows["j"].append(info["j"])
                rows["dv_deg_total_uv"].append(info["dv_deg_total_uv"])
            if term or trunc:
                break

        if d == keep_day:
            traj = rows
        episodes.append({
            "date": f"day_{d:03d}",
            "h2_kg": acc["h2"], "dv_deg_uv": acc["dv"],
            "mean_eta_lhv": float(np.mean(acc["eta"])) if acc["eta"] else 0.0,
            "curtailed_kwh": acc["curt"], "n_cycles": acc["cyc"],
            "mean_abs_ramp": float(np.mean(acc["ramp"])),
            "reward_total": acc["rew"],
            "r_yield": acc["ry"], "r_deg": acc["rd"], "r_ramp": acc["rr"],
        })

    h2 = np.array([e["h2_kg"] for e in episodes])
    dv = np.array([e["dv_deg_uv"] for e in episodes])
    rate = float(dv.sum() / (n_days * 24.0))
    return {
        "run_id": run_id, "policy": policy_name, "seed": seed,
        "weights": {k: cfg["reward"][k] for k in ("w1", "w2", "w3")},
        "profile_set": "longhorizon_90d",
        "episodes": episodes,
        "aggregate": {
            "h2_kg_mean": float(h2.mean()), "h2_kg_std": float(h2.std()),
            "dv_deg_uv_mean": float(dv.mean()), "dv_deg_uv_std": float(dv.std()),
            "deg_rate_uv_per_h": rate,
            "projected_life_years": float(env.deg.projected_life_years(rate)),
            "h2_kg_total": float(h2.sum()),
            "dv_deg_uv_total": float(dv.sum()),
            "dv_eol_uv": env.deg.dv_eol,
            "n_days": n_days,
        },
        "trajectory": traj or {},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--profiles", default=None,
                    help="parquet path; defaults to the real Kutch profiles when built")
    ap.add_argument("--model", default=None, help="trained SB3 checkpoint (Person B)")
    ap.add_argument("--policy", default="sac", help="policy label for --model runs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    profiles = None
    # Default to the REAL Kutch weather whenever it exists, the same rule
    # scripts/calibrate_degradation.py follows. The synthetic generator is a fallback for
    # a fresh checkout, never a silent default once the parquet is built -- a life-extension
    # number quoted from synthetic weather would be wrong in the paper.
    if not args.profiles:
        default = ROOT / cfg["data"]["profile_path"]
        if default.exists():
            args.profiles = str(default)
    if args.profiles:
        import pandas as pd
        df = pd.read_parquet(args.profiles)
        n = cfg["env"]["steps_per_episode"]
        col = "hybrid_w" if "hybrid_w" in df.columns else df.columns[0]
        v = df[col].to_numpy(float)
        profiles = v[: len(v) // n * n].reshape(-1, n)
        print(f"profiles: REAL {args.profiles} ({profiles.shape[0]} days available)")
    else:
        print("profiles: SYNTHETIC -- re-run with --profiles for the paper number")

    jobs = []
    if args.model:
        jobs.append((f"{args.policy}_longhorizon_seed{args.seed}", args.policy,
                     SB3Policy(args.model)))
    else:
        for name, cls in BASELINES.items():
            jobs.append((f"{name}_longhorizon", name, cls(cfg)))

    OUT.mkdir(exist_ok=True)
    print(f"\n{args.days}-day persistent-degradation rollout "
          f"(env.persist_degradation = true)\n")
    print(f"{'policy':<24}{'H2 total':>12}{'dV total':>12}{'rate':>11}{'life':>10}")
    print("-" * 69)
    results = []
    for run_id, label, pol in jobs:
        res = run(cfg, pol, args.days, profiles, run_id, label, seed=args.seed)
        a = res["aggregate"]
        (OUT / f"{run_id}.json").write_text(json.dumps(res))
        results.append(res)
        print(f"{label:<24}{a['h2_kg_total']:11.0f} kg{a['dv_deg_uv_total']:10.1f} uV"
              f"{a['deg_rate_uv_per_h']:9.2f} uV/h{a['projected_life_years']:8.2f} yr")
    print("-" * 69)

    if len(results) == 2:
        a, b = results[0]["aggregate"], results[1]["aggregate"]
        dl = b["projected_life_years"] - a["projected_life_years"]
        dh = (b["h2_kg_total"] / a["h2_kg_total"] - 1.0) * 100.0
        print(f"\n{results[1]['policy']} vs {results[0]['policy']}: "
              f"{dl:+.2f} yr of life ({dl/a['projected_life_years']*100:+.1f}%) "
              f"for {dh:+.1f}% H2")
        print("-> that trade is the Pareto point (DECISIONS.md 8). Both numbers, always.")

    print(f"\nwrote {len(results)} result files to {OUT} "
          f"(schema: CONTRACTS.md 3, profile_set=longhorizon_90d)")
    print("-> Person C: fig_longhorizon() reads these; cumsum episodes[].dv_deg_uv "
          "for the 90-day curve, EoL line at aggregate.dv_eol_uv")


if __name__ == "__main__":
    main()
