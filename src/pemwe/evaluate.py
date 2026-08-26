"""Rollout + results emission. OWNER: Person B.

    python -m pemwe.evaluate --policy baseline_naive
    python -m pemwe.evaluate --policy sac --model models/sac_w2-10.0_seed0.zip
    python -m pemwe.evaluate --all-baselines
    python -m pemwe.evaluate --sweep                 # every model in models/

ONE CODE PATH for scripted and learned policies. The baselines already expose
`.reset()` / `.act(obs, info)` matching an SB3 policy, so `SB3Policy` below just wraps
`model.predict(..., deterministic=True)` in the same shape and `rollout()` never learns
which kind it is holding. That matters: if the baselines and the agent went through
different rollout code, any bug in one of them would look like a result.

Output is the FROZEN schema of CONTRACTS.md 3 -- Person C's entire plotting library is
written against it. `--validate` checks the emitted files key-for-key against
`scripts/fake_results.py` output, which is the reference C built those plots on.

`trajectory` is written for ONE representative episode per run only. Full trajectories for
every episode would be ~1440 x 5 floats x 72 days per run and blow up `results/`.

EVALUATION IS ON THE HELD-OUT TEST SPLIT (DECISIONS.md 1-2). If a number in the paper came
from a training day, the paper is wrong. `--split train` exists for debugging and stamps
`profile_set` accordingly so a training-split run can never be mistaken for a result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import load_config
from .env import PEMWEEnv
from .baselines import BASELINES
from . import profiles

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

# CONTRACTS.md 3, spelled exactly. The validator checks against these AND against the
# reference files, so a drift in either direction is caught.
EPISODE_KEYS = ["date", "h2_kg", "dv_deg_uv", "mean_eta_lhv", "curtailed_kwh",
                "n_cycles", "mean_abs_ramp", "reward_total", "r_yield", "r_deg", "r_ramp"]
AGGREGATE_KEYS = ["h2_kg_mean", "h2_kg_std", "dv_deg_uv_mean", "dv_deg_uv_std",
                  "deg_rate_uv_per_h", "projected_life_years"]
TRAJECTORY_KEYS = ["t_min", "p_renew_w", "p_set_w", "j", "dv_deg_total_uv"]
TOP_KEYS = ["run_id", "policy", "seed", "weights", "profile_set", "episodes",
            "aggregate", "trajectory"]


class SB3Policy:
    """A trained SB3 model behind the scripted-policy interface."""

    def __init__(self, path, algo=None):
        from stable_baselines3 import SAC, PPO
        name = Path(path).name.lower()
        algo = (algo or ("sac" if "sac" in name else "ppo")).lower()
        cls = {"sac": SAC, "ppo": PPO}[algo]
        self.model = cls.load(path, device="cpu")   # inference is cheap; CPU is fine
        self.name = algo

    def reset(self):
        pass

    def act(self, obs, info=None):
        a, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(a, dtype=np.float32).reshape(1)


def rollout_episode(env, policy, day_idx, seed, keep_trajectory=False):
    obs, _ = env.reset(seed=seed, options={"day_idx": day_idx})
    policy.reset()
    dt_h = env.dt_min / 60.0

    h2 = dv = curt = rew = ry = rd = rr = 0.0
    cycles = 0
    etas, ramps = [], []
    traj = {k: [] for k in TRAJECTORY_KEYS} if keep_trajectory else None

    for t in range(env.n_steps):
        obs, r, term, trunc, info = env.step(policy.act(obs))
        h2 += info["h2_kg"]
        dv += info["dv_deg_uv"]
        curt += info["curtailed_w"] * dt_h / 1000.0
        cycles += int(info["cycled"])
        rew += r
        ry += info["r_yield"]
        rd += info["r_deg"]
        rr += info["r_ramp"]
        ramps.append(info["r_ramp"])
        if info["is_on"]:
            etas.append(info["eta_lhv"])
        if traj is not None:
            traj["t_min"].append(t)
            traj["p_renew_w"].append(info["p_renew_w"])
            traj["p_set_w"].append(info["p_set_w"])
            traj["j"].append(info["j"])
            traj["dv_deg_total_uv"].append(info["dv_deg_total_uv"])
        if term or trunc:
            break

    ep = {
        "date": None,                       # filled by the caller, which knows the date
        "h2_kg": h2, "dv_deg_uv": dv,
        "mean_eta_lhv": float(np.mean(etas)) if etas else 0.0,
        "curtailed_kwh": curt, "n_cycles": cycles,
        "mean_abs_ramp": float(np.mean(ramps)),
        "reward_total": rew, "r_yield": ry, "r_deg": rd, "r_ramp": rr,
    }
    return ep, traj


def evaluate(cfg, policy, policy_name, run_id, seed=0, split="test",
             source="hybrid", max_days=None, rep_day=None):
    df = profiles.load_profiles(cfg["data"]["profile_path"])

    if split in ("test", "train"):
        dates = profiles.SPLITS[split]
        profile_set = split
    elif split.startswith("archetype_"):
        key = split[len("archetype_"):]
        dates = [profiles.SPLITS["archetypes"][key]]
        profile_set = split
    else:
        raise ValueError(f"unknown split {split!r}")
    if max_days:
        dates = dates[:max_days]

    prof = profiles.profiles_array(df, dates, source)
    env = PEMWEEnv(cfg, profiles=prof, seed=seed)

    # One representative episode per run: the median-yield day, so the example-day figure
    # shows a typical day rather than the best or worst one.
    if rep_day is None:
        first, _ = zip(*[rollout_episode(env, policy, i, seed) for i in range(len(dates))])
        rep_day = int(np.argsort([e["h2_kg"] for e in first])[len(first) // 2])

    episodes, traj = [], None
    for i, date in enumerate(dates):
        ep, tr = rollout_episode(env, policy, i, seed, keep_trajectory=(i == rep_day))
        ep["date"] = date
        episodes.append(ep)
        if tr is not None:
            traj = tr

    h2 = np.array([e["h2_kg"] for e in episodes])
    dv = np.array([e["dv_deg_uv"] for e in episodes])
    rate = float(dv.mean() / 24.0)

    return {
        "run_id": run_id,
        "policy": policy_name,
        "seed": seed,
        "weights": {k: cfg["reward"][k] for k in ("w1", "w2", "w3")},
        "profile_set": profile_set,
        "episodes": episodes,
        "aggregate": {
            "h2_kg_mean": float(h2.mean()), "h2_kg_std": float(h2.std()),
            "dv_deg_uv_mean": float(dv.mean()), "dv_deg_uv_std": float(dv.std()),
            "deg_rate_uv_per_h": rate,
            "projected_life_years": float(env.deg.projected_life_years(rate)),
        },
        "trajectory": traj or {k: [] for k in TRAJECTORY_KEYS},
    }


# --------------------------------------------------------------------------------------
# schema validation against C's reference
# --------------------------------------------------------------------------------------

def validate(result, reference=None) -> list[str]:
    """Return a list of problems. Empty list means the file matches the frozen schema."""
    problems = []

    def check(name, got, want):
        missing, extra = set(want) - set(got), set(got) - set(want)
        if missing:
            problems.append(f"{name}: MISSING {sorted(missing)}")
        if extra:
            problems.append(f"{name}: unexpected {sorted(extra)}")

    check("top level", result, TOP_KEYS)
    check("weights", result.get("weights", {}), ["w1", "w2", "w3"])
    if result.get("episodes"):
        check("episodes[0]", result["episodes"][0], EPISODE_KEYS)
    else:
        problems.append("episodes: empty")
    check("aggregate", result.get("aggregate", {}), AGGREGATE_KEYS)
    if result.get("trajectory"):
        check("trajectory", result["trajectory"], TRAJECTORY_KEYS)
        lens = {k: len(v) for k, v in result["trajectory"].items()}
        if len(set(lens.values())) > 1:
            problems.append(f"trajectory: ragged lengths {lens}")

    for e in result.get("episodes", []):
        for k in ("h2_kg", "dv_deg_uv", "curtailed_kwh"):
            if e.get(k, 0) < 0:
                problems.append(f"episode {e.get('date')}: negative {k}")

    # Cross-check against the reference C actually built the plots on.
    if reference is not None:
        for name, got, want in [("top level", result, reference),
                                ("aggregate", result["aggregate"], reference["aggregate"])]:
            missing = set(want) - set(got)
            if missing:
                problems.append(f"vs fake_results {name}: MISSING {sorted(missing)}")
        if reference.get("episodes") and result.get("episodes"):
            missing = set(reference["episodes"][0]) - set(result["episodes"][0])
            if missing:
                problems.append(f"vs fake_results episodes: MISSING {sorted(missing)}")
    return problems


def _reference():
    """A results file to cross-check the emitted schema against.

    Must be a TEST-split result. Taking the first file that merely has the right top-level
    keys picked up `*_longhorizon.json`, whose aggregate legitimately carries extra fields
    (n_days, dv_eol_uv, h2_kg_total, dv_deg_uv_total) that an ordinary run does not -- so
    every run was reported as "MISSING" them. The reference has to be the same KIND of
    result as the thing being checked, or the check reports a difference in kind as a
    defect. Pipeline-check output is excluded too: it is never a reference for anything.
    """
    for f in sorted(RESULTS.glob("*.json")):
        if f.name.startswith("pipe_") or "_longhorizon" in f.name:
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if {"episodes", "aggregate", "trajectory"} <= set(d) and d.get("profile_set") == "test":
            return d
    return None


# --------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pemwe.evaluate")
    ap.add_argument("--policy", default=None,
                    help="baseline_naive | baseline_ramplimited | sac | ppo")
    ap.add_argument("--model", default=None, help="SB3 checkpoint for a learned policy")
    ap.add_argument("--all-baselines", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="every checkpoint in models/")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test",
                    help="test (default) | train | archetype_<name>")
    ap.add_argument("--source", default="hybrid", choices=["hybrid", "pv", "wind"])
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--validate", action="store_true", default=True)
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args(argv)

    cfg = load_config(overrides=args.override)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    jobs = []
    if args.all_baselines:
        for name, cls in BASELINES.items():
            jobs.append((name, cls(cfg), args.run_id or name, args.seed))
    elif args.sweep:
        for m in sorted(MODELS.glob("*.zip")):
            meta_f = m.with_suffix(".json")
            meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
            algo = meta.get("algo")
            seed = meta.get("seed", args.seed)
            ov = meta.get("overrides", [])
            c = load_config(overrides=ov) if ov else cfg
            jobs.append((algo or ("sac" if "sac" in m.name else "ppo"),
                         SB3Policy(m, algo), m.stem, seed, c))
    elif args.model:
        name = args.policy or ("sac" if "sac" in Path(args.model).name.lower() else "ppo")
        jobs.append((name, SB3Policy(args.model, name),
                     args.run_id or Path(args.model).stem, args.seed))
    elif args.policy in BASELINES:
        jobs.append((args.policy, BASELINES[args.policy](cfg),
                     args.run_id or args.policy, args.seed))
    else:
        ap.error("give --policy <baseline>, --model <ckpt>, --all-baselines or --sweep")

    ref = _reference()
    if args.validate and ref is None:
        print("note: no reference results found; run scripts/fake_results.py to enable "
              "the cross-check against C's schema", file=sys.stderr)

    print(f"{'run_id':<30}{'H2 kg/day':>11}{'uV/h':>9}{'life yr':>9}"
          f"{'cycles':>8}{'split':>8}")
    print("-" * 75)
    ok = True
    for job in jobs:
        name, pol, run_id, seed = job[:4]
        c = job[4] if len(job) > 4 else cfg
        res = evaluate(c, pol, name, run_id, seed=seed, split=args.split,
                       source=args.source, max_days=args.max_days)
        problems = validate(res, ref) if args.validate else []
        if problems:
            ok = False
            print(f"  SCHEMA PROBLEMS in {run_id}:", file=sys.stderr)
            for p in problems:
                print(f"    {p}", file=sys.stderr)

        (out / f"{run_id}.json").write_text(json.dumps(res))
        a = res["aggregate"]
        cyc = float(np.mean([e["n_cycles"] for e in res["episodes"]]))
        print(f"{run_id:<30}{a['h2_kg_mean']:11.1f}{a['deg_rate_uv_per_h']:9.2f}"
              f"{a['projected_life_years']:9.2f}{cyc:8.1f}{res['profile_set']:>8}")

    print("-" * 75)
    print(f"wrote {len(jobs)} result files to {out}")
    if args.validate:
        print(f"schema check vs CONTRACTS.md 3"
              f"{' and scripts/fake_results.py' if ref else ''}: "
              f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
