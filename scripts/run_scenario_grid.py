"""Parallel evaluation grid: controllers x renewable scenarios x degradation-coefficient variants.

    OMP_NUM_THREADS=1 python scripts/run_scenario_grid.py --workers 12 \
        [--models models/a.zip models/b.zip] [--split test] [--out results_hpc/controller_grid_days.csv]

Experiments A (degradation-parameter sensitivity) and B (renewable variability), plus the
embarrassingly parallel workload for the HPC section. Every (controller, scenario, variant)
combination is one independent unit run in a process pool; results are written one row per
(unit, day), so downstream statistics can pair across days and bootstrap over days/realisations
without re-running anything.

Controllers
  naive_as_published   load-following rule with the 10% idle threshold (ref. as cited in the paper)
  no_threshold         the same rule with the threshold removed (= constant action +1)
  ramp_limited         the tuned slew-limited baseline
  <checkpoint stems>   any SB3 checkpoints given with --models (trained at NOMINAL coefficients on
                       the ORIGINAL profile; evaluating them elsewhere is a robustness/shift test,
                       not retraining)

Degradation-coefficient variants (Experiment A): nominal; each of the five coefficients at the
lower and upper end of the interval recorded in notes/A_calibration.md (one at a time); all-low
(optimistic) and all-high (pessimistic). The intervals are the repository's recorded literature
ranges and have NOT been independently verified.

Scenario set (Experiment B): files from scripts/make_scenarios.py. The coefficient sweep is run on
the original profile only, the scenario sweep at nominal coefficients only (one-factor-at-a-time).
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pemwe.config import load_config                  # noqa: E402

# (config key, nominal, low, high) -- intervals from notes/A_calibration.md
COEFS = [("degradation.r_base_uv_per_h", 1.75, 1.0, 10.0),
         ("degradation.k_j_uv_per_h", 20.0, 5.0, 60.0),
         ("degradation.k_ramp_uv_per_h", 31.0, 0.5, 40.0),
         ("degradation.dv_cycle_uv", 1.0, 0.05, 5.0),
         ("degradation.r_idle_uv_per_h", 0.75, 0.5, 10.0)]
SHORT = {"r_base_uv_per_h": "base", "k_j_uv_per_h": "stress", "k_ramp_uv_per_h": "ramp",
         "dv_cycle_uv": "cycle", "r_idle_uv_per_h": "idle"}

SCEN_NAMES = ("original", "low_variability", "high_solar", "high_wind", "combined_high")
ORIGINAL_PROFILE = "data/processed/kutch_2019_1min.parquet"
FIELDS = ["controller", "scenario", "realisation", "variant", "split", "date", "h2_kg", "dv_deg_uv",
          "n_cycles", "curtailed_kwh", "mean_eta_lhv", "mean_abs_ramp", "reward_total",
          "r_yield", "r_deg", "r_ramp"]


def variants():
    v = [("nominal", [])]
    for key, _nom, lo, hi in COEFS:
        s = SHORT[key.split(".")[-1]]
        v.append((f"{s}_low", [f"{key}={lo}"]))
        v.append((f"{s}_high", [f"{key}={hi}"]))
    v.append(("all_low", [f"{k}={lo}" for k, _n, lo, _h in COEFS]))
    v.append(("all_high", [f"{k}={hi}" for k, _n, _l, hi in COEFS]))
    return v


_CACHE: dict = {}


def run_unit(unit):
    from pemwe.env import PEMWEEnv
    from pemwe.baselines import BASELINES
    from pemwe.evaluate import rollout_episode, SB3Policy
    from pemwe import profiles

    ctrl, model_path, scenario, real, profile_path, vname, overrides, split = unit
    cfg = load_config(overrides=overrides + [f"data.profile_path={profile_path}"])
    key = profile_path
    if key not in _CACHE:
        _CACHE[key] = profiles.load_profiles(ROOT / profile_path)
    df = _CACHE[key]
    dates = profiles.SPLITS[split]
    prof = profiles.profiles_array(df, dates, "hybrid")
    env = PEMWEEnv(cfg, profiles=prof, seed=0)

    if model_path:
        pol = SB3Policy(model_path, algo="sac")        # all checkpoints here are SAC; do not infer from the name
    elif ctrl == "no_threshold":
        pol = BASELINES["baseline_naive"](load_config(overrides=overrides + ["baseline.p_idle_threshold_frac=0.0"]))
    elif ctrl == "naive_as_published":
        pol = BASELINES["baseline_naive"](cfg)
    elif ctrl == "ramp_limited":
        pol = BASELINES["baseline_ramplimited"](cfg)
    else:
        raise ValueError(ctrl)

    rows = []
    for i, date in enumerate(dates):
        ep, _ = rollout_episode(env, pol, i, 0)
        rows.append([ctrl, scenario, real, vname, split, date, ep["h2_kg"], ep["dv_deg_uv"],
                     ep["n_cycles"], ep["curtailed_kwh"], ep["mean_eta_lhv"], ep["mean_abs_ramp"],
                     ep["reward_total"], ep["r_yield"], ep["r_deg"], ep["r_ramp"]])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--realisations", type=int, default=3)
    ap.add_argument("--scenarios-dir", default="data/scenarios")
    ap.add_argument("--grid", default="both", choices=["both", "coefs", "scenarios", "nominal"],
                    help="nominal = original profile, nominal coefficients only (quick comparisons)")
    ap.add_argument("--out", default=str(ROOT / "results_hpc" / "controller_grid_days.csv"))
    ap.add_argument("--limit-days", type=int, default=0, help="smoke test: use fewer days")
    ap.add_argument("--day-stride", type=int, default=1,
                    help="use every k-th day of the (date-sorted) split: a cheap subset that still spans the year")
    a = ap.parse_args()

    ctrls = [("naive_as_published", None), ("no_threshold", None), ("ramp_limited", None)]
    ctrls += [(Path(m).stem, m) for m in a.models]

    units = []
    if a.grid == "nominal":
        for c, m in ctrls:
            units.append((c, m, "original", 0, ORIGINAL_PROFILE, "nominal", [], a.split))
    if a.grid in ("both", "coefs"):
        for vname, ov in variants():
            for c, m in ctrls:
                units.append((c, m, "original", 0, ORIGINAL_PROFILE, vname, ov, a.split))
    if a.grid in ("both", "scenarios"):
        for s in SCEN_NAMES:
            for r in range(a.realisations):
                p = f"{a.scenarios_dir}/kutch_2019_1min__{s}__r{r}.parquet"
                if not (ROOT / p).exists():
                    print("missing scenario file", p, "-- run scripts/make_scenarios.py", file=sys.stderr)
                    continue
                for c, m in ctrls:
                    units.append((c, m, s, r, p, "nominal", [], a.split))
    from pemwe import profiles
    if a.day_stride > 1:
        profiles.SPLITS[a.split] = profiles.SPLITS[a.split][::a.day_stride]
    if a.limit_days:
        profiles.SPLITS[a.split] = profiles.SPLITS[a.split][:a.limit_days]

    print(f"{len(units)} units, {a.workers} workers", flush=True)
    t0 = time.perf_counter()
    n_steps = 0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f, mp.get_context("fork").Pool(a.workers) as pool:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for k, rows in enumerate(pool.imap_unordered(run_unit, units, chunksize=1), 1):
            w.writerows(rows)
            n_steps += len(rows) * 1440
            if k % 10 == 0 or k == len(units):
                el = time.perf_counter() - t0
                print(f"  {k}/{len(units)} units  {el:6.0f}s  {n_steps / el:9.0f} env-steps/s aggregate", flush=True)
    el = time.perf_counter() - t0
    print(f"done: {len(units)} units, {n_steps:,} env-steps in {el:.1f}s "
          f"({n_steps / el:,.0f} steps/s; {n_steps * 60 / el / 86400:.1f} simulated days per wall-second)")
    print("wrote", a.out)
    import json
    Path(a.out).with_suffix(".meta.json").write_text(json.dumps({
        "units": len(units), "env_steps": n_steps, "wall_s": round(el, 1), "workers": a.workers,
        "steps_per_s": round(n_steps / el), "sim_days_per_wall_s": round(n_steps * 60 / el / 86400, 1),
        "split": a.split, "grid": a.grid, "models": a.models,
        "note": "machine was shared with other jobs while this ran"}, indent=2))


if __name__ == "__main__":
    main()
