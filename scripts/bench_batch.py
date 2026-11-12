"""Benchmark: episode-parallel BatchSim against the scalar environment (single process and a process pool).

    python scripts/bench_batch.py [--real] [--workers N] [--out results_batch/bench_batch.csv]

All timings are SAME-MACHINE ratios: the scalar environment and the batched simulator run back to back on the
same hardware in the same session, so the speedup does not depend on the machine. Absolute rates describe only the
machine named in the metadata file written next to the CSV.

  --real   use the 72 held-out days (needs pyarrow for the parquet profile file); default: synthetic days, which
           exercise the same code path (speed does not depend on the data).

Rows: scalar_1proc (one process), scalar_pool (N processes, one episode per task), batched_B (one process, B episodes
in lockstep), and `grid` rows for the 72-point hysteresis rule grid over all days, scalar pool against one batch.
A validation block compares a random subset of the grid episodes against the scalar environment.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import multiprocessing as mp
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                                   # noqa: E402

import rule_family_v2 as rf                          # noqa: E402

N_STEPS = 1440


def make_profiles(real: bool, n_days: int):
    from pemwe import load_config
    from pemwe.env import PEMWEEnv, synthetic_day
    cfg = load_config()
    if real:
        from pemwe import profiles
        df = profiles.load_profiles(ROOT / cfg["data"]["profile_path"])
        return cfg, profiles.profiles_array(df, profiles.SPLITS["test"][:n_days], "hybrid")
    p_rated = PEMWEEnv(cfg).p_rated
    rng = np.random.default_rng(0)
    return cfg, np.stack([synthetic_day(rng, N_STEPS, p_rated, c) for c in np.linspace(0.05, 0.6, n_days)])


def _scalar_chunk(args):
    """Run (params, day_index) episodes with the scalar environment; return the metrics. Top-level: spawn-safe."""
    chunk, profiles = args                    # chunk: list of (index, params, day)
    from pemwe import load_config
    from pemwe.env import PEMWEEnv
    from pemwe.evaluate import rollout_episode
    cfg = load_config()
    env = PEMWEEnv(cfg, profiles=profiles, seed=0)
    out = []
    for idx, params, d in chunk:
        ep, _ = rollout_episode(env, rf.HysteresisRule(*params), d, 0)
        out.append((idx, ep["h2_kg"], ep["dv_deg_uv"]))
    return out


def time_scalar_inproc(episodes, profiles):
    """Scalar environment, this process, environment already built; only the episodes are timed."""
    from pemwe import load_config
    from pemwe.env import PEMWEEnv
    from pemwe.evaluate import rollout_episode
    env = PEMWEEnv(load_config(), profiles=profiles, seed=0)
    rollout_episode(env, rf.HysteresisRule(*episodes[0][1]), episodes[0][2], 0)       # warm-up
    t0 = time.perf_counter()
    for _, params, d in episodes:
        rollout_episode(env, rf.HysteresisRule(*params), d, 0)
    return time.perf_counter() - t0


def time_scalar_pool(episodes, profiles, workers):
    chunks = [episodes[i::workers * 4] for i in range(workers * 4)]
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(workers) as pool:
        res = pool.map(_scalar_chunk, [(c, profiles) for c in chunks if c])
    return time.perf_counter() - t0, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--days", type=int, default=72)
    ap.add_argument("--workers", type=int, default=0, help="scalar pool size (default: all logical CPUs)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--sizes", default="1,8,72,512,2048,5184,16384")
    ap.add_argument("--out", default=str(ROOT / "results_batch" / "bench_batch.csv"))
    a = ap.parse_args()
    import os
    workers = a.workers or (os.cpu_count() or 2)

    from pemwe.batch import BatchSim, BatchHysteresis
    cfg, prof = make_profiles(a.real, a.days)
    D = len(prof)
    grid = list(itertools.product(rf.P_ON, rf.K_UP, rf.K_DOWN, rf.SLEW, rf.CAP))      # 72 rules
    rows = []

    def add(method, size, episodes, seconds, note=""):
        steps = episodes * N_STEPS
        rows.append(dict(method=method, size=size, episodes=episodes, env_steps=steps, seconds=round(seconds, 4),
                         steps_per_s=round(steps / seconds, 1), note=note))
        print(f"{method:14s} size={size!s:>6}  episodes={episodes:>6}  {seconds:8.3f}s  {steps / seconds:12,.0f} steps/s  {note}", flush=True)

    # ---- scalar, one process (a fixed sample of episodes, best of repeats)
    sample = [(i, grid[i % len(grid)], i % D) for i in range(12)]
    ts = [time_scalar_inproc(sample, prof) for _ in range(a.repeats)]
    add("scalar_1proc", 1, len(sample), float(np.median(ts)), f"in-process, median of {a.repeats}; min {min(ts):.2f}s max {max(ts):.2f}s")
    # ---- scalar pool over the whole grid x days
    episodes = [(gi * D + d, g, d) for gi, g in enumerate(grid) for d in range(D)]
    t_pool, _ = time_scalar_pool(episodes, prof, workers)
    add("grid_scalar_pool", workers, len(episodes), t_pool, f"{len(grid)} rules x {D} days, {workers} spawn-pool workers")

    # ---- batched, one process, several batch sizes (parameters random per episode)
    rng = np.random.default_rng(1)
    for B in [int(s) for s in a.sizes.split(",")]:
        big = prof[np.arange(B) % D]
        idx = rng.integers(len(grid), size=B)
        cols = [np.array([grid[i][k] for i in idx]) for k in range(5)]
        sim = BatchSim(cfg, big)
        times = []
        for _ in range(a.repeats):
            t0 = time.perf_counter()
            sim.rollout(BatchHysteresis(*cols))
            times.append(time.perf_counter() - t0)
        add("batched", B, B, float(np.median(times)), f"median of {a.repeats}; min {min(times):.2f}s max {max(times):.2f}s")

    # ---- the whole grid in one batch
    cols = [np.repeat([g[k] for g in grid], D) for k in range(5)]
    big = np.tile(prof, (len(grid), 1))
    sim = BatchSim(cfg, big)
    times = []
    for _ in range(a.repeats):
        t0 = time.perf_counter()
        res = sim.rollout(BatchHysteresis(*cols))
        times.append(time.perf_counter() - t0)
    add("grid_batched", len(big), len(big), float(np.median(times)),
        f"{len(grid)} rules x {D} days, one batch, median of {a.repeats}; min {min(times):.2f}s max {max(times):.2f}s")

    # ---- validation: random subset of the grid episodes against the scalar environment (every metric pair)
    sel = rng.choice(len(big), size=min(24, len(big)), replace=False)
    eps = [(int(i), grid[i // D], int(i % D)) for i in sel]
    _, scalar_res = time_scalar_pool(eps, prof, min(workers, 4))
    got = {idx: (h2, dv) for chunk_res in scalar_res for idx, h2, dv in chunk_res}
    h2_s = np.array([got[int(i)][0] for i in sel]); dv_s = np.array([got[int(i)][1] for i in sel])
    h2_b, dv_b = res["h2_kg"][sel], res["dv_deg_uv"][sel]
    err = float(max(np.max(np.abs(h2_b - h2_s) / np.abs(h2_s)), np.max(np.abs(dv_b - dv_s) / np.abs(dv_s))))
    print(f"validation: {len(sel)} random grid episodes, max relative difference batched vs scalar = {err:.2e}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    meta = dict(platform=platform.platform(), processor=platform.processor(), python=sys.version.split()[0],
                numpy=np.__version__, logical_cpus=os.cpu_count(), scalar_pool_workers=workers, real_profiles=a.real,
                days=D, grid_rules=len(grid), validation_episodes=int(len(sel)), validation_max_rel_diff=err)
    Path(a.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
