"""HPC benchmark 1: simulation throughput and parallel scaling (CPU only).

    OMP_NUM_THREADS=1 python scripts/bench_hpc_env.py [--reps 5] [--out results_hpc/hpc_env.csv]

Measures raw PEMWEEnv stepping (random actions, no policy, no SB3) in worker processes, so the
numbers isolate the simulator from the learner:

  * strong scaling : fixed total work (64 episodes = 92,160 steps) split over n workers
                     -> speedup S(n) = T(1)/T(n), efficiency E(n) = S(n)/n
  * weak scaling   : fixed work per worker (8 episodes) -> per-worker throughput vs n
  * placement      : `unpinned` (OS decides), `physical` (one worker per physical core),
                     `smt_pairs` (workers packed onto hyper-thread sibling pairs). This separates
                     "the code does not scale" from "the OS put two workers on one core".
  * SB3 vector envs: DummyVecEnv (one process) vs SubprocVecEnv, random actions.

All workers wait on a barrier after their warm-up, so process start-up is excluded from timing.
Each case is repeated; the CSV keeps every repeat, so confidence intervals come from raw data.
Linux only (uses sched_setaffinity and sysfs topology).
"""
from __future__ import annotations

import argparse
import csv
import glob
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                    # noqa: E402

from pemwe.config import load_config                  # noqa: E402
from pemwe import train as T                          # noqa: E402
from pemwe.env import PEMWEEnv                        # noqa: E402

COLS = ["experiment", "backend", "n_workers", "repeat", "steps", "wall_s", "steps_per_s", "note"]
CFG = None
PROF = None


def _expand(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def physical_cores() -> list[list[int]]:
    """One list of logical CPUs per physical core, from sysfs."""
    paths = glob.glob("/sys/devices/system/cpu/cpu[0-9]*/topology/thread_siblings_list")
    paths.sort(key=lambda p: int(p.split("/cpu")[-1].split("/")[0]))
    seen, cores = set(), []
    for p in paths:
        txt = open(p).read().strip()
        if txt not in seen:
            seen.add(txt)
            cores.append(_expand(txt))
    return cores


def _placement(mode: str, n: int, cores: list[list[int]]):
    if mode == "unpinned":
        return [None] * n
    if mode == "physical":
        return [[cores[i][0]] for i in range(n)] if n <= len(cores) else None
    if mode == "smt_pairs":
        if any(len(c) < 2 for c in cores) or n > 2 * len(cores):
            return None
        return [[cores[i // 2][i % 2]] for i in range(n)]
    raise ValueError(mode)


def _split(total: int, n: int) -> list[int]:
    base, extra = divmod(total, n)
    return [base + (1 if i < extra else 0) for i in range(n)]


def worker(cpus, n_eps, seed, barrier, q):
    if cpus:
        os.sched_setaffinity(0, cpus)
    env = PEMWEEnv(CFG, profiles=PROF, seed=seed)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    for _ in range(100):                               # warm-up, excluded from timing
        env.step(rng.uniform(-1, 1, size=1).astype(np.float32))
    barrier.wait()
    steps = 0
    for e in range(n_eps):
        env.reset(seed=seed * 1000 + e)
        for _ in range(env.n_steps):
            env.step(rng.uniform(-1, 1, size=1).astype(np.float32))
            steps += 1
    q.put((steps, time.perf_counter()))                # CLOCK_MONOTONIC: comparable across processes


def run_case(scaling, mode, n, cores, total_eps=64, weak_eps=8):
    placement = _placement(mode, n, cores)
    if placement is None:
        return None
    eps = _split(total_eps, n) if scaling == "strong" else [weak_eps] * n
    ctx = mp.get_context("fork")
    barrier, q = ctx.Barrier(n + 1), ctx.Queue()
    procs = [ctx.Process(target=worker, args=(placement[i], eps[i], 1000 + i, barrier, q))
             for i in range(n)]
    for p in procs:
        p.start()
    barrier.wait()
    t0 = time.perf_counter()
    res = [q.get() for _ in procs]
    for p in procs:
        p.join()
    return sum(r[0] for r in res), max(r[1] for r in res) - t0


def main():
    global CFG, PROF
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "results_hpc" / "hpc_env.csv"))
    ap.add_argument("--quick", action="store_true", help="1 rep, fewer points (smoke test)")
    a = ap.parse_args()

    CFG = load_config()
    PROF = T.train_profiles(CFG, None)
    cores = physical_cores()
    reps = 1 if a.quick else a.reps
    print(f"logical CPUs={os.cpu_count()} physical cores={len(cores)} topology={cores}", flush=True)

    rows = []
    plan = [("strong", "unpinned", (1, 2, 4, 8, 12, 16)),
            ("strong", "physical", (1, 2, 4, 8)),
            ("strong", "smt_pairs", (2, 4, 8, 16)),
            ("weak", "unpinned", (1, 2, 4, 8, 16)),
            ("weak", "physical", (1, 2, 4, 8))]
    if a.quick:
        plan = [("strong", "unpinned", (1, 2)), ("strong", "physical", (2,))]
    for scaling, mode, ns in plan:
        for n in ns:
            for rep in range(reps):
                out = run_case(scaling, mode, n, cores)
                if out is None:
                    break
                steps, wall = out
                rows.append([f"raw_env_{scaling}_scaling", mode, n, rep, steps, wall, steps / wall, ""])
                print(f"{scaling:6s} {mode:9s} n={n:2d} rep={rep} {steps:7d} steps {wall:6.2f}s "
                      f"{steps / wall:9.0f} steps/s", flush=True)

    print("== SB3 vector-env overhead, random actions, 24,000 transitions ==", flush=True)
    for backend in ("dummy", "subproc"):
        for n in ((1, 4) if a.quick else (1, 2, 4, 8, 16, 32)):
            if backend == "subproc" and n == 1:
                continue
            for rep in range(reps):
                venv = T.make_vec_env(CFG, PROF, n, seed=0, subproc=(backend == "subproc"))
                venv.reset()
                nvec = 24000 // n
                acts = np.random.default_rng(rep).uniform(-1, 1, size=(nvec, n, 1)).astype(np.float32)
                t0 = time.perf_counter()
                for k in range(nvec):
                    venv.step(acts[k])
                dt = time.perf_counter() - t0
                venv.close()
                rows.append(["sb3_vecenv_random_policy", backend, n, rep, nvec * n, dt, nvec * n / dt, ""])
                print(f"{backend:8s} n_envs={n:2d} rep={rep} {nvec * n / dt:9.0f} steps/s", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        w.writerows(rows)
    print("wrote", a.out, len(rows), "rows")


if __name__ == "__main__":
    main()
