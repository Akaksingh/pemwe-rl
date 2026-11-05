"""HPC diagnostic: why does per-worker simulator throughput fall between 2 and 4 workers?

    OMP_NUM_THREADS=1 python scripts/bench_hpc_diag.py [--reps 5] [--out results_hpc/hpc_diag.csv]

bench_hpc_env.py showed efficiency ~1.0 at 2 workers but ~0.75 at 4 and 8, with or without
pinning to physical cores, so OS placement is not the cause. Candidate causes, separated here:

  * LOAD IMBALANCE   workers draw different days; step cost depends on on/off state. Test: run
                     `same_work` (every worker simulates exactly the same episodes) against
                     `different_work`. If the drop persists with identical work, it is not imbalance.
  * CLOCK FREQUENCY  fewer active cores may run at a higher turbo bin. Test: read each worker's
                     scaling_cur_freq while it is under load.
  * CONTENTION       shared cache / memory bandwidth. If per-worker rate (steps_i / t_i) falls with n
                     while the frequency does not, contention is the remaining candidate; this
                     script cannot prove it, only exclude the other two.

All workers are pinned to distinct physical cores. Each reports its own elapsed time, so the
slowest-to-fastest spread within a case is visible. Linux only.
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                    # noqa: E402

import bench_hpc_env as B                             # noqa: E402  (reuses topology helpers)
from pemwe.config import load_config                  # noqa: E402
from pemwe import train as T                          # noqa: E402
from pemwe.env import PEMWEEnv                        # noqa: E402

CFG = PROF = None


def _freq_mhz(cpu: int) -> float:
    try:
        return int(open(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq").read()) / 1000.0
    except Exception:
        return float("nan")


def worker(idx, cpu, n_eps, seed, barrier, q):
    os.sched_setaffinity(0, [cpu])
    env = PEMWEEnv(CFG, profiles=PROF, seed=seed)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    for _ in range(100):
        env.step(rng.uniform(-1, 1, size=1).astype(np.float32))
    barrier.wait()
    t0 = time.perf_counter()
    steps, f_mid = 0, float("nan")
    for e in range(n_eps):
        env.reset(seed=seed * 1000 + e)
        for k in range(env.n_steps):
            env.step(rng.uniform(-1, 1, size=1).astype(np.float32))
            steps += 1
        if e == n_eps // 2:
            f_mid = _freq_mhz(cpu)                    # sampled while the core is busy
    q.put((idx, steps, time.perf_counter() - t0, f_mid))


def run_case(n, same_work, cores, total_eps):
    eps = B._split(total_eps, n) if not same_work else [total_eps // 8] * n    # same_work: 8 eps each
    ctx = mp.get_context("fork")
    barrier, q = ctx.Barrier(n + 1), ctx.Queue()
    seeds = [1000] * n if same_work else [1000 + i for i in range(n)]
    ps = [ctx.Process(target=worker, args=(i, cores[i][0], eps[i], seeds[i], barrier, q)) for i in range(n)]
    for p in ps:
        p.start()
    barrier.wait()
    res = [q.get() for _ in ps]
    for p in ps:
        p.join()
    return sorted(res)


def main():
    global CFG, PROF
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "results_hpc" / "hpc_diag.csv"))
    a = ap.parse_args()
    CFG = load_config()
    PROF = T.train_profiles(CFG, None)
    cores = B.physical_cores()

    def rd(p):
        try:
            return open(p).read().strip()
        except Exception:
            return "n/a"
    print("governor:", rd("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
          "| driver:", rd("/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver"),
          "| no_turbo:", rd("/sys/devices/system/cpu/intel_pstate/no_turbo"),
          "| max MHz:", float(rd("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")) / 1000
          if rd("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq") != "n/a" else "n/a", flush=True)

    rows = []
    print(f"{'mode':15s}{'n':>3} {'mean worker steps/s':>20}{'slow/fast time':>16}{'mean MHz':>10}", flush=True)
    for same in (False, True):
        for n in (1, 2, 3, 4, 5, 6, 8):
            per_rate, spread, freqs = [], [], []
            for rep in range(a.reps):
                res = run_case(n, same, cores, total_eps=64)
                times = [r[2] for r in res]
                for idx, steps, t, f in res:
                    rows.append(["same_work" if same else "different_work", n, rep, idx, steps, t,
                                 steps / t, f])
                    per_rate.append(steps / t)
                    freqs.append(f)
                spread.append(max(times) / min(times))
            print(f"{'same_work' if same else 'different_work':15s}{n:3d} {np.mean(per_rate):20.0f}"
                  f"{np.mean(spread):16.3f}{np.nanmean(freqs):10.0f}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "n_workers", "repeat", "worker", "steps", "elapsed_s", "steps_per_s_worker", "freq_mhz"])
        w.writerows(rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
