import sys, time, csv, os, multiprocessing as mp
sys.path.insert(0, "src")
import numpy as np
from pemwe.config import load_config
from pemwe import train as T
from pemwe.env import PEMWEEnv

cfg = load_config()
prof = T.train_profiles(cfg, None)
OUT = "results_hpc/hpc_scaling.csv"
rows = []

def work(args):
    seed, n_eps = args
    env = PEMWEEnv(cfg, profiles=prof, seed=seed)
    rng = np.random.default_rng(seed)
    steps = 0
    for e in range(n_eps):
        env.reset(seed=seed * 1000 + e)
        for _ in range(env.n_steps):
            env.step(rng.uniform(-1, 1, size=1).astype(np.float32)); steps += 1
    return steps

TOTAL_EPS = 64                      # fixed total work: 64 episodes x 1440 steps = 92,160 steps
def split(n):
    base, extra = divmod(TOTAL_EPS, n)
    return [base + (1 if i < extra else 0) for i in range(n)]

print(f"cpu_count={os.cpu_count()}", flush=True)
print("== strong scaling: fixed 64 episodes (92,160 steps) split across n worker processes ==", flush=True)
for n in (1, 2, 4, 8, 12, 16):
    for rep in range(3):
        with mp.get_context("fork").Pool(n) as pool:
            pool.map(work, [(900 + i, 1) for i in range(n)])          # warm-up, excluded from timing
            t0 = time.perf_counter()
            steps = sum(pool.map(work, [(i, k) for i, k in enumerate(split(n))]))
            dt = time.perf_counter() - t0
        rows.append(["raw_env_strong_scaling", "multiprocessing", n, rep, steps, dt, steps / dt, ""])
        print(f"n={n:2d} rep={rep} {steps} steps {dt:6.2f}s {steps/dt:9.0f} steps/s", flush=True)

print("== SB3 vector-env overhead, random actions, 24,000 transitions ==", flush=True)
for backend in ("dummy", "subproc"):
    for n in (1, 2, 4, 8, 16, 32):
        if backend == "subproc" and n == 1: continue
        for rep in range(3):
            venv = T.make_vec_env(cfg, prof, n, seed=0, subproc=(backend == "subproc"))
            venv.reset()
            nvec = 24000 // n
            acts = np.random.default_rng(rep).uniform(-1, 1, size=(nvec, n, 1)).astype(np.float32)
            t0 = time.perf_counter()
            for k in range(nvec):
                venv.step(acts[k])
            dt = time.perf_counter() - t0
            venv.close()
            rows.append(["sb3_vecenv_random_policy", backend, n, rep, nvec * n, dt, nvec * n / dt, ""])
            print(f"{backend:8s} n_envs={n:2d} rep={rep} {nvec*n/dt:9.0f} steps/s", flush=True)

new = not os.path.exists(OUT)
with open(OUT, "a", newline="") as f:
    w = csv.writer(f)
    if new: w.writerow(["experiment", "backend", "n_workers", "repeat", "steps", "wall_s", "steps_per_s", "note"])
    w.writerows(rows)
print("wrote", OUT)
