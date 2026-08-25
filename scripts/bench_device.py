"""Measure where the training bottleneck actually is: env stepping, or the network.

SB3 with an MlpPolicy over a 1-D action space is usually bound by Python env stepping,
not by matmuls -- so a big GPU can be no faster, or slower, than CPU because of
host-to-device transfer overhead on tiny batches. This script measures it instead of
assuming, on whatever machine you run it on.

    python scripts/bench_device.py            # full sweep
    python scripts/bench_device.py --quick    # fewer steps

Report the numbers to the team before anyone commits an overnight run to a device.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from pemwe import load_config, PEMWEEnv

CFG = load_config()


def make_env():
    return lambda: PEMWEEnv(CFG)


def bench_raw_env(n_steps=20000):
    """Pure environment throughput -- the ceiling nothing else can exceed."""
    env = PEMWEEnv(CFG)
    env.reset(seed=0)
    a = np.array([0.3], dtype=np.float32)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        _, _, _, trunc, _ = env.step(a)
        if trunc:
            env.reset()
    return n_steps / (time.perf_counter() - t0)


def bench_algo(algo, device, n_envs, total, vec=SubprocVecEnv):
    venv = (DummyVecEnv if n_envs == 1 else vec)([make_env() for _ in range(n_envs)])
    kw = dict(policy="MlpPolicy", env=venv, device=device, verbose=0, seed=0)
    model = (PPO(n_steps=256, batch_size=256, **kw) if algo is PPO
             else SAC(learning_starts=500, batch_size=256, train_freq=1, **kw))
    model.learn(total_timesteps=256 * n_envs)          # warm up / compile
    t0 = time.perf_counter()
    model.learn(total_timesteps=total, reset_num_timesteps=False)
    dt = time.perf_counter() - t0
    venv.close()
    return total / dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    ppo_steps = 20000 if not args.quick else 8000
    sac_steps = 4000 if not args.quick else 2000

    cuda = torch.cuda.is_available()
    print(f"torch {torch.__version__}   cuda_available={cuda}"
          + (f"   {torch.cuda.get_device_name(0)}" if cuda else ""))
    import os
    print(f"cpu cores visible: {os.cpu_count()}\n")

    raw = bench_raw_env()
    print(f"{'raw env stepping (no NN)':<34} {raw:>10,.0f} steps/s   <-- hard ceiling\n")

    rows = []
    for n_envs in (1, 8, 32, 64):
        rows.append((f"PPO  cpu   n_envs={n_envs:<3}", bench_algo(PPO, "cpu", n_envs, ppo_steps)))
    if cuda:
        for n_envs in (8, 32, 64):
            rows.append((f"PPO  cuda  n_envs={n_envs:<3}", bench_algo(PPO, "cuda", n_envs, ppo_steps)))
    rows.append(("SAC  cpu   n_envs=1  ", bench_algo(SAC, "cpu", 1, sac_steps)))
    if cuda:
        rows.append(("SAC  cuda  n_envs=1  ", bench_algo(SAC, "cuda", 1, sac_steps)))

    best = max(r[1] for r in rows)
    print(f"{'config':<34} {'throughput':>12}   {'vs best':>8}")
    print("-" * 58)
    for name, sps in rows:
        print(f"{name:<34} {sps:>8,.0f} steps/s   {sps/best:>7.0%}")

    print(f"\n2M steps at best rate: {2e6/best/60:.1f} min per seed")


if __name__ == "__main__":
    main()
