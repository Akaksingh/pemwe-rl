"""SAC is the throughput bottleneck. Find out whether it is affordable.

Default SB3 SAC does one gradient update per environment step with a single env, which is
why it benchmarks ~34x slower than PPO. Batching the updates (train_freq=N with
gradient_steps=N across N parallel envs) keeps the same update-to-step ratio while paying
the Python/kernel-launch overhead once instead of N times.

Run this before deciding how much of the experiment matrix SAC gets.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from pemwe import load_config, PEMWEEnv

CFG = load_config()
STEPS = 6000


def run(device, n_envs, train_freq, grad_steps):
    venv = (DummyVecEnv if n_envs == 1 else SubprocVecEnv)(
        [lambda: PEMWEEnv(CFG) for _ in range(n_envs)])
    m = SAC("MlpPolicy", venv, device=device, verbose=0, seed=0,
            learning_starts=500, batch_size=512,
            train_freq=(train_freq, "step"), gradient_steps=grad_steps)
    m.learn(total_timesteps=1000)                      # warm up
    t0 = time.perf_counter()
    m.learn(total_timesteps=STEPS, reset_num_timesteps=False)
    dt = time.perf_counter() - t0
    venv.close()
    return STEPS / dt


def main():
    print(f"cuda={torch.cuda.is_available()}  "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}\n")
    cfgs = [
        ("cuda", 1, 1, 1), ("cpu", 1, 1, 1),
        ("cuda", 8, 8, 8), ("cpu", 8, 8, 8),
        ("cuda", 32, 32, 32), ("cpu", 32, 32, 32),
        ("cuda", 32, 32, 8),
    ]
    rows = []
    for dev, ne, tf, gs in cfgs:
        try:
            sps = run(dev, ne, tf, gs)
        except Exception as e:
            print(f"  {dev} n_envs={ne} failed: {type(e).__name__}: {e}")
            continue
        rows.append((f"SAC {dev:<4} n_envs={ne:<3} train_freq={tf:<3} grad_steps={gs:<3}", sps))
        print(f"  {rows[-1][0]}  {sps:>8,.0f} steps/s")

    best_name, best = max(rows, key=lambda r: r[1])
    print(f"\nbest: {best_name.strip()}  {best:,.0f} steps/s")
    print(f"1M steps at that rate: {1e6/best/60:.1f} min per seed")
    print(f"20-run SAC matrix (5 seeds x 4 weights), 4 GPUs in parallel: "
          f"{20*1e6/best/3600/4:.1f} h wall-clock, {20*1e6/best/3600:.1f} GPU-h of quota")


if __name__ == "__main__":
    main()
