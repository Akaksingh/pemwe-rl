# Measured throughput — `hcd-dept-h200`

Run on 4× NVIDIA H200 / AMD EPYC 9535 (128 cores), torch 2.13.0+cu130, SB3 2.9.0.
Reproduce with `scripts/bench_device.py` and `scripts/bench_sac.py`.

**Read this before committing an overnight run to a device.** Two of the results are
counter-intuitive and both cost real time if you guess instead of measuring.

---

## PPO — the GPU makes it *slower*

```
raw env stepping (no NN)          8,225 steps/s   <-- hard ceiling, single process

PPO  cpu   n_envs=32              9,712 steps/s   100%   <-- use this
PPO  cpu   n_envs=8               8,636 steps/s    89%
PPO  cpu   n_envs=64              8,252 steps/s    85%   oversubscribed
PPO  cuda  n_envs=32              6,814 steps/s    70%
PPO  cuda  n_envs=8               6,173 steps/s    64%
```

The workload is bound by **Python environment stepping**, not by matmuls — an MLP policy
over a 1-D action space is a tiny network, and this env does a 40-iteration bisection in
`j_from_power` every step. Moving those small batches to the device costs more than the
device saves.

**PPO runs on CPU, over plain SSH, with no `gpurun` involved.** `n_envs=32`; 64 oversubscribes.

**2M steps = 3.4 min per seed.**

## SAC — the GPU helps a lot, but only after fixing the defaults

```
SAC cuda n_envs=1   train_freq=1   grad_steps=1        246 steps/s   <-- SB3 default
SAC cpu  n_envs=1   train_freq=1   grad_steps=1         81 steps/s
SAC cuda n_envs=8   train_freq=8   grad_steps=8      1,873 steps/s
SAC cpu  n_envs=8   train_freq=8   grad_steps=8        626 steps/s
SAC cuda n_envs=32  train_freq=32  grad_steps=32     6,520 steps/s   <-- use this
SAC cpu  n_envs=32  train_freq=32  grad_steps=32     2,450 steps/s
SAC cuda n_envs=32  train_freq=32  grad_steps=8     13,889 steps/s   <-- NOT equivalent, see below
```

SB3's SAC default does **one gradient update per environment step on a single env**, which
pays Python and kernel-launch overhead once per step. Batching the updates — 32 parallel
envs, `train_freq=32`, `gradient_steps=32` — keeps the **same 1:1 update-to-step ratio**
while paying that overhead once per 32 steps. That is a **26× speedup at identical learning
dynamics**, and here the GPU is genuinely worth it: 2.7× over CPU.

**SAC runs on GPU, under `gpurun -g 1`.** 2M steps = 5.1 min per seed.

### The 13,889 steps/s row is a trap

`gradient_steps=8` does 8 updates per 32 env steps instead of 32 — a **4× lower update
ratio**. That is a different algorithm configuration, not a free speedup, and it will cost
sample efficiency. Do not use it because it is the biggest number in the table. If you want
it, validate that it still learns by comparing learning curves against `gradient_steps=32`
at equal *env steps*, not equal wall-clock.

The same caution applies more mildly to the recommended row: `n_envs=32` changes SAC's
replay and exploration dynamics versus `n_envs=1`. Sanity-check one learning curve before
launching the full matrix.

---

## What this means for the experiment matrix

Full matrix = 2 algorithms × 5 seeds × 4 reward weights = 40 runs at 2M steps.

| | Device | Per seed | 20 runs | Concurrency | Wall-clock |
|---|---|---|---|---|---|
| PPO | CPU, `n_envs=32` | 3.4 min | 68 min | 4 concurrent (128 cores) | **~17 min** |
| SAC | GPU, `gpurun -g 1` | 5.1 min | 102 min | 4 concurrent (4 GPUs) | **~26 min** |

PPO and SAC use **different resources**, so run them at the same time. The entire matrix is
well under an hour, and costs about **1.7 GPU-h** of the 90 GPU-h weekly quota.

`gpurun` bills wall-clock time a job *holds* a GPU, not utilisation — so never run PPO under
the broker. It would be slower *and* burn quota.

## The consequence that matters for the paper

Person B's Day-3 reward-weight sweep was planned as 4 coarse points because it was budgeted
as the sprint's most expensive step. At these rates it is minutes.

**Spend that on a denser w₂ sweep — 12–15 points instead of 4.** The headline figure is the
yield-vs-degradation Pareto frontier (`DECISIONS.md` §8), and a frontier traced by 15 points
is a far stronger figure than one traced by 4. This is the single best use of the server for
the paper's quality, and it is worth more than the raw speedup.
