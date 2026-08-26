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

### Sustained vs warm-up: an open question, and a retracted claim

An earlier revision of this file claimed the SAC benchmark was 4.1x optimistic, on the
basis of a 2M-step run that sustained 1,578 steps/s. **That claim was wrong and is
retracted.** The run in question had `--device cpu` forced by `scripts/pipeline_check.py`,
so it exercised the CPU path (benchmarked at 2,450 steps/s, lower here under contention on
a shared box) and said nothing about the GPU rate.

The underlying concern is still legitimate and remains **unverified**: `bench_sac.py` runs
only 6,000 steps, so the replay buffer never exceeds ~6.5k entries, while a real run fills
`buffer_size=1_000_000`. Sampling a batch from a filled buffer is slower than from a
nearly-empty one, so the benchmark may still overstate sustained throughput. How much is
not yet measured.

**Do not plan the SAC arm against either number until a full-length run on an actual GPU
has been timed.** PPO is unaffected either way -- it is on-policy with no replay buffer,
and its measured sweep rate (~4,950 steps/s per run at `n_envs=32`, four concurrent) is
consistent with its benchmark.

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

## Final hardware allocation

| | Device | Broker | Settings | Throughput |
|---|---|---|---|---|
| **PPO** | 128-core CPU | **none** | `n_envs=32` | 9,712 steps/s |
| **SAC** | 1× H200 | `gpurun -g 1` | `n_envs=32, train_freq=32, gradient_steps=32` | 6,520 steps/s |

PPO gets no GPU allocation at all. `gpurun` bills wall-clock time a job *holds* a GPU, not
utilisation, so submitting PPO through the broker would be **both slower and quota-burning**.

## What this means for the experiment matrix

The sweep is now 13 w₂ values × 5 seeds × 2 algorithms = **130 runs** at 2M steps
(`configs/default.yaml → sweep.w2`). Derived from the measured rates above:

| | Per seed | 65 runs, sequential | Concurrency | Wall-clock |
|---|---|---|---|---|
| PPO | 3.4 min | 221 min | 4 concurrent (128 cores ÷ 32 envs) | **~55 min** |
| SAC | 5.1 min | 332 min | 4 concurrent (4 GPUs) | **~83 min** |

PPO and SAC use **different resources**, so run them at the same time: total wall-clock is
set by the slower arm, **≈83 min**, and the GPU cost is **≈5.5 GPU-h** of the 90 GPU-h
weekly quota.

Even at 3× the original experiment count, this is a fraction of one week's quota.

## The consequence that matters for the paper

Person B's Day-3 reward-weight sweep was planned as 4 coarse points because it was budgeted
as the sprint's most expensive step. At these measured rates it is minutes, so it is now
**13 points** — see `configs/default.yaml → sweep.w2`.

The headline figure is the yield-vs-degradation Pareto frontier (`DECISIONS.md` §8), and
**each w₂ value is exactly one point on it**. Four points do not trace a frontier; they
sample it too sparsely to show its shape, and a reviewer cannot tell a genuine trade-off
curve from four unconnected results. Thirteen points at quarter-decade spacing resolve the
knee of the curve — the region where a small yield sacrifice buys a large degradation
reduction, which is the paper's actual claim.

This is the single best use of the server for the paper's quality, and it is worth more than
the raw speedup.

**The range was later narrowed to 0.1–20** for a reason unrelated to throughput: above
w₂ ≈ 20 the reward-optimal policy is to shut the plant down, so those points trained a
switched-off controller rather than tracing a frontier. See `DECISIONS.md` §8 and
`scripts/reward_landscape.py`.
