# Compute: the `hcd-dept-h200` server

Shared training box. Person B owns the run schedule on it; A and C use it for the
long-horizon rollout and for figure regeneration respectively.

```
host    192.168.2.69   (hcd-dept-h200)
user    anshgupta456ansh          key-based SSH, no password
cpu     AMD EPYC 9535, 128 cores
ram     1 TB
gpu     4x NVIDIA H200, 143 GB each
disk    17 TB free
os      Ubuntu 24.04.4, Python 3.12.3
```

## Connecting

```bash
ssh anshgupta456ansh@192.168.2.69
```

The `-L 11434:localhost:11434` tunnel is Ollama's port. **Nothing on that port is running
on this server, and nothing in this project uses it** — no LLM is in the training loop. Drop
the flag unless you are using the box for something else.

## GPUs are brokered — you cannot touch them directly

The account is not in the `gpuaccess` group and `nvidia-smi` fails on purpose. Every GPU job
goes through the broker:

```bash
gpurun --status                  # your budget + which GPUs are free
gpurun -g 1 <command>            # run <command> holding 1 GPU
gpurun -g 2 <command>            # 2 GPUs
gpurun --kill <job_id>
```

**Quota: 90 GPU-hours per week.** It is consumed by wall-clock time the job holds the GPU,
not by utilisation — so a job that holds a GPU and then does CPU-bound work burns quota for
nothing. That matters here; see below.

## Setup (already done once)

```bash
cd ~/pemwe-rl
export PATH=$HOME/.local/bin:$PATH        # uv lives here
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch gymnasium stable-baselines3 \
    numpy pandas scipy matplotlib pyyaml pyarrow tensorboard
```

Installed: `torch 2.13.0+cu130`, `gymnasium 1.3.0`, `stable-baselines3 2.9.0` — the same
gymnasium and SB3 versions as the local Windows venv, so results are comparable across
machines.

## Syncing code

The repo is private, and the server has no GitHub credentials. Push from your laptop:

```bash
git archive HEAD | ssh anshgupta456ansh@192.168.2.69 'tar -x -C ~/pemwe-rl'
```

Only committed files go across, which is the point — if it is not committed, it does not run
on the server. Pull results back with `scp -r ...:~/pemwe-rl/results/ .`

## Which device to actually use — measured, see `BENCHMARK.md`

| | Device | Why |
|---|---|---|
| **PPO** | **CPU**, plain SSH, `n_envs=32` | 9,712 steps/s. On the H200 it is **~30 % slower** — the job is bound by Python env stepping, not matmuls |
| **SAC** | **GPU**, `gpurun -g 1`, `train_freq=32 gradient_steps=32` | 6,520 steps/s, 2.7× over CPU. SB3's defaults are 26× slower than this |

They use different resources, so **run them at the same time**. The full 40-run matrix is
under an hour and costs ~1.7 of the 90 weekly GPU-hours.

Re-run the benchmarks yourself if anything about the env changes:

```bash
cd ~/pemwe-rl
.venv/bin/python scripts/bench_device.py            # PPO/CPU vs GPU sweep
gpurun -g 1 .venv/bin/python scripts/bench_sac.py   # SAC configurations
```

## Running a job that survives your SSH session dropping

```bash
cd ~/pemwe-rl
nohup .venv/bin/python -m pemwe.train --algo sac --seed 0 > logs/sac_s0.log 2>&1 &
```

For a whole seed sweep, launch them as separate processes — with 128 cores there is no
reason to run seeds sequentially. PPO on cores, SAC on GPUs, concurrently:

```bash
cd ~/pemwe-rl && mkdir -p logs

# PPO: CPU only, no broker. 4 concurrent x 32 envs = 128 cores.
for s in 0 1 2 3 4; do
  OMP_NUM_THREADS=8 nohup .venv/bin/python -m pemwe.train --algo ppo --seed $s \
    > logs/ppo_s$s.log 2>&1 &
done

# SAC: one GPU each, 4 at a time (4 GPUs available).
for s in 0 1 2 3; do
  OMP_NUM_THREADS=8 nohup gpurun -g 1 .venv/bin/python -m pemwe.train --algo sac --seed $s \
    > logs/sac_s$s.log 2>&1 &
done
```

**Set `OMP_NUM_THREADS`.** Torch defaults to grabbing every core per process; five
unconstrained processes on 128 cores will fight each other and run slower than one.

**Watch the queue** with `gpurun --status` — more than 4 concurrent SAC jobs will queue
rather than fail, and a queued job still counts against your wall-clock plans.
