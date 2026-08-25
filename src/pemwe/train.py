"""SB3 training. OWNER: Person B.

    python -m pemwe.train --algo {ppo,sac} --seed N --override reward.w2=X --run-id ID

That signature is not negotiable -- `scripts/run_ppo_sweep.sh` and `scripts/run_sac_sweep.sh`
already call it exactly this way.

Everything that decides throughput comes from `configs/default.yaml`, never from a flag
default, because the device split is MEASURED and not advisory (BENCHMARK.md):

    PPO -> cpu,  n_envs=32                              9,712 steps/s
    SAC -> cuda, n_envs=32, train_freq=gradient_steps=32  6,520 steps/s

PPO is ~30 % SLOWER on an H200 than on the 128-core CPU: this workload is bound by Python
environment stepping, not matmuls. SAC is 2.7x faster on the GPU, but only once its updates
are batched across the 32 envs at an unchanged 1:1 update-to-step ratio -- SB3's default
(one update per step on one env) is 26x slower than necessary.

THE ONE NON-OBVIOUS REQUIREMENT is `RewardComponentCallback` below. Total reward cannot
distinguish "learning" from "found a degenerate policy": an agent that parks at idle and
one that has genuinely learned the trade-off can post similar totals for opposite reasons.
The three components are logged UNWEIGHTED and separately so the yield term and the
degradation term can be watched moving independently. Without that, the reward-weight sweep
is unreadable and reward hacking costs a day.

Training uses the TRAIN split only (DECISIONS.md 1-2). Held-out days belong to evaluate.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .config import load_config, config_id
from .env import PEMWEEnv
from . import profiles

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
TB = ROOT / "results" / "tb"


# --------------------------------------------------------------------------------------
# the callback the brief calls the one non-obvious requirement
# --------------------------------------------------------------------------------------

def _make_callback_cls():
    """Imported lazily so `--help` and the dry-run path do not need torch."""
    from stable_baselines3.common.callbacks import BaseCallback

    class RewardComponentCallback(BaseCallback):
        """Log r_yield / r_deg / r_ramp separately, UNWEIGHTED, plus operating stats.

        CONTRACTS.md 1 requires the components in `info` unweighted; this is what makes
        them visible during training rather than only at evaluation. `self.locals["infos"]`
        is a list with one dict per parallel env, so everything is averaged over the
        vector before it is recorded.
        """

        COMPONENTS = ("r_yield", "r_deg", "r_ramp")
        STATS = ("h2_kg", "dv_deg_uv", "j", "eta_lhv", "curtailed_w")

        def __init__(self, weights, log_every=2048, verbose=0):
            super().__init__(verbose)
            self.w1, self.w2, self.w3 = weights
            self.log_every = log_every
            self._buf = {k: [] for k in (*self.COMPONENTS, *self.STATS)}
            self._on = []
            self._cycles = 0
            self._steps = 0

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                if "r_yield" not in info:          # terminal-observation dicts
                    continue
                for k in self._buf:
                    if k in info:
                        self._buf[k].append(float(info[k]))
                self._on.append(1.0 if info.get("is_on") else 0.0)
                self._cycles += int(bool(info.get("cycled")))
                self._steps += 1

            if self._steps >= self.log_every:
                self._flush()
            return True

        def _flush(self):
            if not self._buf["r_yield"]:
                return
            mean = {k: float(np.mean(v)) for k, v in self._buf.items() if v}

            # record_mean, not record: SB3 dumps on its own schedule, and PPO/SAC dump at
            # very different rates. With plain record, several flushes between two dumps
            # would silently become last-one-wins; record_mean averages them until the
            # dump, so the logged value means the same thing for both algorithms.
            rec = self.logger.record_mean

            # UNWEIGHTED -- the whole point. These are comparable across the w2 sweep.
            for k in self.COMPONENTS:
                rec(f"components/{k}", mean[k])

            # ...and weighted, which is what actually drives the gradient, so the two can
            # be compared directly when diagnosing a degenerate policy.
            rec("weighted/yield", self.w1 * mean["r_yield"])
            rec("weighted/deg", -self.w2 * mean["r_deg"])
            rec("weighted/ramp", -self.w3 * mean["r_ramp"])

            for k in self.STATS:
                if k in mean:
                    rec(f"plant/{k}", mean[k])
            rec("plant/duty_cycle", float(np.mean(self._on)))
            rec("plant/cycles_per_1k_steps",
                1000.0 * self._cycles / max(self._steps, 1))

            # The two degenerate policies this sweep is designed to catch.
            rec("diag/parked_at_idle", float(np.mean(self._on) < 0.05))
            rec("diag/pinned_at_rated", float(mean.get("j", 0.0) > 1.9))

            self._buf = {k: [] for k in self._buf}
            self._on = []
            self._cycles = 0
            self._steps = 0

        def _on_training_end(self) -> None:
            self._flush()

    return RewardComponentCallback


# --------------------------------------------------------------------------------------
# env construction
# --------------------------------------------------------------------------------------

def train_profiles(cfg, n_days=None):
    """(n_days, 1440) array of TRAIN days. Never the test split."""
    df = profiles.load_profiles(cfg["data"]["profile_path"])
    dates = profiles.SPLITS["train"]
    if n_days:
        dates = dates[:n_days]
    return profiles.profiles_array(df, dates)


def make_vec_env(cfg, prof, n_envs, seed, subproc=True):
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor

    def factory(rank):
        def _f():
            env = PEMWEEnv(cfg, profiles=prof, seed=seed * 1000 + rank)
            env.reset(seed=seed * 1000 + rank)
            return env
        return _f

    cls = SubprocVecEnv if (subproc and n_envs > 1) else DummyVecEnv
    venv = cls([factory(i) for i in range(n_envs)])
    return VecMonitor(venv)


def build_model(algo, cfg, venv, seed, tb_dir, device_override=None):
    from stable_baselines3 import PPO, SAC

    t = cfg["train"]
    n_envs = t["n_envs"]

    if algo == "ppo":
        device = device_override or t["ppo"]["device"]
        # n_steps chosen so the rollout buffer is 32 x 256 = 8192 transitions, a few
        # minutes of simulated plant time per update rather than a few seconds.
        return PPO("MlpPolicy", venv, seed=seed, device=device, verbose=0,
                   n_steps=256, batch_size=1024, n_epochs=10,
                   learning_rate=3e-4, gamma=0.999, gae_lambda=0.95,
                   clip_range=0.2, ent_coef=0.0,
                   policy_kwargs=dict(net_arch=[256, 256]),
                   tensorboard_log=str(tb_dir))

    s = t["sac"]
    device = device_override or s["device"]
    # gradient_steps == train_freq == n_envs keeps the update-to-step ratio at 1:1 while
    # batching the updates. run_sac_sweep.sh aborts if that identity is broken, because
    # gradient_steps=8 would be a different algorithm wearing the same name.
    return SAC("MlpPolicy", venv, seed=seed, device=device, verbose=0,
               train_freq=s["train_freq"], gradient_steps=s["gradient_steps"],
               batch_size=s["batch_size"], learning_rate=3e-4,
               buffer_size=1_000_000, learning_starts=10_000,
               gamma=0.999, tau=0.005, ent_coef="auto",
               policy_kwargs=dict(net_arch=[256, 256]),
               tensorboard_log=str(tb_dir))


# --------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pemwe.train")
    ap.add_argument("--algo", choices=["ppo", "sac"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--override", action="append", default=[],
                    help="config override, e.g. reward.w2=10.0. Repeatable.")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--total-timesteps", type=int, default=None,
                    help="override train.total_timesteps (short sanity runs)")
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--device", default=None, help="override the configured device")
    ap.add_argument("--train-days", type=int, default=None)
    ap.add_argument("--no-subproc", action="store_true",
                    help="DummyVecEnv instead of SubprocVecEnv (debugging)")
    args = ap.parse_args(argv)

    cfg = load_config(overrides=args.override)
    run_id = args.run_id or f"{args.algo}_{config_id(cfg)}_seed{args.seed}"
    steps = args.total_timesteps or cfg["train"]["total_timesteps"]
    n_envs = args.n_envs or cfg["train"]["n_envs"]

    device = args.device or cfg["train"][args.algo]["device"]
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            print(f"[{run_id}] configured device is cuda but no GPU is visible; "
                  f"falling back to cpu. Throughput will be ~2.7x lower (BENCHMARK.md).",
                  file=sys.stderr)
            device = "cpu"
    if args.algo == "ppo" and device != "cpu":
        print(f"[{run_id}] WARNING: PPO on {device}. BENCHMARK.md measures this ~30% "
              f"SLOWER than CPU, and under gpurun it also burns quota.", file=sys.stderr)

    prof = train_profiles(cfg, args.train_days)
    print(f"[{run_id}] {args.algo.upper()} seed={args.seed} device={device} "
          f"n_envs={n_envs} steps={steps:,} "
          f"w=({cfg['reward']['w1']}, {cfg['reward']['w2']}, {cfg['reward']['w3']}) "
          f"train_days={len(prof)}", flush=True)

    MODELS.mkdir(exist_ok=True)
    tb_dir = TB / run_id
    tb_dir.mkdir(parents=True, exist_ok=True)

    venv = make_vec_env(cfg, prof, n_envs, args.seed, subproc=not args.no_subproc)
    try:
        model = build_model(args.algo, cfg, venv, args.seed, tb_dir, device)
        cb_cls = _make_callback_cls()
        cb = cb_cls((cfg["reward"]["w1"], cfg["reward"]["w2"], cfg["reward"]["w3"]))

        t0 = time.time()
        model.learn(total_timesteps=steps, callback=cb, tb_log_name="run",
                    progress_bar=False)
        dt = time.time() - t0

        path = MODELS / f"{run_id}.zip"
        model.save(path)
    finally:
        venv.close()

    meta = {
        "run_id": run_id, "algo": args.algo, "seed": args.seed, "device": device,
        "n_envs": n_envs, "total_timesteps": steps,
        "weights": {k: cfg["reward"][k] for k in ("w1", "w2", "w3")},
        "overrides": args.override,
        "degradation": cfg["degradation"],
        "train_days": len(prof),
        "wall_s": round(dt, 1), "steps_per_s": round(steps / dt, 1),
        "model": str(path.relative_to(ROOT)),
        "tensorboard": str(tb_dir.relative_to(ROOT)),
    }
    (MODELS / f"{run_id}.json").write_text(json.dumps(meta, indent=2))

    print(f"[{run_id}] done in {dt/60:.1f} min ({steps/dt:,.0f} steps/s) -> {path}",
          flush=True)
    print(f"[{run_id}] evaluate with:  python -m pemwe.evaluate --model {path} "
          f"--policy {args.algo} --run-id {run_id}", flush=True)
    return meta


if __name__ == "__main__":
    main()
