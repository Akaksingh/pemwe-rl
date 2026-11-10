"""Follow-up experiments for the paper, orchestrated for a Kaggle notebook (any Linux box with a GPU works).

    python run_experiments.py --phase all            # everything, unattended (use "Save & Run All")
    python run_experiments.py --phase all --smoke    # ~10-15 minute end-to-end check; run this FIRST
    python run_experiments.py --phase rules|ratio|seeds [--utd 0.25]

Phases
  rules   a richer simple-rule family (hysteresis + cap + slew limit), 72 grid points on the 72 test days.
          Answers: "can a smarter rule close the learned policy's low-degradation advantage?". CPU only.
  ratio   w2 = 20 at update ratios 1/8 and 1/4 (2 seeds each, gamma 0.95, 300k steps), evaluated on every 5th
          TRAINING day against the full-ratio policies (E5) and the cheap-ratio policies (check) already in the
          repo. Decides which intermediate ratio (if any) learns the weight's effect. No test days are used.
  seeds   6 more seeds (2..7) for w2 = 12.9 and 20 at the ratio chosen by `ratio`, evaluated once on the 72
          test days, together with the existing seeds 0 and 1. Skipped (with a message) if no ratio qualifies.

Everything this script launches is an existing, already-validated entry point of the repo
(python -m pemwe.train, scripts/run_scenario_grid.py, scripts/rule_family_v2.py); this file only sequences
them, records the hardware and versions, decides, and zips the outputs.

A time budget (--max-hours, default 8) stops new jobs from starting late. Outputs: OUT/ and one zip.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_URL = os.environ.get("REPO_URL", "https://github.com/Akaksingh/pemwe-rl.git")
WORK = Path(os.environ.get("WORK_DIR", "/kaggle/working"))
REPO = WORK / "repo"
OUT = WORK / "out"
START = time.time()

# decision rule for "this update ratio learns the weight's effect at w2 = 20": within tolerance of the
# full-ratio policies on the same training days
TOL_DEG = 1.10          # degradation at most 10% above the full-ratio policies
TOL_H2 = 0.95           # hydrogen at least 95% of the full-ratio policies


def log(*a):
    print(f"[{(time.time() - START) / 60:6.1f} min]", *a, flush=True)


def run(cmd, dry, cwd=None, env=None, check=True):
    log("$", " ".join(str(c) for c in cmd))
    if dry:
        return 0
    return subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, check=check).returncode


def sh_out(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:                                   # noqa: BLE001
        return f"(unavailable: {e})"


def time_left(max_hours):
    return max_hours * 3600 - (time.time() - START)


# --------------------------------------------------------------------------------------------------
def setup(a):
    OUT.mkdir(parents=True, exist_ok=True)
    if not (REPO / ".git").exists():
        url = REPO_URL
        tok = os.environ.get("GH_TOKEN")
        if tok and url.startswith("https://"):
            url = url.replace("https://", f"https://{tok}@")
        run(["git", "clone", "--depth", "1", url, REPO], a.dry_run)
    run([sys.executable, "-m", "pip", "install", "-q", "stable-baselines3==2.9.0", "gymnasium==1.3.0",
         "pyarrow", "pyyaml", "tensorboard"], a.dry_run)
    if a.dry_run:
        return "cpu"
    import torch
    device = "cpu"
    if torch.cuda.is_available():
        try:
            (torch.zeros(8, device="cuda") + 1).sum().item()     # P100/T4 kernels actually runnable?
            device = "cuda"
        except Exception as e:                                   # noqa: BLE001
            log("CUDA reported available but a test kernel failed; falling back to CPU:", e)
    spec = [f"date: {time.strftime('%F %T %Z')}", f"platform: {platform.platform()}",
            f"python: {sys.version.split()[0]}", f"torch: {torch.__version__}  cuda runtime: {torch.version.cuda}",
            f"training device: {device}", f"cpus: {os.cpu_count()}",
            "nvidia-smi: " + sh_out(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
            "lscpu: " + sh_out(["bash", "-c", "lscpu | grep -E 'Model name|^CPU\\(s\\)|Thread|Core' | tr '\\n' ';'"]),
            "memory: " + sh_out(["bash", "-c", "free -g | head -2 | tr '\\n' ' '"]),
            "repo commit: " + sh_out(["git", "-C", str(REPO), "rev-parse", "HEAD"])]
    (OUT / "hardware_specifications_kaggle.txt").write_text("\n".join(spec) + "\n")
    log("hardware recorded; training device =", device)
    return device


def train_job(name, w2, gs, seed, steps, device, dry):
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"), OMP_NUM_THREADS="2")
    d = OUT / "curves"
    d.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pemwe.train", "--algo", "sac", "--seed", seed, "--override", f"reward.w2={w2}",
           "--override", "train.sac.train_freq=32", "--override", f"train.sac.gradient_steps={gs}",
           "--override", "train.sac.gamma=0.95", "--device", device, "--total-timesteps", steps,
           "--progress-csv", d / f"{name}.csv", "--progress-every", 10000, "--run-id", name]
    t0 = time.time()
    run(cmd, dry, cwd=REPO, env=env)
    log(f"finished {name} in {(time.time() - t0) / 60:.1f} min")


def run_jobs(jobs, a, device):
    """jobs: list of (name, w2, gs, seed). Runs `parallel` at a time; skips what no longer fits the budget."""
    steps = 12000 if a.smoke else 300000
    done = []

    def one(j):
        name, w2, gs, seed = j
        if time_left(a.max_hours) < 20 * 60:
            log("time budget nearly used; skipping", name)
            return None
        train_job(name, w2, gs, seed, steps, device, a.dry_run)
        return name

    with ThreadPoolExecutor(max_workers=a.parallel) as ex:
        for r in ex.map(one, jobs):
            if r:
                done.append(r)
    return done


def grid_eval(models, split, stride, out_csv, a, limit_days=0):
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"), OMP_NUM_THREADS="1")
    cmd = [sys.executable, "scripts/run_scenario_grid.py", "--grid", "nominal", "--split", split,
           "--day-stride", stride, "--workers", max(os.cpu_count() or 2, 2), "--out", out_csv, "--models", *models]
    if limit_days:
        cmd += ["--limit-days", limit_days]
    run(cmd, a.dry_run, cwd=REPO, env=env)


# --------------------------------------------------------------------------------------------------
def phase_rules(a):
    d = OUT / "rules"
    d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"), OMP_NUM_THREADS="1")
    cmd = [sys.executable, "scripts/rule_family_v2.py", "--workers", max(os.cpu_count() or 2, 2),
           "--out", d / "rule_family_v2.csv"] + (["--quick"] if a.smoke else [])
    run(cmd, a.dry_run, cwd=REPO, env=env)


def phase_ratio(a, device):
    d = OUT / "ratio"
    d.mkdir(parents=True, exist_ok=True)
    utds = [0.125] if a.smoke else [0.125, 0.25]
    seeds = [0] if a.smoke else [0, 1]
    jobs = [(f"kg_utd{u:g}_w2-20.0_s{s}", 20.0, int(round(1024 * u)), s) for u in utds for s in seeds]
    done = run_jobs(jobs, a, device)
    models = [f"models/{n}.zip" for n in done]
    refs = [f"models/e5_w2-20.0_s{s}.zip" for s in (0, 1)] + [f"models/chk_utd32_w2-20.0_s{s}.zip" for s in (0, 1)]
    if a.dry_run:
        grid_eval(models + refs, "train", 5, d / "grid_ratio_train.csv", a)
        return None
    if not models:
        log("no ratio runs finished; nothing to evaluate")
        return None
    grid_eval(models + refs, "train", 5, d / "grid_ratio_train.csv", a, limit_days=3 if a.smoke else 0)
    return decide(d / "grid_ratio_train.csv", d / "decision.json")


def decide(grid_csv, out_json):
    import pandas as pd
    g = pd.read_csv(grid_csv)
    g = g[(g.scenario == "original") & (g.variant == "nominal")]
    per = g.groupby("controller").agg(h2=("h2_kg", "mean"), dv=("dv_deg_uv", "mean"))

    def grp(prefix):
        s = per[per.index.str.startswith(prefix)]
        return (float(s.h2.mean()), float(s.dv.mean()), int(len(s))) if len(s) else None

    full, cheap = grp("e5_w2-20.0"), grp("chk_utd32_w2-20.0")
    table, chosen = {}, None
    for u in (0.125, 0.25):
        r = grp(f"kg_utd{u:g}_w2-20.0")
        if r is None:
            continue
        ok = bool(full and r[1] <= TOL_DEG * full[1] and r[0] >= TOL_H2 * full[0])
        table[f"{u:g}"] = dict(h2=r[0], dv=r[1], seeds=r[2], matches_full_ratio=ok)
        if ok and chosen is None:
            chosen = u
    res = dict(full_ratio_ref=dict(h2=full[0], dv=full[1], seeds=full[2]) if full else None,
               cheap_ratio_ref=dict(h2=cheap[0], dv=cheap[1], seeds=cheap[2]) if cheap else None,
               intermediate=table, chosen_utd=chosen,
               rule=f"match = degradation <= {TOL_DEG} x full-ratio and hydrogen >= {TOL_H2} x full-ratio (train days)")
    Path(out_json).write_text(json.dumps(res, indent=2))
    log("decision:", json.dumps(res))
    return chosen


def phase_seeds(a, device, utd):
    if utd is None:
        log("no update ratio qualified (or --utd not given): skipping the extra seeds. "
            "The full ratio is too slow for 12 runs on this machine.")
        return
    d = OUT / "seeds"
    d.mkdir(parents=True, exist_ok=True)
    gs = int(round(1024 * utd))
    weights = [20.0] if a.smoke else [12.9, 20.0]
    seeds = [2] if a.smoke else [2, 3, 4, 5, 6, 7]
    jobs = [(f"kg_utd{utd:g}_w2-{w}_s{s}", w, gs, s) for w in weights for s in seeds]
    done = run_jobs(jobs, a, device)
    models = [f"models/{n}.zip" for n in done] + [f"models/e5_w2-{w}_s{s}.zip" for w in weights for s in (0, 1)]
    if not done and not a.dry_run:
        log("no seed runs finished")
        return
    grid_eval(models, "test", 1, d / "grid_seeds_test.csv", a, limit_days=3 if a.smoke else 0)


def finish(a):
    if a.dry_run:
        return
    z = WORK / "kaggle_results.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in OUT.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(OUT).as_posix())
        # the trained policies are small and worth keeping
        for p in (REPO / "models").glob("kg_*.zip"):
            zf.write(p, f"models/{p.name}")
    log(f"wrote {z} ({z.stat().st_size / 1e6:.1f} MB). Download it from the notebook's Output tab.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["all", "rules", "ratio", "seeds"], default="all")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end check (12k-step runs, 3 evaluation days)")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    ap.add_argument("--utd", type=float, default=None, help="update ratio for the seeds phase (default: decided by `ratio`)")
    ap.add_argument("--parallel", type=int, default=2, help="training jobs at once on the one GPU")
    ap.add_argument("--max-hours", type=float, default=8.0)
    a = ap.parse_args()

    device = setup(a)
    chosen = a.utd
    try:
        if a.phase in ("all", "rules"):
            phase_rules(a)
        if a.phase in ("all", "ratio"):
            decided = phase_ratio(a, device)
            chosen = chosen if chosen is not None else decided
        if a.phase in ("all", "seeds"):
            if chosen is None and a.phase == "seeds":
                prev = OUT / "ratio" / "decision.json"
                chosen = json.loads(prev.read_text()).get("chosen_utd") if prev.exists() else None
            phase_seeds(a, device, chosen)
    finally:
        finish(a)


if __name__ == "__main__":
    main()
