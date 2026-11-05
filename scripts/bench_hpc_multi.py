"""HPC benchmark 2: aggregate SAC learner throughput with several concurrent runs.

    python scripts/bench_hpc_multi.py [--out results_hpc/hpc_multi.csv]

Single-run SAC at a true update-to-data ratio of 1 is learner-bound (BENCHMARK finding), so the
only way to finish many independent runs (seeds x reward weights) is to run them concurrently.
This measures how much total learner throughput k concurrent runs actually deliver on this
machine, per device mix. Each worker builds its model and fills its replay buffer, then ALL
workers wait for a common wall-clock start time so the timed window overlaps; the reported
aggregate rate is total updates / (last finish - first start).

Updates are timed with model.train(); environment stepping is excluded, so this is the learner
ceiling, not end-to-end training speed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CASES = [("gpu_x1", [("cuda", 1)]),
         ("gpu_x2", [("cuda", 1)] * 2),
         ("gpu_x4", [("cuda", 1)] * 4),
         ("cpu4t_x1", [("cpu", 4)]),
         ("cpu4t_x2", [("cpu", 4)] * 2),
         ("cpu4t_x4", [("cpu", 4)] * 4),
         ("cpu2t_x8", [("cpu", 2)] * 8),
         ("mixed_gpu2_cpu4t_x3", [("cuda", 1)] * 2 + [("cpu", 4)] * 3)]


def worker_main(a):
    import torch
    from pemwe.config import load_config
    from pemwe import train as T
    torch.set_num_threads(a.threads)
    cfg = load_config()
    s = cfg["train"]["sac"]
    s["train_freq"], s["gradient_steps"], s["batch_size"] = 8, 64, 512
    prof = T.train_profiles(cfg, 20)
    venv = T.make_vec_env(cfg, prof, 8, seed=a.idx, subproc=False)
    model = T.build_model("sac", cfg, venv, a.idx, ROOT / f".tb_multi_{a.idx}", a.device)
    model.learn(total_timesteps=10_240, progress_bar=False)       # fill buffer past learning_starts
    wait = a.t0 - time.time()
    if wait > 0:
        time.sleep(wait)
    if a.device == "cuda":
        torch.cuda.synchronize()
    ts = time.time()
    model.train(gradient_steps=a.n_updates, batch_size=512)
    if a.device == "cuda":
        torch.cuda.synchronize()
    te = time.time()
    venv.close()
    print("RESULT " + json.dumps({"idx": a.idx, "device": a.device, "threads": a.threads,
                                  "n_updates": a.n_updates, "t_start": ts, "t_end": te}), flush=True)


def run_case(label, procs_spec, lead_s):
    t0 = time.time() + lead_s
    ps = []
    for i, (dev, thr) in enumerate(procs_spec):
        n_upd = 1500 if dev == "cuda" else 400
        env = dict(os.environ, OMP_NUM_THREADS=str(thr), MKL_NUM_THREADS=str(thr))
        cmd = [sys.executable, __file__, "--worker", "--idx", str(i), "--device", dev,
               "--threads", str(thr), "--n-updates", str(n_upd), "--t0", repr(t0)]
        ps.append(subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   text=True, env=env, cwd=ROOT))
    res = []
    for p in ps:
        out, _ = p.communicate()
        for line in out.splitlines():
            if line.startswith("RESULT "):
                res.append(json.loads(line[7:]))
    if len(res) != len(procs_spec):
        print(f"  !! {label}: only {len(res)}/{len(procs_spec)} workers reported", flush=True)
    if not res:
        return None
    total = sum(r["n_updates"] for r in res)
    t_a, t_b = min(r["t_start"] for r in res), max(r["t_end"] for r in res)
    per = [r["n_updates"] / (r["t_end"] - r["t_start"]) for r in res]
    return {"case": label, "n_procs": len(res),
            "n_gpu": sum(r["device"] == "cuda" for r in res),
            "n_cpu": sum(r["device"] == "cpu" for r in res),
            "cpu_threads": max([r["threads"] for r in res if r["device"] == "cpu"], default=0),
            "total_updates": total, "wall_s": t_b - t_a, "aggregate_updates_per_s": total / (t_b - t_a),
            "mean_per_proc_updates_per_s": sum(per) / len(per),
            "max_start_offset_s": max(r["t_start"] for r in res) - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--n-updates", type=int, default=400)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--lead", type=float, default=75.0, help="seconds workers get to build before the shared start")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "results_hpc" / "hpc_multi.csv"))
    a = ap.parse_args()
    if a.worker:
        return worker_main(a)

    rows = []
    for label, spec in CASES:
        for rep in range(a.reps):
            r = run_case(label, spec, a.lead)
            if r:
                r["repeat"] = rep
                rows.append(r)
                print(f"{label:22s} rep={rep} procs={r['n_procs']} aggregate={r['aggregate_updates_per_s']:7.1f} "
                      f"upd/s  per-proc={r['mean_per_proc_updates_per_s']:6.1f}  "
                      f"start-skew={r['max_start_offset_s']:+.1f}s", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cols = ["case", "repeat", "n_procs", "n_gpu", "n_cpu", "cpu_threads", "total_updates", "wall_s",
            "aggregate_updates_per_s", "mean_per_proc_updates_per_s", "max_start_offset_s"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
