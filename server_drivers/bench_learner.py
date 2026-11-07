import sys, time, csv, os
sys.path.insert(0, "src")
from pathlib import Path
import torch
from pemwe.config import load_config
from pemwe import train as T

OUT = "results_hpc/hpc_scaling.csv"
rows = []
def one(device, threads, batch, n_updates):
    cfg = load_config(); s = cfg["train"]["sac"]
    s["train_freq"] = 8; s["gradient_steps"] = 64; s["batch_size"] = batch
    prof = T.train_profiles(cfg, 20)
    venv = T.make_vec_env(cfg, prof, 8, seed=0, subproc=False)
    if threads: torch.set_num_threads(threads)
    model = T.build_model("sac", cfg, venv, 0, Path(".t1_tb"), device)
    model.learn(total_timesteps=10_240, progress_bar=False)     # fill replay buffer past learning_starts
    if device == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.train(gradient_steps=n_updates, batch_size=batch)
    if device == "cuda": torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    venv.close()
    return dt

for rep in range(3):
    for batch in (256, 512, 1024, 2048):
        dt = one("cuda", None, batch, 2000)
        rows.append(["learner_updates_per_s", "cuda", 0, rep, 2000, dt, 2000 / dt, f"batch={batch}"])
        print(f"cuda batch={batch:4d} rep={rep} {2000/dt:7.0f} updates/s", flush=True)
for threads in (4, 8, 16):
    for rep in range(2):
        dt = one("cpu", threads, 512, 300)
        rows.append(["learner_updates_per_s", "cpu", threads, rep, 300, dt, 300 / dt, "batch=512"])
        print(f"cpu threads={threads:2d} batch=512 rep={rep} {300/dt:7.1f} updates/s", flush=True)

with open(OUT, "a", newline="") as f:
    csv.writer(f).writerows(rows)
print("appended", OUT)
