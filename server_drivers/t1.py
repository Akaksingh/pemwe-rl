import sys, time
sys.path.insert(0, "src")
from pathlib import Path
from pemwe.config import load_config
from pemwe import train as T
cfg = load_config()
s = cfg["train"]["sac"]
print("config:", {"n_envs": cfg["train"]["n_envs"], "train_freq": s["train_freq"],
                  "gradient_steps": s["gradient_steps"], "batch": s["batch_size"], "device": s["device"]})
prof = T.train_profiles(cfg, 20)
venv = T.make_vec_env(cfg, prof, cfg["train"]["n_envs"], seed=0, subproc=False)
model = T.build_model("sac", cfg, venv, 0, Path(".t1_tb"), "cuda")
steps = 40_000
t0 = time.time()
model.learn(total_timesteps=steps, progress_bar=False)
dt = time.time() - t0
start = model.learning_starts
print(f"num_timesteps={model.num_timesteps}  learning_starts={start}  n_updates={model._n_updates}")
eligible = model.num_timesteps - start
print(f"transitions after learning_starts: {eligible}")
print(f"measured update-to-data ratio = {model._n_updates / eligible:.4f}   (docs claim 1.0)")
print(f"wall {dt:.0f}s -> {steps/dt:.0f} steps/s")
venv.close()
