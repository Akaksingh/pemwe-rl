import sys, time
sys.path.insert(0, "src")
from pathlib import Path
from pemwe.config import load_config
from pemwe import train as T

def run(n_envs):
    cfg = load_config()
    s = cfg["train"]["sac"]
    s["train_freq"] = n_envs
    s["gradient_steps"] = n_envs * n_envs          # UTD = gradient_steps / (train_freq * n_envs) = 1
    prof = T.train_profiles(cfg, 20)
    venv = T.make_vec_env(cfg, prof, n_envs, seed=0, subproc=False)
    model = T.build_model("sac", cfg, venv, 0, Path(".t1_tb"), "cuda")
    chunk = 10_240 if n_envs == 32 else 10_000
    model.learn(total_timesteps=chunk, progress_bar=False)            # warm-up (+ first updates)
    u0, s0, t0 = model._n_updates, model.num_timesteps, time.time()
    model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
    dt = time.time() - t0
    du, ds = model._n_updates - u0, model.num_timesteps - s0
    print(f"n_envs={n_envs:2d} train_freq={n_envs} gradient_steps={s['gradient_steps']:4d}: "
          f"UTD={du/ds:.3f}  {du/dt:6.0f} updates/s  {ds/dt:6.0f} env-steps/s  "
          f"-> 2M steps = {2_000_000/(ds/dt)/60:5.0f} min", flush=True)
    venv.close()

for n in (8, 32):
    run(n)
