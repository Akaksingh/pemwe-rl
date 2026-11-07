import glob, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
for s in (0, 1):
    fs = sorted(glob.glob(f"results/tb/pilot_utd1_w2-0.1_seed{s}/*/events*"))
    if not fs: print(f"seed{s}: no tb events yet"); continue
    ea = EventAccumulator(fs[-1].rsplit("/", 1)[0], size_guidance={"scalars": 0}); ea.Reload()
    tags = ea.Tags()["scalars"]
    t = "rollout/ep_rew_mean" if "rollout/ep_rew_mean" in tags else (tags[0] if tags else None)
    if t is None: print(f"seed{s}: no scalars"); continue
    ev = ea.Scalars(t)
    print(f"seed{s}: last logged step={ev[-1].step:,}  {t}={ev[-1].value:.1f}  (n_logged={len(ev)})")
