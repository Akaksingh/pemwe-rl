import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
for g in ("0.99", "0.95"):
    fs = sorted(glob.glob(f"results/tb/pilot_utd1_g{g}_seed0/*/events*"))
    if not fs: print(g, "no events yet"); continue
    ea = EventAccumulator(fs[-1].rsplit("/", 1)[0], size_guidance={"scalars": 0}); ea.Reload()
    ev = ea.Scalars("rollout/ep_rew_mean")
    print(f"gamma={g}: last step={ev[-1].step:,}  ep_rew_mean={ev[-1].value:.0f}  (earlier gamma=0.999 run at ~270k: ~20,500-21,400)")
