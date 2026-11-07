import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
for rid in ("pilot2_g0.9_s0", "pilot2_g0.8_s0", "pilot2_g0.95_s1", "pilot2_g0.95_rs10_s0"):
    fs = sorted(glob.glob(f"results/tb/{rid}/*/events*"))
    if not fs:
        print(rid, "no events yet"); continue
    ea = EventAccumulator(fs[-1].rsplit("/", 1)[0], size_guidance={"scalars": 0}); ea.Reload()
    tags = ea.Tags()["scalars"]
    if "rollout/ep_rew_mean" not in tags:
        print(rid, "no reward scalar yet"); continue
    ev = ea.Scalars("rollout/ep_rew_mean")
    print(f"{rid:24s} step={ev[-1].step:>8,}  ep_rew_mean={ev[-1].value:8.1f}")
