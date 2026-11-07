# E3b: INCOMPLETE. Stopped on request. Do not use as results.

Follow-up to the update-ratio study at the high degradation weight (w2 = 20, gamma 0.95, 32 environments,
300k steps planned, checkpoints every 50k steps), update ratios 1, 1/2, 1/4, 1/8 with two seeds each.

The experiment was stopped by the authors on 2026-11-07 at about 17:08 (server clock), before any run finished
and before the evaluation step. State at the stop:

| run | steps reached |
|---|---|
| e3b_utd1_s0, e3b_utd1_s1 | 70,016 of 300,000 |
| e3b_utd0.5_s0, e3b_utd0.5_s1 | 140,000 of 300,000 |
| e3b_utd0.25_*, e3b_utd0.125_* | never started |

The progress CSVs here (timesteps, wall seconds, gradient updates, training reward) and the six intermediate
checkpoints in `models/ckpt/e3b_*` are what existed at that moment. No held-out or training-day evaluation was run,
so nothing here supports a claim about at which update ratio the weight's effect is learned. The related completed
result is `results_check/` (update ratios 1/32 vs 1 at w2 = 12.9 and 20), which shows the cheap ratio does not learn it.
