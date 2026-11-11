# First full Kaggle run (2026-11-10/11): incomplete

Kernel `pemwe-followup` version 2, Tesla T4, 4 CPUs, 31 GB RAM, repo commit 2e73552 (ran for 62 minutes, then ended with an error).

* `../rule_family_v2.csv` (the hysteresis-rule grid, 72 points, test days) finished and is valid.
* Training was run two jobs at a time. Only one of the four jobs finished (UTD 1/4, seed 1: `models/kg_utd0.25_w2-20.0_s1.zip`).
  UTD 1/8 seed 0 died with SIGKILL after about 60,000 steps (`curves/`); the others died too, with `UnpicklingError: pickle data
  was truncated` in the worker processes. The cause is most likely memory (two jobs x 32 worker processes), but Kaggle gives no
  kernel log to confirm it. The orchestrator then aborted: no evaluation and no decision were produced.
* Nothing here is a result about update ratios. The second run (one job at a time, with retries and a memory log) replaces it.
