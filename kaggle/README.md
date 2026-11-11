# Running the follow-up experiments on Kaggle

What this does and why (in order of how much it helps the paper):

1. **`rules`**: a smarter simple rule (hysteresis + cap + slew limit). If the learned policy's low-degradation
   advantage survives it, the claim is much safer against the obvious reviewer objection.
2. **`ratio`**: the open question from the update-ratio study. At w2 = 12.9 and 20 the cheap ratio (1/32) did not learn
   the weight's effect and the full ratio (1) did. This tests 1/8 and 1/4 at w2 = 20 on training days only and decides
   automatically whether either matches the full-ratio policies (degradation within 10%, hydrogen within 5%).
3. **`seeds`**: if a ratio qualifies, 6 more seeds (2 to 7) at w2 = 12.9 and 20, evaluated once on the 72 test days. If none
   qualifies it stops, because the full ratio is too slow for 12 runs on a free GPU.

Expected time (estimates, not measured on Kaggle): rules about 5 minutes; ratio about 1 to 1.5 hours; seeds about 1 to 3
hours. Everything fits one session if you use **Save and Run All** (see below).

## One-time setup

1. Create a Kaggle notebook. In **Settings**, set **Accelerator = GPU** (T4 is best; P100 also works) and **Internet = On**.
   (Internet needs a phone-verified account.)
2. If the GitHub repo is private, add a Kaggle secret named `GH_TOKEN` (a read-only token) and uncomment the token lines in
   the cell below. If it is public, nothing is needed.
3. Paste this into the first cell:

```python
import os, subprocess
# from kaggle_secrets import UserSecretsClient; os.environ["GH_TOKEN"] = UserSecretsClient().get_secret("GH_TOKEN")
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/Akaksingh/pemwe-rl.git", "/kaggle/working/repo"], check=True)
```

## Step A: smoke test (do this first, about 10 to 15 minutes)

```python
!python /kaggle/working/repo/kaggle/run_experiments.py --phase all --smoke --utd 0.125
```

It runs the whole pipeline on tiny settings (12,000-step runs, 3 evaluation days). It should end with a line
`wrote /kaggle/working/kaggle_results.zip`. If anything fails, copy the last 30 lines of the output and send them to me; do
not start the full run until the smoke test passes.

## Step B: the full run

Delete the smoke outputs first, then run:

```python
!rm -rf /kaggle/working/out /kaggle/working/kaggle_results.zip /kaggle/working/repo/models/kg_*.zip
!python /kaggle/working/repo/kaggle/run_experiments.py --phase all --max-hours 8
```

To keep it running after you close the browser, click **Save Version, then Save and Run All (Commit)**. Kaggle then runs it
in the background and keeps the output. If a session limit stops it, rerun individual phases (`--phase seeds --utd 0.125`);
finished work in `out/` is reused by the decision step.

## When it finishes

Download `kaggle_results.zip` from the notebook's **Output** tab and put it in the repo folder. Then tell me. I will unzip it
into `results_kaggle/`, regenerate the frontier and the paper numbers, and update the text. The zip contains the rule-grid
CSV, the decision JSON with the numbers behind it, the evaluation CSVs, the training curves, the new policies, and
`hardware_specifications_kaggle.txt` (so the results are labelled with the platform they came from).

## Notes

- **Hardware numbers.** Do not use anything timed on Kaggle in the paper's scaling results; those are measured on the
  workstation. Kaggle runs are for statistical seeds and the rule comparison only.
- **Quotas.** Free GPU time is limited per week and sessions are limited in length; check your current quota on Kaggle's site.
- **Time budget.** `--max-hours` (default 8) stops new jobs from starting late, and the zip is written even if the run is cut short.
- **Your own GPU.** The same script also runs on your laptop's GTX 1650 if PyTorch works there (it needs Linux or WSL2 for the
  `fork`-based evaluation step; on plain Windows use Kaggle).

## Status (2026-11-11)

Both runs were done through the Kaggle API (`kaggle/kernel/`, `kaggle kernels push -p kaggle/kernel`; the token is an environment
variable and is never stored in the repo). Run 1 (two jobs in parallel) lost three of four jobs, probably to memory; one SAC job
uses about 22.5 GB of the 31 GB. The default is now `--parallel 1`, a killed job is retried once, and memory is logged every
2 minutes. Run 2 finished in 271 minutes; its data is in `results_kaggle/run2/` and its analysis is `scripts/analyze_kaggle.py`.
