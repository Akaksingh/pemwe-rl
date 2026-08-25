# Person B lane — train.py, evaluate.py, tuned baseline

`train.py` and `evaluate.py` did not exist, and both sweep launchers called a `train.py`
that was not there. That is unblocked.

```bash
python -m pemwe.train --algo sac --seed 0 --override reward.w2=10.0 --run-id sac_w2-10.0_seed0
python -m pemwe.evaluate --all-baselines          # held-out test split
python -m pemwe.evaluate --sweep                  # every checkpoint in models/
python scripts/tune_baseline.py                   # the baseline frontier
python scripts/pipeline_check.py                  # train -> evaluate -> schema, ~5 min
pytest tests/ -q                                  # 52 passed
```

## What is done

| | Status |
|---|---|
| `train.py`, exact launcher interface | ✅ both sweep scripts now find it (the "does not exist yet" NOTE is gone) |
| device split read from config, not flags | ✅ PPO→cpu, SAC→cuda + `train_freq==gradient_steps==n_envs` |
| `r_yield`/`r_deg`/`r_ramp` logged separately to TB | ✅ verified in the event files for both algorithms |
| `evaluate.py`, frozen CONTRACTS §3 schema | ✅ validated key-for-key vs `scripts/fake_results.py` |
| one code path, scripted + learned | ✅ `SB3Policy` wears the baselines' `reset()`/`act()` |
| held-out evaluation | ✅ defaults to `test`; a test asserts no train-day leakage |
| `RampLimitedBaseline` tuned | ✅ 0.02 → **0.03**/min |
| tests for this lane | ✅ 13 new (52 total) |
| the 130-run matrix | ❌ **server job** — see below |

## Baseline tuning: why not the obvious criterion

`ramp_limit_frac_per_min` is now chosen at the **knee** of the controller's own yield–life
frontier over 24 train days: **0.03/min**, keeping **98.7 %** of the naive rule's yield for
**+1.46 years** of projected life.

Choosing by argmax of reward at the default weights would have picked 0.06 → 5.77 yr. At
w2 = 1 the degradation term is ~1 % of the yield term, so reward spans only **3.6 % across
the entire grid** and its argmax is not a meaningful choice — it gives up almost all the
life benefit, i.e. a strawman on exactly the axis the paper's claim lives on. Maximising
life is the mirror-image error: it picks a controller too sluggish to follow the resource.
The full frontier is printed by the script so the choice is auditable, and a test fails if
this baseline ever stops being strong.

## ⚠️ The one thing you must check before trusting a sweep

**w₂ does not yet move the policy.** Pipeline checks on the real data, evaluated on
held-out days:

| steps | H₂ spread across w2 ∈ [0.1, 100] | degradation spread |
|---|---|---|
| 25 k | 0.2 % | 0.9 % |
| 180 k | 0.1 % | 3.6 % |

Direction is right (higher w2 → less degradation) and neither degenerate policy appears,
but the magnitude is not yet a frontier. **This is expected at 9 % of the 2 M-step budget**
— and the reward scaling says it should resolve: per episode `r_yield` ≈ 14 900 and
`r_deg` ≈ 730, so w2 = 0.1 makes degradation 0.5 % of the objective and w2 = 100 makes it
**5× the yield term**. The knob has the authority; the policy has not had the steps to use
it.

`pipeline_check.py` deliberately reports **INCONCLUSIVE**, not PASS, below a 5 % spread: a
policy that ignores w2 entirely satisfies a direction-only test half the time by chance.

**Run the check at full steps on the server before committing the 130-run matrix.** If the
spread is still <5 % at 2 M steps, there is no Pareto frontier and the headline figure
fails — and that is worth 20 minutes to find out rather than discovering it on Day 4.

## Then

```bash
./scripts/validate_experiment_config.py    # 22 invariants
./scripts/run_ppo_sweep.sh                 # dry run first
./scripts/run_sac_sweep.sh
./scripts/run_ppo_sweep.sh --go &          # CPU,  ~55 min
./scripts/run_sac_sweep.sh --go &          # GPU,  ~82 min, ~5 GPU-h
python -m pemwe.evaluate --sweep           # -> results/*.json for C's figures
```

## Notes

- Local numbers here are from a **CPU laptop at ~250 steps/s** and 4–8 envs. They exist to
  prove the apparatus, not to be results. `results/pipeline_check/` is gitignored and every
  run carries a `pipe_` prefix so it can never be mistaken for a paper number.
- SAC `learning_starts=10_000`: below ~15 k steps nothing trains at all. Do not read a
  short run as evidence of anything.
- `torch` + `stable-baselines3` were missing from `.venv`; installed (CPU torch locally —
  the server has its own CUDA build).
- The SB3 tests skip cleanly when torch is absent, so A's and C's suites still run without it.
