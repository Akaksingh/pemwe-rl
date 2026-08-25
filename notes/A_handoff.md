# Standup — Person A: the environment is FROZEN

Gate **G1 passes on all three checks**. Everything below regenerates from the repo.

```bash
python scripts/smoke_test.py             # G1.1 G1.2 G1.3 -> PASS PASS PASS
python scripts/validate_physics.py       # 11/11 plant-model checks PASS + the figure
python -m pytest tests/ -q               # 39 passed
python scripts/calibrate_degradation.py  # term decomposition (--solve to re-search)
python scripts/longhorizon_rollout.py    # 90-day rollout -> results/*.json
python scripts/validate_experiment_config.py   # B's config guard, still passing
```

| Gate G1 | Target | Now (8-day mean) |
|---|---|---|
| efficiency non-monotonic, peak mid-range | peak at 0.5–0.8 A/cm² | **70.5 % at 0.73** (36 % of rated) ✅ |
| jittery vs smooth degradation | ≥ 3× | **5.33×** ✅ |
| baseline runs, tracks the input | — | ✅, curtailment down from 332 → 64 kWh/day |
| baseline degradation calibrated | 4.0 ± 0.5 µV/h | **4.49 µV/h → 4.50 yr** ⚠️ passes, at the band edge |

Plus 11/11 plant-model checks, 39 tests, `gymnasium.utils.env_checker` clean, and
`scripts/validate_experiment_config.py` (B's) still passing on the merged config.

### ⚠️ One open decision for standup: the calibration is fitted to pre-correction physics

The degradation coefficients in `DECISIONS.md` §5 were solved *before* the plant model was
validated and before the three env bugs below were fixed. Those fixes moved the exposure
integrals — chiefly because the naive baseline now genuinely tracks the resource and so
spends more time at higher current density.

Both gates still pass, but the numbers have drifted from what §5 records:

| | `DECISIONS.md` §5 | now |
|---|---|---|
| `baseline_naive` | 4.15 µV/h | **4.49 µV/h** (band edge; single days reach 5.01) |
| separation | 4.50× | 5.33× |

Re-running `scripts/calibrate_degradation.py --solve` on the corrected physics converges to
`r_base 1.5 · k_j 20.0 · k_ramp 11.0 · dv_cycle 3.5 · r_idle 2.0` → **3.78 µV/h at 4.50×**,
which is better centred and restores the 4.50× separation §5 already quotes.

**I have not applied it.** It would change values recorded in a shared, locked document,
and that is a standup call rather than a unilateral one. Either way §5's quoted rates need
a refresh, because the physics under them moved. Say the word and it is one command.

Docs: [`A_constants.md`](A_constants.md) (every constant cited or corrected) ·
[`A_calibration.md`](A_calibration.md) (the procedure) ·
[`A_methodology.md`](A_methodology.md) (draft paper prose).

---

## ⚠️ Person B — read this, your baseline numbers have moved

Nothing frozen changed: observation shape `(8,)`, action `Box(-1,1,(1,))`, and every
`info` key are exactly as `CONTRACTS.md` §1 specifies. `gymnasium.utils.env_checker`
passes. But **three latent bugs were fixed, and all three change the numbers you get**:

1. **`obs[2]` and `obs[3]` were byte-identical.** The stub did `self.j_prev, self.j = j, j`,
   so `p_set_prev` was a copy of `j_frac`: one of eight observation dimensions was dead
   and the previous setpoint the contract promises was never supplied. Both channels are
   now distinct and carry what they are named after.
2. **`obs[0]` was one step stale.** After stepping at *t* the observation reported
   `p_renew(t)`, but the action it feeds is capped by `p_renew(t+1)` — so no policy,
   including your baselines, could ever match the cap. Now it reports the power available
   at the step being decided. **This is why naive-baseline curtailment fell from 332 to
   63 kWh/day and its yield rose from 126 to 134 kg** — your ref [8] load-following law is
   now actually achieving `P_set = P_renew` as written. Any checkpoint trained before this
   should be re-run.
3. **`P_rated` excluded the BoP parasitic**, so `power_w(j_rated) = 1.02 × P_rated` and an
   action of `+1` could never reach rated current density. `P_rated` is now the total plant
   draw at `j_rated`: **1049 kW** (was 1017 kW). Action `+1` maps exactly onto `j_rated`.

Nothing of yours was overwritten in the merge: `sweep:` (13-point w2 grid) and `train:`
(n_envs 32, PPO on CPU / SAC on CUDA) came through intact, and your
`validate_experiment_config.py` passes on the merged file.

Also, the plant now cannot run below **36.8 kW (3.5 % of rated)** — stack at `j_min` plus
the parasitic floor. The stub gated turn-on on stack power alone, so between 21 and 37 kW
it switched on and drew *more than the renewable input supplied*, quietly breaking the
DECISIONS §3 hard cap. `tests/` pins this now.

Two things for your Day-3 sweep, from the calibrated model:
- One step of `r_deg` is O(10⁻²) after `deg_scale`, against `r_yield` O(1). At `w2 = 1`
  degradation is nearly invisible in the total — expect the interesting band to start well
  above 1. Your 13-point log grid already spans it; the useful resolution is likely between
  w2 = 3 and w2 = 30.
- The ramp-limited baseline is genuinely strong — see the 90-day numbers in
  `A_methodology.md` §D. That is the frontier to beat, not the naive one.

## Person C — the figure is ready

`results/figures/fig_validation.pdf` — vector, 7.0 × 2.7 in (IEEE two-column), 8 pt type.
Panel (a) polarization with the loss budget; panel (b) the non-monotonic efficiency curve.
This is Methodology Fig. 1 and the G1 evidence. Draft prose in
[`A_methodology.md`](A_methodology.md), in the paper's voice, ready for your editing pass.

`scripts/longhorizon_rollout.py` writes `results/*_longhorizon.json` in the `CONTRACTS.md`
§3 schema with `profile_set: "longhorizon_90d"` — schema-checked against every required
key. For `fig_longhorizon()`: cumsum `episodes[].dv_deg_uv` for the 90-day curve, EoL line
at `aggregate.dv_eol_uv`.

**Still owed to you:** the calibration is currently on *synthetic* profiles
(`env.synthetic_day`). DECISIONS §5 specifies real Kutch data. When
`kutch_2019_1min.parquet` lands, ping me.

Heads-up on a gap: `scripts/calibrate_degradation.py` has **no profile-loading flag** — it
only ever rolls out the synthetic generator. `longhorizon_rollout.py` does take
`--profiles`. So landing your parquet needs a small change to the calibration script first;
that is mine to make, and I would rather do it against your real column names than guess
them. Send me the file (or just its schema) and it is a short job:

```bash
python scripts/calibrate_degradation.py --solve     # once it can read the parquet
python scripts/validate_physics.py
python -m pytest tests/ -q
python scripts/longhorizon_rollout.py --days 90 --profiles data/processed/kutch_2019_1min.parquet
```

## Files I touched

Mine only: `src/pemwe/{stack,degradation,env}.py`, `configs/default.yaml`, `tests/*`,
`notes/*`, and two new A-owned scripts (`validate_physics.py`, `longhorizon_rollout.py`).
I did **not** touch `baselines.py`, `train.py`, `evaluate.py`, `profiles.py`, `plots.py`
or `paper/`.

On the merge: I had independently written a `scripts/calibrate_degradation.py` before
pulling. **Yours won** — it is the version in the tree, unchanged. It reaches the same
linear-exposure insight and adds a constraint mine lacked (the separation must come from
*both* ramping and cycling, ≥20 % each), which is the better model. The only thing I kept
from mine is `DegradationModel.basis()`, which makes that linearity explicit in the model
rather than re-derived in the script, and a test pinning it to 1e-12.

One thing to flag: `longhorizon_rollout.py` writes into `results/`, which is B's. It is
append-only with unique `run_id`s per the standing rules, and `results/*.json` is
gitignored — but say if you would rather it wrote elsewhere.

## Environment setup

The repo had no `.venv` (it is gitignored, and README describes it as pre-existing). I
recreated it at `.venv/` with Python 3.13 + gymnasium 1.3.0 and the A-lane dependencies.
**`torch` and `stable-baselines3` are not installed** — B's lane, and not needed by
anything above. `pip install -r requirements.txt` completes it.

**Correction to something I said before reading `BENCHMARK.md`:** I claimed no GPU was
needed anywhere. That is right for PPO and right for my whole lane (the env is pure NumPy),
but **wrong for SAC**. B's measurements show SAC on CUDA at `n_envs=32` with batched
updates runs 6,520 steps/s against 2,450 on CPU — 2.7× faster. The reason my reasoning
missed it: SAC does many gradient updates per environment step, so once those are batched
across 32 envs the device work stops being negligible. PPO stays CPU-bound, exactly as
`BENCHMARK.md` records. Trust the measurements over my generic argument.
