# Standup — Person A: the environment is FROZEN

Gate **G1 passes on all three checks**. Everything below regenerates from the repo.

```bash
python scripts/smoke_test.py            # G1.1 G1.2 G1.3 -> PASS PASS PASS
python scripts/validate_physics.py      # 10/10 plant-model checks PASS + the figure
python -m pytest tests/ -q              # 38 passed
python scripts/calibrate_degradation.py # re-solves the coefficients, all targets met
python scripts/longhorizon_rollout.py   # 90-day rollout -> results/*.json
```

| Gate G1 | Target | Now |
|---|---|---|
| efficiency non-monotonic, peak mid-range | peak at 0.5–0.8 A/cm² | **70.5 % at 0.73** (36 % of rated) ✅ |
| jittery vs smooth degradation | ≥ 3× | **3.48×** ✅ |
| baseline runs, tracks the input | — | ✅, curtailment down from 332 → 63 kWh/day |
| baseline degradation calibrated | 4.0 ± 0.5 µV/h | **4.15 µV/h → 4.87 yr** ✅ |

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

Also, the plant now cannot run below **36.8 kW (3.5 % of rated)** — stack at `j_min` plus
the parasitic floor. The stub gated turn-on on stack power alone, so between 21 and 37 kW
it switched on and drew *more than the renewable input supplied*, quietly breaking the
DECISIONS §3 hard cap. `tests/` pins this now.

Two things for your Day-3 sweep, from the calibrated model:
- One step of `r_deg` is O(10⁻²) after `deg_scale`, against `r_yield` O(1). At `w2 = 1`
  degradation is nearly invisible in the total — expect the interesting band to start well
  above 1, which is consistent with the `{0.1, 1, 10, 100}` sweep already planned.
- The ramp-limited baseline is genuinely strong: on the 90-day rollout it buys **+1.06 yr
  (+19 %) of life for −3.4 % H₂**. That is the frontier to beat, not the naive one.

## Person C — the figure is ready

`results/figures/fig_validation.pdf` — vector, 7.0 × 2.7 in (IEEE two-column), 8 pt type.
Panel (a) polarization with the loss budget; panel (b) the non-monotonic efficiency curve.
This is Methodology Fig. 1 and the G1 evidence. Draft prose in
[`A_methodology.md`](A_methodology.md), in the paper's voice, ready for your editing pass.

`scripts/longhorizon_rollout.py` writes `results/*_longhorizon.json` in the `CONTRACTS.md`
§3 schema with `profile_set: "longhorizon_90d"` — schema-checked against every required
key. For `fig_longhorizon()`: cumsum `episodes[].dv_deg_uv` for the 90-day curve, EoL line
at `aggregate.dv_eol_uv`.

**Still owed to you:** the calibration is currently on *synthetic* profiles. DECISIONS §5
specifies real Kutch data. When `kutch_2019_1min.parquet` lands, ping me — it is four
commands and the numbers move together:

```bash
python scripts/calibrate_degradation.py --profiles data/processed/kutch_2019_1min.parquet --write
python scripts/validate_physics.py
python -m pytest tests/ -q
python scripts/longhorizon_rollout.py --days 90 --profiles data/processed/kutch_2019_1min.parquet
```

## Files I touched

Mine only: `src/pemwe/{stack,degradation,env}.py`, `configs/default.yaml`, `tests/*`,
`notes/*`, and three new A-owned scripts (`validate_physics.py`,
`calibrate_degradation.py`, `longhorizon_rollout.py`). I did **not** touch `baselines.py`,
`train.py`, `evaluate.py`, `profiles.py`, `plots.py` or `paper/`.

One thing to flag: `longhorizon_rollout.py` writes into `results/`, which is B's. It is
append-only with unique `run_id`s per the standing rules, and `results/*.json` is
gitignored — but say if you would rather it wrote elsewhere.

## Environment setup

The repo had no `.venv` (it is gitignored, and README describes it as pre-existing). I
recreated it at `.venv/` with Python 3.13 + gymnasium 1.3.0 and the A-lane dependencies.
**`torch` and `stable-baselines3` are not installed** — B's lane, and not needed by
anything above. `pip install -r requirements.txt` completes it.

No GPU is needed anywhere in this project: the env is pure NumPy, and with a 1-D action
and a small MLP, SB3 on CPU with `n_envs: 8` is the right configuration (a GPU is
typically *slower* here — per-step kernel-launch overhead dominates, and the bottleneck is
stepping the env 1440× per episode).
