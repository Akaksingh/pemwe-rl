# Stage 1 research audit — PEM electrolyzer RL paper

Status: audit only. Nothing was trained, no source file was modified, no result was regenerated.
Date: 2026-11-05. Scope and limits are in section 1; read them before relying on any item.

Evidence labels used throughout:

- **[RAN]** I executed code or arithmetic on the repo's own files/parameters and got the stated number.
- **[READ]** Established by reading the code/config/text; not executed.
- **[WEB]** Checked against an external source; the source is named.
- **[UNVERIFIED]** I could not check it. It may be right or wrong.

---

## 1. Scope and limits of this audit

Read in full: `src/pemwe/{stack,degradation,env,baselines,evaluate,train,profiles}.py`,
`configs/default.yaml`, `CONTRACTS.md`, `DECISIONS.md`, `BENCHMARK.md`, `SERVER.md`,
`notes/{A_methodology,A_calibration,B_handoff}.md`, `paper/{main.tex,refs.bib,REFERENCES.md}`,
`scripts/{run_sac_sweep.sh,reward_landscape.py,validate_experiment_config.py}`, test names, and the
65 SAC result files plus 2 baseline files in `results/`.

**Not read:** `README.md`, `PLAN.md`, `briefs/`, `notes/A_constants.md`, `notes/A_handoff.md`,
`scripts/` other than the three above (notably `build_profiles.py` beyond the hybrid line,
`tune_baseline.py`, `calibrate_degradation.py`, `longhorizon_rollout.py`, `make_paper_figures.py`,
`validate_physics.py`, both bench scripts), `src/pemwe/plots.py`, test bodies, the figures.

**Could not do:**

- The local `.venv` is broken (its `pyvenv.cfg` points at another user's Python at
  `C:\Users\akaks\...`) and has no torch/gymnasium/SB3, so I could not run the test suite or any
  environment rollout. Checks marked [RAN] used `stack.py` loaded directly plus the stored JSON
  results.
- `models/` is empty locally: no checkpoints. Nothing learned can be re-evaluated here.
- `results/` has **no PPO result files** (only `smoke_ppo.json`, a single w2=1.0 smoke run).
- MDPI returns HTTP 403 to automated fetches; I did not try to work around that.

---

## 2. Ranked findings

| # | Severity | Finding | Evidence |
|---|---|---|---|
| C1 | Critical | SAC runs at an update-to-data ratio of 1/32, not 1:1 as documented | [RAN] measured on SB3 2.9.0 with the repo config: 0.032 |
| C2 | Critical | SAC is beaten on its own training reward by a one-line controller at low w2 | [RAN] stored results |
| C3 | Critical | The headline lifetime (7.39 yr) is a linear extrapolation; the learned policy is never run with persistent degradation and cannot observe degradation | [READ] + [RAN] |
| C4 | Critical | Model's own crossover value puts H2-in-O2 above 4 vol% for all j < 0.79 A/cm²; the operating-state model is also too thin to call "physically consistent" | [RAN] |
| C5 | Critical | Several manuscript statements are wrong or unverifiable (a 4.0% claim that is 2.0%; a 404 DOI; two anchor references unverified) | [RAN]/[WEB] |
| M1 | Major | All sub-hourly variability is synthetic and unvalidated; `k_ramp` is calibrated against it, so the two are confounded | [READ] |
| M2 | Major | PPO-vs-SAC conclusion is confounded and its raw results are not in the repo | [READ] |
| M3 | Major | Statistics: 5 seeds, no CIs/tests; w2 range was chosen using held-out days | [READ] |
| M4 | Major | Reward/degradation structure: arbitrary w2 scale, shutdown incentive is a model artefact, double counting of start/stop | [READ]/[RAN] |
| M5 | Major | HPC numbers are short, single-shot, measured on a since-changed environment, and conflated with C1 | [READ] |
| M6 | Major | Reproducibility: results gitignored, no checkpoints, broken venv, no code/config hash in result files | [READ] |
| M7 | Major | Resource cap (1.0 MW) is below plant rated power (1.049 MW): rated point is unreachable | [RAN]/[READ] |
| m* | Minor | See section 3.4 | |

---

## 3. Findings in detail

### 3.1 Critical

#### C1. SAC update-to-data ratio is 1/32, not 1:1

Config: `n_envs=32, train_freq=32, gradient_steps=32` (`configs/default.yaml`, `train.py:185`).
Every document (DECISIONS §4, BENCHMARK, SERVER, the paper, `run_sac_sweep.sh`,
`validate_experiment_config.py`) states this "holds the update-to-step ratio at exactly 1:1".

SB3's `collect_rollouts` increments `num_collected_steps += 1` **once per vectorized step**
and stops when `num_collected_steps < train_freq.frequency` is false (SB3 `master`,
`off_policy_algorithm.py` and `common/utils.py::should_collect_more_steps`) [WEB]. With 32
envs, `train_freq=32` therefore collects 32 × 32 = **1,024 transitions** per phase, followed by
`gradient_steps=32` updates.

[RAN] UTD = 32 / 1024 = **0.031**. Over 2M steps that is ≈ **62,500** gradient updates, against
≈ 2,000,000 for a true 1:1 run. The `gradient_steps=8` variant that the docs call "4× fewer
updates" is 1/128.

Consequences:

1. "Same learning dynamics as the single-env default at 26× the throughput" is false. The
   measured 26× (246 → 6,520 steps/s) is of the same order as the 32× reduction in gradient
   work. The speedup is largely less learning per step, not free parallelism.
2. All SAC results are from a heavily under-updated learner. C2 shows direct evidence of that.
3. The guards (`run_sac_sweep.sh:50`, `validate_experiment_config.py`) enforce
   `gradient_steps == train_freq == n_envs`, i.e. they lock in the mistake. A true 1:1 needs
   `gradient_steps = train_freq * n_envs = 1024`.
4. Every HPC claim derived from it (GPU 2.7×, PPO-on-CPU, "SAC is gradient-bound at 100 ms per
   update") is conditional on the wrong workload. At UTD=1 the gradient cost is 32× higher, which
   changes the CPU/GPU balance completely.

**Measured (2026-11-05, [RAN]):** SB3 2.9.0 (the version `SERVER.md` documents), the repo's
own `build_model("sac", cfg, ...)` with the unmodified config (`n_envs=32, train_freq=32,
gradient_steps=32, batch 512`), 40,960 environment steps on a GPU: SB3 performed
**992 gradient updates for 30,960 transitions after `learning_starts`: ratio 0.032, not 1.0.**
The installed source shows the same rule (`num_collected_steps += 1` per vectorized step;
`should_collect_more_steps` returns `num_collected_steps < train_freq.frequency`). Test T1
(section 7) should be added to the suite so this cannot silently regress.

#### C2. A hand-written controller beats SAC on SAC's own objective

`results/ablation_threshold.json` (and paper Table I) gives the "threshold removed" controller on
the 72 held-out days: **205.0 kg/day, 82.7 µV/day, 7.0 transitions/day**. Removing the threshold
makes `target = p_renew`, i.e. a constant action of +1.

SAC at the lowest degradation weights (mean of 5 seeds, same 72 days) [RAN]:

| w2 | SAC H2 kg/d | SAC dV µV/d | curtailed kWh/d | reward of constant-action-+1 vs SAC mean |
|---|---|---|---|---|
| 0.100 | 195.9 | 84.5 | 473 | +4.7% for the hand-written controller |
| 0.156 | 195.4 | 84.5 | 499 | +5.0% |

At these weights the constant policy yields more hydrogen **and** (marginally) less degradation, so
it dominates the trained agent on both axes. The ramp-shaping term is < 2 reward units per day
and does not change the comparison. SAC also curtails ~470–500 kWh/day there for no reward
reason. This is an optimisation shortfall, not a trade-off. It is consistent with C1.

Implication: the existing frontier is likely shifted down, and the "reach, not dominance"
narrative cannot be trusted until SAC is re-run with a true UTD. Also note the paper's Fig. 1
caption plots only the naive and ramp-limited controllers; the threshold-removed controller
(Table I) is not on the frontier plot although it dominates the SAC points at w2 ≤ 0.156.

(Assumes `ablation_threshold.json` was computed on the same 72-day test split, as paper Table I
states. I did not read the script that produced it.)

**Pilot at the corrected update ratio (2026-11-05, [RAN]).** SAC, w2 = 0.1, UTD = 1 (`train_freq=32,
gradient_steps=1024`, 32 envs), **300k env steps (≈300k updates)**, seeds 0 and 1, evaluated
in-sample on all 293 **train** days (test split untouched; in-sample favours SAC):

| controller | H2 kg/d | µV/h | transitions/d | curtailed kWh/d |
|---|---|---|---|---|
| constant action +1 (threshold removed) | 219.6 | 3.54 | 6.0 | 22 |
| SAC seed 0 | 204.3 | 3.96 | 14.1 | 774 |
| SAC seed 1 | 201.0 | 3.80 | 12.0 | 917 |
| naive rule as published | 217.2 | 3.95 | 17.1 | 162 |
| ramp-limited | 214.7 | 3.33 | 7.5 | 287 |

Paired per day against constant +1: SAC had **less hydrogen, more degradation and lower reward on
293 of 293 days** for both seeds (mean reward −1,539 and −1,861 per day, about 7–9%). On the
representative day the policy commanded on average 87–90% of available power, with 45–58% of
steps below 95% of availability, and it also cycles about twice as often. **The pilot failed its
go/no-go criterion:** correcting the update ratio alone, at this budget, does not recover the
trivial policy. Relative gap on different days: old runs −4.4% (test, 2M steps, UTD 0.03); pilot
−7 to −9% (train, 300k steps, UTD 1). Not directly comparable; neither run has converged to the
simple optimum.

Not yet tested (hypotheses, not findings): (a) `gamma = 0.999` (`train.py`) gives Q-values on the
order of 10⁴ while the action-dependent part of the reward is a small fraction of that, so the
critic may be unable to resolve action differences; the day's returns hardly depend on the future
(degradation state is unobservable, C3), so such a long horizon has no obvious justification;
(b) no reward/observation normalisation; (c) the optimum sits on the action boundary (u = 1), which
a tanh-squashed Gaussian approaches only asymptotically; (d) 300k steps may simply be too few.
Cheapest next checks: gamma ∈ {0.99, 0.95} and a reward rescale, each at the same pilot budget.

**Gamma pilots (2026-11-05, [RAN]).** Same setup as the pilot above (w2 = 0.1, UTD = 1, 300k steps,
32 envs, seed 0, in-sample on 293 train days), only `train.sac.gamma` changed:

| gamma | H2 kg/d | dV µV/d | transitions/d | curtailed kWh/d | reward vs constant +1 |
|---|---|---|---|---|---|
| 0.999 (earlier pilot, seed 0) | 204.3 | 94.9 | 14.1 | 774 | −1,539 / day |
| 0.99 | 210.0 | 85.0 | 7.1 | 493 | −954 / day |
| 0.95 | 212.6 | 81.1 | 3.8 | 373 | −699 / day |
| constant action +1 | 219.6 | 84.9 | 6.0 | 22 | 0 |

Every metric improves monotonically as gamma falls, which supports hypothesis (a): the very long
horizon was part of the problem. But one seed per setting, in-sample, 300k steps: this is a
trend, not an established effect. SAC is still below the simple controller in reward on 293 of 293
days at gamma = 0.95; it degrades less (81.1 vs 84.9 µV/d) and cycles less (3.8 vs 6.0) but gives up
7 kg/day of hydrogen, more than w2 = 0.1 values.

**Round 2 (2026-11-05, [RAN], same setup, 293 train days, in-sample):** gamma 0.9, 0.8, a replicate seed at
0.95, and 0.95 with the reward divided by 10.

| run | H2 kg/d | dV µV/d | transitions/d | reward vs constant +1 |
|---|---|---|---|---|
| gamma 0.8, seed 0 | 212.9 | 80.1 | 3.1 | −665 [−678, −652] |
| gamma 0.9, seed 0 | 212.9 | 80.9 | 3.7 | −667 [−680, −654] |
| gamma 0.95, seed 0 | 212.6 | 81.1 | 3.8 | −698 [−712, −684] |
| gamma 0.95, seed 1 | 212.5 | 80.9 | 4.2 | −706 [−720, −691] |
| gamma 0.95, reward / 10 | 212.5 | 80.8 | 3.6 | −708 [−721, −696] |
| tuned slew limit | 214.7 | 80.0 | 7.5 | −481 [−513, −451] |

Findings: (i) everything at gamma ≤ 0.95 lands in the same place; seed noise is ~10 reward units, so 0.8,
0.9 and 0.95 are not distinguishable; (ii) reward scale is **not** the problem (−708 vs −698); (iii) the
discount factor was a real cause (−1,539/−1,862 at 0.999); (iv) SAC is still ~3% below the constant-action
rule and also below the tuned slew limit on every day: at w2 = 0.1 the optimum is the action boundary
(u = 1), which a tanh-squashed policy approaches slowly (hypothesis, not tested). The informative regime is
a higher w2 where the optimum is interior; that is E5.

**Baseline sensitivity grid (2026-11-05, [RAN], 72 held-out test days, rule-based controllers only):**
projected life of the published rule ranges from 1.2 to 19 years across the 13 degradation-coefficient
variants (a factor of 16), while the three controllers differ by tens of percent at nominal coefficients.
Lifetime ordering is identical across the five resource scenarios but changes in 6 of 13 coefficient
variants; hydrogen ordering never changes. The grid also reproduces the paper's baseline numbers exactly
(201.8/96.9, 205.0/82.7, 199.5/78.3 kg and µV per day), which validates the new harness against the old
evaluation code.

**E3 — update-ratio study (2026-11-07, [RAN], gamma 0.95, w2 = 0.1, 2 seeds, 300k steps, every 5th train
day, checkpoints every 50k steps).** UTD 1/32, 1/8, 1/2 and 1 all sit on the same plateau, 3.0–3.3% below the
constant-action rule, from the *first* checkpoint onward (UTD 1/32: 3.2% after 0.4 min and ~1.2k updates).
Final wall-clock (four runs sharing the GPU): 2.7 min / ~9k updates at UTD 1/32 against 55 min / ~290k updates
at UTD 1, a factor of 20, for no quality difference. So the update-ratio error (C1) is real but did **not**
cause the poor legacy SAC results; the discount factor did. The corrected-ratio runs are a 20x waste of learner
time for this problem.

**What the policy learns ([RAN] probe, 12 test days):** the UTD-1 policy commands on average 94% of available
power, stays off 95% of the time when the resource is only 3.5–5% of rated (avoiding marginal start-ups), and
commands 98% at high availability. At w2 = 20 it commands 70% on average and is off in 24% of steps where the
resource would allow operation.

**E5 — frontier on the held-out test days (2026-11-07, [RAN], gamma 0.95, UTD 1, 300k steps, 2 seeds per w2,
test split used once).** Rule grids (slew limit with/without the idle threshold, power cap, 35 points) give a
non-dominated envelope. SAC at w2 = 1.41 and 5.32 sits on or slightly below it (−1.1% and −0.3% hydrogen at the
same degradation); SAC at w2 = 12.9 makes 181 kg/day at 56.3 µV/day, 24% (35 kg) more than the envelope at that
degradation; w2 = 20 gives 166 kg/day at 49.0 µV/day against 89 kg/day for the lowest-degradation rule point.
The earlier SAC runs stopped at 65.6 µV/day. Matched comparison under coefficient/resource shifts (policies
not retrained): against a slew limit matched to the policy's degradation, SAC makes 14% (w2 = 12.9) to 86%
(w2 = 20) more hydrogen with a median life ratio of ~1.0; it outlives the matched rule in 10 of 13 coefficient
variants and 5 of 5 scenarios (w2 = 12.9) but not in all (worst life ratio 0.85). **Defensible claim:** more
hydrogen at the same degradation in the low-degradation regime, lifetime roughly a tie that depends on the
coefficients; **not** a longer life.

**Open limits of this evidence:** two seeds per setting; gamma chosen from in-sample pilots; one-dimensional
rule families only (a richer rule or MPC may close part of the gap); w2 has no economic calibration; PPO not
re-run, so the old PPO claim is withdrawn.

#### C3. Lifetime claims rest on a linear extrapolation of reset-to-zero days

- Training and evaluation episodes reset `dv_deg_uv` to 0 (`persist_degradation: false`).
  `obs[4] = dv/dv_eol` is therefore ≤ ~4.5e-4 inside an episode: the agent effectively cannot
  observe degradation. Degradation reaches it only through the reward. [READ]
- [RAN] Feedback into voltage: 0.005% of rated cell voltage per day, 0.47% over 90 days, 9.5%
  over 5 years at 4 µV/h. The "degradation reduces future yield" coupling is real in the model
  but is **never exercised by any learned policy**: the 90-day persistent rollout
  (`results/*_longhorizon.json`) exists for the two rule-based controllers only, and the
  long-horizon figure caption in `main.tex` says so.
- `projected_life_years = dv_eol / mean_rate / 8760` (`degradation.py:76`) is a constant-rate
  extrapolation from 72 sampled days. 7.39 years is a model output on a calibrated scale, not a
  lifetime. The paper does say "projected" and, in the long-horizon section, that the
  extrapolation is not a prediction. The abstract and conclusion do not say that the 7.39 years
  comes from reset-to-zero single days with no persistent-degradation run of the learned policy.
- [RAN] `dv_eol_uv = 177000` is 10% of **1.77 V**, but the model's own rated cell voltage is
  **1.8458 V** (10% = 184,577 µV). A 4.3% inconsistency in the absolute lifetime scale. It is
  partly absorbed by the calibration to "5 years", but it should be one number derived from the
  model, not a literal.

#### C4. Low-load physics and operating states

What the code does [READ]: the action is a fraction of available power. If the commanded power is
below `p_min_on` (36.8 kW = 3.51% of 1,049.4 kW) the plant is OFF, current is zeroed, and the
commanded power is zeroed. So the agent **cannot** operate below 36.8 kW; sub-threshold
commands are shutdowns. The question "is the controller exploiting an invalid region" must
therefore be asked about the region just above it.

- 36.8 kW is a model-derived floor (stack at `j_min = 0.05 A/cm²` plus parasitic), not a physical
  specification. `j_min = 2.5%` of rated current is below the 5–10% turndown that the config
  comment itself quotes for PEMWE.
- [RAN] Using the model's own `j_cross = 0.01654 A/cm²`, hydrogen in the anode gas stream
  (dry basis, `x = r/(1+r)`, `r = 2·j_cross/j`) is:

  | j (A/cm²) | % of rated | H2 in O2 |
  |---|---|---|
  | 0.05 | 2.5 | 39.8% |
  | 0.20 | 10 | 14.2% |
  | 0.40 | 20 | 7.6% |
  | 0.73 (the efficiency peak) | 36.5 | 4.3% |
  | 1.00 | 50 | 3.2% |
  | 2.00 | 100 | 1.6% |

  Staying under 4 vol% (the standard H2-in-O2 flammability limit; please confirm the figure and
  the usual alarm/trip levels in your safety reference) needs j ≥ 0.79 A/cm² (40% of rated);
  under 2% needs j ≥ 1.62 (81%). Either `j_cross` (30 bar, 70 µm, permeability 2e-11) is too
  high for a real stack with recombination layers and thicker membranes, or the whole low-load
  half of the efficiency curve that motivates the control problem is unsafe. The model cannot
  currently say which. This needs a literature check of permeability and of real minimum loads.
- Operating states are one boolean (`is_on`). There is no distinction between OFF, hot standby
  and active; start-up is instantaneous, costs no energy and no time, and carries one
  1 µV penalty. BoP power is zero while off (`power_w`), so standby parasitics are not charged.
  There are no minimum up/down times and no thermal state.
- What I could not do: the actual distribution of j under the learned policies. Only one
  trajectory per run is stored, and no checkpoints exist locally. This is the first thing to
  measure (T7).

Proposal that does not change existing assumptions: add an `operating_model` config switch
(`legacy` default, reproduces all current numbers bit-for-bit; `v2` for the new states). See
section 5.

#### C5. Manuscript and bibliography errors

1. **"The baseline is 4.0% better at the shared operating point" is 2.0%.** [RAN] Interpolating
   the 5-seed-mean SAC frontier at the ramp-limited baseline's 78.3 µV/day gives 195.5 kg/day
   against the baseline's 199.5 (1.99%). The 4.0% in the text uses 191.8, which is the w2=20
   endpoint at 65.6 µV/day. The conclusion repeats the 4.0%. (The direction is unchanged:
   the baseline is better there.)
2. **DOI `10.3390/16073605` (drl_scheduling_2026) returns 404.** [WEB] *Fixed 2026-11-05:* the paper is
   Applied Sciences 16(7), 3605 and `10.3390/app16073605` resolves to it. Same-day verification of
   the other flagged entries [WEB, titles/DOIs only, full texts not read]: `bernhardt2026degradation`
   exists (Bernhardt, Lange, Ilse; doi 10.1002/admt.202502419; authors added); `degmodels2025` exists but
   is in *Journal of Power Sources* 655 (2025) 238003 (Makhsoos et al.), **not** *Electric Power Systems
   Research* as the bib said (corrected); **`intermittency2025` and `awe_rl_2025` were not found by title
   search** (a verified related review, Sayed-Ahmed et al., RSER 189 (2024), was added as
   `sayedahmed2024dynamic`); the DOIs of both MDPI *Hydrogen* papers resolve, but their titles/claims
   could not be read (publisher returns 403).
3. **Baseline-law source `10.3390/hydrogen7020068`** resolves to "Dynamic Control of a PV/T
   Electrolysis System for Hydrogen and Hot-Water Production: Multi-Regional Analysis with
   Machine Learning" (Hydrogen 7(2), 68; the bib title omits the subtitle). The only abstract text
   I could see describes rule-based control with **hysteresis thresholds on hydrogen-storage
   state of charge**. That is not obviously the `P_set = min(P_renew, P_rated) if P_renew ≥
   P_idle else P_idle` law with a 10% threshold that the paper calls "used verbatim". [UNVERIFIED]
   Full text needed.
4. **Calibration anchor `10.3390/hydrogen7020076`**: could not be read or found by search.
   The paper uses it for "~5 years" *and* for "5 → 7.5 years under predictive control".
   Every absolute lifetime in the paper hangs on that single reference. [UNVERIFIED]
5. Bib entries `degmodels2025`, `intermittency2025`, `awe_rl_2025` have no authors or DOI;
   `bernhardt2026degradation` has two authors where REFERENCES.md says "et al." and no DOI;
   `drl_scheduling_2026` DOI is bad (item 2). Not submittable as is.
6. **"The resource averages 31% of rated power" (Results, Conclusion; `DECISIONS.md` §1) is not what
   the shipped data gives.** [RAN, `scripts/make_scenarios.py`, shipped parquet] The hybrid input
   averages **0.442 of 1 MW = 0.421 of the plant's 1,049 kW rating**; it is below the plant's
   minimum power (36.8 kW) for 8.3% of minutes, in ≈2.7 intervals/day of ≈44 min on average. The
   31% appears to be a leftover from the early synthetic profile. The mechanism paragraph that
   leans on it ("the stack spends most of its time well below the stress threshold") needs
   re-checking against the real distribution.
7. Paper Section IV still carries `\TODO`s (authors, data provenance, hyperparameter table).
   The data sidecar says `source: open_meteo` (ERA5 via Open-Meteo, conversions are ours), so
   the ERA5 and PVWatts branch of the TODO applies; the Renewables.ninja citation is not needed.

### 3.2 Major

#### M1. Variability is synthetic; calibration is confounded with it
All 1-minute structure comes from OU processes (PV: τ = 12 min, amplitude 0.55·depth·base; wind:
τ = 3 min, turbulence intensity 12%) added to hourly ERA5 [READ, `profiles.py`]. No
high-frequency measurement validates these. `k_ramp` was then calibrated *against these
synthetic ramps* and lands at 94% of its interval; the notes defend that as "the data put it
there", but the data is the assumed noise. Lower the assumed variability and the calibration
would push `k_ramp` up, so the controller ranking and the "cycling dominates" mechanism are
conditional on an untested noise model. Also: a single stochastic realisation (seed 0) of the
whole year, a single year (2019), and a random day-level split (adjacent days, correlated
weather, fall in both train and test). Experiments A and B of your plan address exactly this.

Note on mechanism wording: in the naive baseline the per-day split is base 30.7, stress 3.6,
ramp 24.8, cycle 32.1, idle 4.9 µV (total 96.0) [READ, `A_calibration.md`]. Cycling + ramping
is 59% of it; base is 32%. "Dominated by cycling" is stronger than that table supports.

#### M2. PPO vs SAC
- No PPO results exist in the repo, so Table II ("PPO span 0.0%") cannot be checked.
- PPO configuration [READ]: `ent_coef=0.0`, `gamma=0.999`, no reward or observation
  normalisation (reward scale ~35 per step), unsquashed Gaussian policy clipped by the
  environment. A mean action that drifts outside [−1, 1] gives a saturated constant action
  that is identical across weights and seeds. That is the "byte-identical policies" symptom
  that `DECISIONS.md` §4 and the paper treat as a finding. It is at least as consistent with a
  saturation artefact as with an algorithmic property.
- Constant action +1 equals the threshold-removed controller (205.0 kg/day, 82.7 µV/day). If PPO
  really converged to it, PPO would sit above SAC's whole frontier on yield. The paper's framing
  ("algorithm choice is a precondition for observing the trade-off") is not supported.
- SAC is itself affected by C1. The comparison needs a re-run on equal footing (Experiment D).

#### M3. Statistics and test-set hygiene
- 5 seeds; the paper says error bars are the standard deviation across seeds, but the stored
  `aggregate.*_std` is the standard deviation across **days** [RAN: 88.49 both ways]. There are
  no confidence intervals, no paired per-day tests, no effect sizes.
- [RAN] Per-seed gaps are consistent in sign (SAC at w2=20 vs ramp-limited: −6 to −9 kg and
  −9.5 to −14 µV in all five seeds), so the "reach" ordering is not a seed fluke within this
  model. That is a robustness observation about seeds only; it does not address C1/C2.
- `reward_landscape.py` calls `env_profiles(N_DAYS, split="test")`: the w2 range cut from
  0.1–100 to 0.1–20 was decided by scoring policies on **held-out test days**. Mild, but it is
  test-set use for a design decision and should be redone on a validation split.
- There is no validation split at all: train (293) / test (72). Hyperparameters were not tuned,
  which avoids leakage, but also means nothing was tuned.

#### M4. Reward and degradation structure
- [RAN] Exchange rate: 1 kg H2 = 100 reward; 1 µV = 10·w2 reward. So w2=1 prices 1 µV at 0.1 kg
  of hydrogen, w2=5 at 0.5 kg, w2=20 at 2 kg. No economic basis is given for any point on the
  sweep, which is why a levelised-cost analysis is needed before saying which point is "good".
- The shutdown threshold (w2 ≈ 20) exists because `r_idle` (0.75 µV/h) is smaller than `r_base`
  (1.75 µV/h): a stack that is off degrades more slowly than one that is on, in this model. That is
  a modelling choice that bounds the sweep; it is not a property of PEM stacks demonstrated by
  evidence.
- Start/stop is penalised three times: the cycle term (1 µV per transition), the ramp term
  (|Δj| from 0 to j or j to 0 is also charged at `k_ramp`), and the `w3·|Δj|/j_r` shaping term.
  The third is not physical and is held fixed, but it shapes the policy independently of w2.
- Terminology: `n_cycles` counts every ON↔OFF **transition**. "22.9 events per day" is about
  11.4 start/stop cycles. The mechanism-figure caption says "cycles".
- Units and accumulation [READ]: all five terms are dimensionally consistent; the ramp term
  reduces to `k_r·|Δj|/60` per step independent of dt. No error found there.

#### M5. HPC evidence
- The manuscript has no HPC section. Only implementation remarks ("PPO 30% slower on GPU, SAC
  2.7× faster") appear, and they come from `BENCHMARK.md`.
- Benchmark limits [READ]: SAC benchmark is 6,000 steps (replay buffer nearly empty, so
  sampling cost is understated); sustained rate under contention was 2,451 steps/s against
  6,520 (gap unexplained); `n_envs` was only varied over {1, 8, 32 (64 for PPO)}; no speedup or
  efficiency curve, no utilisation, memory or time-to-quality measurement.
- `BENCHMARK.md` explains the env-bound result by "a 40-iteration bisection in `j_from_power`".
  `stack.py` says that bisection has since been replaced by table inversion (~30× cheaper). The
  env-bound vs GPU-bound conclusion, and the device split, were measured on a different
  workload and need re-measuring.
- Steps/s is not a measure of equal work once UTD differs (C1). The right metric is time to a
  predefined policy quality at a fixed, stated UTD.
- **Measured 2026-11-05 on the A5000 workstation [RAN]** (`results_hpc/`, first pass, clean
  machine): raw simulator 12.7k steps/s on one core, 69.5k on 8 workers, ~76k on 16 (saturates at
  the 8 physical cores; SMT adds ~10%); SB3 `DummyVecEnv` 8.2k → 10.7k steps/s from 1 to 32 envs.
  SAC learner (updates only): GPU 166–175 updates/s, **flat across batch size 256–2048** (launch/
  Python-overhead-bound, not compute-bound); CPU 97 (4 threads), 110 (8), 92 (16) updates/s. The
  repo's earlier statement that one CPU update costs "~100 ms" (≈10 updates/s) is not reproduced:
  it is ≈10 ms here, so the GPU advantage is ≈1.6×, not the documented 2.7×.
- The first benchmark script crashed in its SB3 section (no `__main__` guard) and never wrote its
  CSV; the numbers above come from the log of its first pass, which is complete for the sections
  cited. Replaced by `scripts/bench_hpc_env.py` (pinned/unpinned/SMT placement, 5 repeats).
- Local hardware [RAN, Windows]: i5-12450H (8 cores / 12 threads), 7.7 GB RAM, GTX 1650 4 GB.
  That cannot support the 32-environment workload. The documented server is `hcd-dept-h200`
  (4× H200, EPYC 9535, 128 cores) at a private address with a weekly GPU quota; its current
  availability is unknown to me (see questions).

#### M6. Reproducibility
`results/*.json` and `models/` are gitignored; the 65 SAC result files exist only locally and the
checkpoints that produced them are not here. Result files carry no code commit, config hash,
library versions or hardware record. The local venv does not run. PPO raw results are absent.
Hyperparameters other than `n_envs/train_freq/gradient_steps/batch_size/device` are literals
inside `train.py`, contrary to the repo's "no magic numbers" rule.

#### M7. Resource cap below plant rating
`hybrid_w = min(pv_w + wind_w, 1.0e6)` while `P_rated = 1,049.4 kW` [READ, `build_profiles.py:77`;
RAN for `P_rated`]. The plant can never be commanded to its rated current (max available power is
95.3% of rated), `obs[0]` never exceeds 0.953, and the claim "a full power command reaches exactly
rated" holds only in the stack unit test, not in the full system. The consequence is small but
it should be stated, or the plant rating and the resource scaled consistently.

### 3.3 Checks that passed [RAN]
`stack.py` reproduced: E_rev = 1.2463 V (+48.8 mV Nernst), rated cell voltage 1.8458 V
(paper: 1.846), 1.5067 V at 0.05 A/cm² (paper: 1.507), rated plant power 1,049.4 kW, parasitic
15.7 kW, minimum power 36.8 kW (3.51%), `j_cross` 0.01654 A/cm². Faraday's law and the LHV
voltage (241.8 kJ/mol ÷ 2F = 1.2531 V) are consistent. I did not re-run the full efficiency curve
(peak 70.5%) or the 90-day rollout. The 65 SAC files reproduce the paper's frontier numbers
(191.8–195.9 kg/day, 65.6–84.5 µV/day, 10.7 → 2.0 transitions/day, 694 kWh/day curtailment,
7.39 yr).

### 3.4 Minor
- `h2_rate_norm` is the rate of the step just taken, one step stale relative to `obs[0]`: no
  leakage found; no future information enters the observation. [READ]
- Time-of-day features let a policy memorise the diurnal cycle for one site; fine for the study,
  worth a sentence.
- `evaluate.py` rolls the whole test set twice to choose the representative day.
- Evaluation uses deterministic SAC actions only; stochastic evaluation is not reported.
- The high-current stress term (threshold j/j_r = 0.6) is almost inactive on this resource
  (3.6 of 96 µV for the naive baseline); no conclusion about it is supported.
- `env.synthetic_day` placeholder remains in `env.py`.
- `eta_lhv` in `info` is 0 when off; the episode mean covers ON steps only (documented).
- Local `.venv/pyvenv.cfg` records `D:\rl\.venv`; the repo now lives at `D:\D Drive\rl`.

---

## 4. File-by-file

| File | Does | Matches manuscript? | Issues |
|---|---|---|---|
| `stack.py` | Polarization, crossover, power, H2, inversion table | Yes for numbers re-run | C4 (crossover vs safety, j_min, one `is_on` state); BoP zero when off; EOL not derived from model |
| `degradation.py` | Five-term scalar, `basis()` linear in coefficients | Yes | C3 (extrapolated life, EOL literal); M4 (double counting, r_idle<r_base) |
| `env.py` | Gym env, action = fraction of available power | Yes | C3 (obs[4]≈0, persistent degradation off); C4; M7 |
| `baselines.py` | Naive and slew-limited controllers | Naive matches the paper's description of the rule; whether it matches the cited source is unverified (C5-3) | Slew limit is on power fraction, none on the RL agent |
| `profiles.py` | ERA5 → PV/wind → 1-min OU upsample, splits | Matches | M1; random day split; single realisation |
| `train.py` | SB3 training, device split, callback | Hyperparameters not in the paper | C1; M2 (PPO settings); M6 (literals) |
| `evaluate.py` | Rollout, frozen schema | Yes | std is across days not seeds; no CIs; M3 |
| `configs/default.yaml` | Constants | Yes | C1 config; `dv_eol` literal; `j_min` comment vs value |
| `scripts/run_sac_sweep.sh`, `validate_experiment_config.py` | Guards | | Encode the wrong UTD identity (C1) |
| `scripts/reward_landscape.py` | Fixed-policy scan | | Uses test days (M3) |
| `paper/main.tex`, `refs.bib` | Manuscript | | C5, abstract/conclusion wording (C3), Table II (M2) |

---

## 5. Proposed physically consistent operating-state model (opt-in, legacy preserved)

Add `stack.operating_model: legacy | v2` (default `legacy`; `v2` runs are reported separately).

- States: `OFF` (no BoP, no current), `STANDBY` (BoP on, no current, OCV), `ACTIVE` (`j ≥ j_min_safe`).
- `j_min_safe` derived from a stated H2-in-O2 limit and the chosen permeability, **not** from a
  power threshold; keep the legacy value reachable for comparison.
- Transitions: start-up delay and energy, minimum dwell times, and a state-dependent idle rate
  rather than one `r_idle`.
- Action semantics stay "fraction of available power"; below `p_min_active` the environment maps to
  STANDBY or OFF explicitly, and that mapping is logged in `info`.
- Needs from you or the literature before it is coded: permeability and membrane data for a
  real 30-bar stack, a minimum-turndown figure, start-up time/energy, standby power, and
  whether the H2-in-O2 limit should be 2% or 4%.

---

## 6. Corrected implementation plan (ordered; nothing is applied yet)

1. `train.py`, `configs/default.yaml`, `run_sac_sweep.sh`, `validate_experiment_config.py`:
   define UTD explicitly (`gradient_steps = utd · train_freq · n_envs`), log it into the model
   sidecar, and make the guard check UTD instead of the old identity. Add T1.
2. Run a small SAC sanity pilot (not the full matrix): w2 = 0.1 must recover at least the
   constant-action-+1 reward (≥ ~205 kg/day, ≤ ~83 µV/day). If it cannot, stop and debug before
   anything else.
3. `degradation.py`/`config`: derive `dv_eol` from the model's rated voltage (or state the
   reference voltage explicitly); expose coefficients to be perturbed for Experiment A; add an
   option to remove ramp charging on start/stop transitions for the double-counting ablation.
4. `env.py`/`stack.py`: `operating_model` switch (section 5); add `info` fields for state and
   H2-in-O2; add a persistent-degradation evaluation mode for learned policies.
5. `profiles.py`: parameterise OU settings, generate multiple independent realisations, add a
   validation split and a blocked (contiguous-week) split option; report ramp-rate statistics.
6. `train.py`: move all hyperparameters to config; add reward normalisation and entropy options
   for PPO; log commit hash, library versions, hardware.
7. `evaluate.py`: report across-seed and across-day statistics separately; add paired per-day
   bootstrap CIs; write results with config hash and commit.
8. Paper: fix C5 items; qualify lifetime wording; add the threshold-removed controller to the
   frontier figure.

---

## 7. Pre-experiment validation checklist (tests to add before any final training)

| ID | Test | Catches |
|---|---|---|
| T1 | Short SAC run: `_n_updates / (num_timesteps − learning_starts)` equals the intended UTD | C1 |
| T2 | At w2 = 0.1, SAC reward ≥ constant-action-+1 reward on train days | C2 |
| T3 | EOL voltage equals 10% of the model's own rated voltage | C3 |
| T4 | Persistent-degradation rollout runs for a learned policy; feedback magnitude matches 8.64 mV over 90 d at 4 µV/h | C3 |
| T5 | H2-in-O2 at every ACTIVE j is below the configured limit | C4 |
| T6 | Energy balance: electrical in = stack + BoP; H2 energy out ≤ in; LHV efficiency ≤ 1 | C4 |
| T7 | Histogram of j and time-in-state for each controller on test days | C4 |
| T8 | Start/stop charged once per mechanism when the double-counting option is on | M4 |
| T9 | Max resource power ≤ P_rated or the rating mismatch is asserted | M7 |
| T10 | Identical seeds give identical results; distinct seeds give distinct results | M6 |
| T11 | No result file lacks config hash and commit | M6 |
| T12 | Test split is never read in any script that selects a hyperparameter or a w2 range | M3 |

---

## 8. Information I need (details in the chat summary)

1. Is the H200 server still reachable, with how much quota, and can the checkpoints and PPO logs
   be recovered from it?
   *Update 2026-11-05:* a different machine was offered: the shared A5000 workstation (address omitted) (Ubuntu 20.04,
   i9-9900KF 8C/16T, 94 GB RAM, 1× RTX A5000 24 GB, ~1.2 TB free on `/home`, idle at time of
   check, no GPU broker, internet reachable). It has no `pemwe` project, no checkpoints, and no
   torch in the base Python. It is a lived-in shared account with many other projects, so
   nothing was installed or changed there. Far less CPU than the 128-core H200 box, which
   matters because the environment is CPU-bound; throughput must be re-measured.
2. Full texts of the two anchor references (`hydrogen7020076`, `hydrogen7020068`) and of
   `intermittency2025` and `degmodels2025`, plus complete author lists and DOIs.
3. INDIACom 2027 submission deadline and page limit; whether Track 2 requires measured
   parallel-scaling results.
4. Permission to build a fresh Python environment (the current venv is unusable) so the existing
   52 tests and T1–T12 can run.
5. Whether a physics change to crossover/min-load is acceptable, or must be handled as a
   labelled limitation with the legacy model kept as the headline.

---

## 9. Deferred until you approve

The experiment matrix (A–H), compute and storage estimates, the execution sequence, the LCOH
analysis and the manuscript rewrite are not produced here. They depend on C1 (a true UTD
changes the cost of every SAC run by roughly an order of magnitude or more), on server access,
and on the deadline.

**Measured 2026-11-05 on the A5000 workstation (the shared A5000 workstation (address omitted)), SB3 2.9.0, repo SAC
settings, batch 512, 256×256 networks, true UTD = 1 [RAN]:**

| n_envs | train_freq | gradient_steps | updates/s | 2M env steps |
|---|---|---|---|---|
| 8 | 8 | 64 | 166 | ~201 min |
| 32 | 32 | 1024 | 169 | ~197 min |

The rate does not depend on `n_envs`: at the correct update ratio SAC is limited by the learner
(launch/Python overhead), not by environment stepping. Two runs sharing the GPU each slowed to
roughly 90–115 env-steps/s. So the previously planned 130-run, 2M-step SAC matrix would cost on
the order of 400 GPU-hours on this machine and is not feasible as designed; the budget has to be
chosen in updates (not env steps) and the number of runs cut. The older "6,520 steps/s" figures
were measured at 1/32 of the intended update work and must not be used for planning.
