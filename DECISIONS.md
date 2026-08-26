# Locked Decisions (Section 8) — do not reopen after Day 1 kickoff

Resolving these was the stated prerequisite to coding. Each carries the reasoning so
the paper's Methodology can quote it directly, and so a reviewer question has an answer.

| # | Decision | Locked value |
|---|---|---|
| 1 | Episode structure | Train on **independent 24 h episodes**; evaluate additionally on a **90-day persistent-degradation rollout** |
| 2 | Control timestep | **Δt = 1 min** (1440 steps/episode) |
| 3 | Grid supplement | **Renewable-only.** `P_set(t) ≤ P_renew(t)`, hard cap |
| 4 | Algorithm | **Both SAC and PPO** (SB3). SAC primary (measurement-backed), PPO the stability fallback. PPO→CPU, SAC→H200 |
| 5 | Degradation terms | **5 terms** from day one: base, high-j stress, ramp stress, on/off cycling, idle/OCV |
| 6 | Site | **Kutch, Gujarat, India — 23.25 °N, 69.00 °E** |
| 7 | Stack | **1 MW PEM, 280 cells, 1000 cm²/cell, j_rated = 2.0 A/cm², 60 °C** |
| 8 | Headline figure | **Yield–degradation Pareto plot**, not a single %-gain bar |

---

## 1 & 2. Episode length and timestep — why 1 minute matters

The paper's whole claim to novelty is *real-time* setpoint control rather than
scheduling (ref #9 is DQN scheduling; that space is taken). **If the control timestep is
one hour, a reviewer correctly says "this is scheduling with a different solver."**
Sub-hourly resolution is not a detail, it is the contribution.

Renewables.ninja is hourly, so Person C upsamples to 1 min with a documented stochastic
model — an Ornstein–Uhlenbeck cloud-transient process on PV and a Kaimal-style turbulence
process on wind, both conditioned to preserve the hourly mean. This is standard practice
and it is also the *only* way the intermittency-driven degradation mechanisms of ref #4
(ON/OFF cycling, thermal fatigue from ramping) can appear at all. At hourly resolution
those mechanisms are invisible, which would gut the reward function.

- Train: 24 h episodes = 1440 steps. Days drawn from a **training split** of the year.
- Evaluate: **held-out days never seen in training** — non-negotiable, a reviewer will check.
- Long-horizon: one 90-day rollout with degradation persisting, for the life-extension number.

## 3. Renewable-only, hard cap

If the agent may backfill from the grid, the optimal policy collapses to "sit at the
peak-efficiency current density forever and buy the difference," the renewable
intermittency disappears from the problem, and "maximise yield" stops being a meaningful
objective. Hard cap keeps the question sharp: *given this power, how should you use it?*

State this explicitly in Methodology. Grid-assist is named in Future Work.

## 4. SAC primary, PPO secondary

The action space is 1-D continuous — SAC's natural home, and far more sample-efficient.
PPO is more tolerant of bad hyperparameters. Both are three lines in SB3, so run both:
a result that holds across two algorithm families is a stronger claim than a result from
one, and if SAC destabilises, PPO is already running.

### Device assignment — measured, not assumed (see `BENCHMARK.md`)

This decision was re-examined against measurements on the H200 server. The algorithm choice
stands; the device and hyperparameters are now fixed by data:

| | Device | Settings | Throughput |
|---|---|---|---|
| **PPO** | **CPU**, no `gpurun` | `n_envs=32` | 9,712 steps/s |
| **SAC** | **GPU**, `gpurun -g 1` | `n_envs=32, train_freq=32, gradient_steps=32` | 6,520 steps/s |

Two findings worth carrying into the paper's reproducibility notes:

- **PPO is ~30 % slower on an H200 than on CPU.** The bottleneck is Python env stepping, not
  the network. Do not "optimise" by moving PPO to the GPU.
- **SAC's SB3 defaults are 26× slower than necessary** (one gradient update per env step on
  a single env). Batching updates across 32 parallel envs at an unchanged 1:1
  update-to-step ratio recovers all of it, and there the GPU genuinely wins by 2.7×.

Because the two use different resources, **run them concurrently**. The 130-run matrix
(13 w₂ × 5 seeds × 2 algorithms) is ≈83 min of wall-clock and ≈5.5 of the 90 weekly GPU-hours.

### Why SAC stays primary — the earlier reasoning survived, the earlier numbers did not

A first benchmark appeared to show SAC ~34× slower than PPO, which would have been a real
argument for demoting it. It was an artifact of **SB3's defaults**, not of SAC:
`n_envs=1, train_freq=1, gradient_steps=1` pays Python and kernel-launch overhead on every
single environment step. Batching the same work across 32 envs recovers 26× of it.

So the decision is unchanged, and now measurement-backed:

- Properly batched SAC is **fast enough to be practical** — 5.1 min per 2M-step seed.
- SAC **benefits substantially from the H200** (2.7× over CPU); PPO does not benefit at all.
- PPO remains **extremely fast on CPU** and needs no GPU resource.
- The two therefore have a **sensible, non-competing hardware allocation** — which is the
  reason running both costs barely more wall-clock than running one.

### The `gradient_steps=8` configuration is excluded from the primary comparison

The fastest row in `BENCHMARK.md` is SAC at 13,889 steps/s with `gradient_steps=8`.
**It must not be used for the primary SAC run or the headline comparison.**

It performs **4× fewer gradient updates per environment step** — a different effective
algorithmic budget, not a free optimisation. Reporting it against PPO, or against
`gradient_steps=32`, would compare two different algorithms and attribute the difference to
hardware.

It may be run **only as a separate, clearly labelled experiment**, and only if learning
curves are validated at **equal environment steps** (never equal wall-clock).

The primary SAC configuration is `n_envs=32, train_freq=32, gradient_steps=32`, which holds
the update-to-step ratio at 1:1 — identical learning dynamics to the SB3 default, just
batched.

### Throughput did not change the science

The device split and batching are engineering decisions about *where* work runs. No
methodology was altered to gain throughput: the reward function, episode structure, seed
count and evaluation protocol are all unchanged. The only experimental change the speedup
bought is **more w₂ points on the frontier** (§8), which increases resolution rather than
changing what is measured.

(The w₂ *range* was later narrowed 0.1–100 → 0.1–20, but for an unrelated reason: the top
of the old range trains a shut-down policy. That was a correction to a Day-0 guess, not a
throughput trade. See §8.)

## 5. Degradation model — five terms, and idle is NOT free

Rates are lumped into a single scalar `dV_deg` (cell-voltage rise, µV), which feeds back
into the polarization curve. Physically this is combined ohmic + kinetic degradation, and
it is the quantity manufacturers actually report.

| Term | Form | Justification |
|---|---|---|
| Base | `r_base` | Steady-state PEMWE degradation, 1–10 µV/h (ref #3) |
| High-current stress | `k_j · max(0, j/j_rated − 0.6)²` | Ir dissolution / membrane thinning accelerate superlinearly at high overpotential (refs #1, #5) |
| Ramp stress | `k_r · \|dj/dt\|` | Thermal fatigue independent of absolute power (ref #4) |
| ON/OFF cycling | `ΔV_cycle` per transition | Start/stop is the dominant intermittency mechanism (ref #4) |
| Idle / OCV | `r_idle` when off | Anode sits at high potential at open circuit → Ir dissolution (refs #1, #5) |

**The idle term is load-bearing.** Without it, the reward-maximising degenerate policy is
"stay at P_idle forever" — zero degradation, near-zero yield — and a badly-weighted run
will find it. Including it is both physically correct and a guard against the most likely
reward-hacking failure.

### Calibration procedure (Person A, Day 2) — this is a methodology contribution, not a fudge

Do not hand-pick the coefficients. Tune them so that **the literature rule-based baseline
(ref #8) driven by the real Kutch profiles reproduces the ~5-year PEM stack lifetime
reported in ref #7**, with end-of-life defined as a 10 % rise in cell voltage (177 mV).

Target: baseline average degradation rate **4.0 ± 0.5 µV/h**. Hold the *relative* weights
at the literature-implied ratios (worst-case cycling ≈ 50 µV/h from ref #4 as the upper
bound) and scale the set. Then "how much life does the RL policy add?" is answered on a
scale anchored to published numbers rather than to invented ones. Write this paragraph
into Methodology — it pre-empts the obvious "where did these constants come from?"

### Calibration result — solved, and how

`scripts/calibrate_degradation.py` does this properly rather than by hand. Every term is
**linear in its coefficient**, so the script measures each term's integral once per policy
with all coefficients set to 1.0, then solves the two constraints analytically over a
literature-bounded grid. Hand-tuning does not converge because the targets pull against
each other: separation needs the *policy-dependent* terms (ramp, cycling, high-j) to
dominate the *common-mode* terms (base, idle) that every policy pays equally, while the
absolute target constrains the total.

The diagnostic that made it tractable: **base + idle were 91 % of the smooth policy's
degradation**, which capped the achievable ratio at ~2.1× no matter how the other terms
were scaled. That common-mode share is now 72 %.

**These are the values calibrated on REAL Kutch weather.** The coefficients have been
re-solved twice, and both times for a reason worth recording:

1. **After PR #1**, which corrected the rated-power definition, a stale observation and a
   hard-cap leak. Those fixes moved the exposure integrals and pushed `baseline_naive` to
   4.49 µV/h — inside the ±0.5 band but 0.01 from its ceiling.
2. **After the real profiles landed.** Real weather is far more punishing than the
   synthetic placeholder: under the previous coefficients the baseline ran at
   **6.95 µV/h** against a 4.0 target. Synthetic days are near-identical sine curves;
   real Kutch days span calm and gusty, and the hybrid resource adds turbine cut-outs.

| Coefficient | Day-0 stub | Post-PR#1 (synthetic) | **Now (real data)** |
|---|---|---|---|
| `r_base_uv_per_h` | 1.5 | 1.5 | **1.75** |
| `r_idle_uv_per_h` | 2.0 | 2.0 | **0.75** |
| `k_ramp_uv_per_h` | 4.0 | 11.0 | **31.0** |
| `dv_cycle_uv` | 1.0 | 3.5 | **1.0** |
| `k_j_uv_per_h` | 25.0 | 20.0 | **20.0** |

Resulting behaviour on real held-out weather (mean of 8 train days), both Gate-G1 targets met:

| Policy | Rate | Projected life | |
|---|---|---|---|
| jittery (adversarial) | 12.47 µV/h | 1.62 yr | separation **4.38×** vs smooth ✅ |
| `baseline_naive` (ref #8) | **4.00 µV/h** | **5.05 yr** | calibration target 4.0 ± 0.5 ✅ |
| `baseline_ramplimited` | 2.86 µV/h | **7.07 yr** | |
| smooth reference | 2.85 µV/h | 7.09 yr | |

Two things to carry into the paper:

- `baseline_naive` lands at **exactly 4.00 µV/h**, mid-band rather than at an edge, so a
  later physics change will not flip the gate.
- **5.05 → 7.07 years from power smoothing alone.** Ref #7 reports 5 → 7.5 years for the
  same effect under predictive control. Our two rule-based baselines reproduce a published
  result on real weather *before* RL enters the picture — which is the strongest available
  evidence that the degradation model is not merely self-consistent.

### The solver objective was also wrong, and that is worth knowing

The first two solves ranked candidates by ratio-closeness-to-4.5 *first*, which parks the
absolute rate at the band edge (4.49 against a 4.5 ceiling). The absolute target has a hard
band; the ratio only needs to clear 3. Centring the rate first and maximising separation
second is what produces the 4.00 above. If you re-solve, keep that ordering.

Two constraints the search enforced, so the fit stays physical rather than numerical:

- The aggressive-policy rate must stay under ref #4's **50 µV/h** worst-case ceiling — it
  lands at 10.2, comfortably inside.
- The separation must come from **both** intermittency mechanisms ref #4 names — ramping
  *and* on/off cycling — with neither supplying less than 20 % of it. A solution where
  `dv_cycle` alone did the work would be a curve fit, not a model.

`r_base = 2.25 µV/h` sits inside ref #3's 1–10 µV/h steady-state band, and `r_idle` stays
strictly positive so "idle forever" remains costly.

> ⚠️ **Re-solve whenever the physics or the profiles change.** It has already had to be
> redone once, after PR #1 corrected the plant model. It must be redone again on real data:
> this calibration still uses the synthetic placeholder profiles in `env.synthetic_day`, not
> real weather.** It must be re-run — `python scripts/calibrate_degradation.py --solve` — once
> Person C lands `data/processed/kutch_2019_1min.parquet`. Real profiles have different
> cloud-transient statistics, so the ramp and cycling integrals will shift. Person C tells
> Person A the moment that file exists; it is a Day-2 handoff, not a Day-4 discovery.

## 6. Site — Kutch, Gujarat (23.25 °N, 69.00 °E)

Needs strong solar *and* strong wind from one coordinate so the solar-heavy / wind-heavy
ablation is not confounded by geography. Kutch qualifies (it is the site of the Khavda
hybrid renewable park), has a distinct monsoon/dry seasonal split that supplies genuinely
different profile archetypes, and connects the paper to India's National Green Hydrogen
Mission — a live policy hook that costs one sentence in the Introduction.

Pull a **full year** (2019, a non-anomalous MERRA-2 year) of PV and wind at 1 MW capacity,
then select archetype days from it. Do not pull three days directly — the archetypes must
be justified by where they sit in the year's distribution.

## 7. Stack parameters

1 MW is the scale at which the EMS/power-allocation literature (ref #3) operates, so the
numbers are comparable. 280 cells × 1000 cm² × 2.0 A/cm² at ≈1.77 V/cell ≈ 991 kW.

Peak *efficiency* lands near j ≈ 0.5–0.8 A/cm² (25–40 % of rated) because Faradaic
efficiency collapses at low j (H₂ crossover) while voltage efficiency collapses at high j.
That non-monotonic curve is what makes the control problem non-trivial — maximum H₂ *rate*
and maximum H₂ *per joule* are at different setpoints — and it is validation item 1.

## 8. Headline figure — Pareto, not a bar chart

**The most likely bad outcome of this project is that RL does not beat the baseline on
hydrogen yield.** It probably shouldn't: the baseline already harvests nearly all
available energy, so there is little yield headroom, and the RL agent buys degradation
reduction by giving up some yield.

A "%-gain" framing turns that into a failed paper. A Pareto plot — cumulative H₂ on one
axis, cumulative degradation on the other, both baselines as points, the RL family across
the w₂ sweep as a frontier — turns the same data into the actual finding: *the learned
policy dominates the rule-based controller's operating point, and traces a frontier the
rule-based controller cannot reach.* Same experiments, result robust to which axis wins.

### Sweep density — 13 points, fixed in `configs/default.yaml → sweep.w2`

Each w₂ value is **exactly one point on the frontier**. The sweep was originally budgeted at
4 coarse points because it was thought to be the sprint's most expensive step; the measured
throughput (`BENCHMARK.md`) makes it minutes, so it is now **13 points**:

```
[0.1, 0.156, 0.242, 0.376, 0.585, 0.909, 1.41, 2.2, 3.42, 5.32, 8.27, 12.9, 20.0]
```

Log spacing over **0.1–20**. The range was originally 0.1–100; it was **narrowed on
measured evidence**, and that correction matters more than the density change:

> **Above w₂ ≈ 20 the reward-optimal policy is to shut the plant down.** Running beats
> idling only while `r_yield > w₂·(r_deg_run − r_deg_idle)` — i.e. `15790 > w₂·(960−180)`,
> so `w₂ < 20.2`. `scripts/reward_landscape.py` confirms it empirically: scoring 16 fixed
> policies on held-out days, w₂ ∈ {31.6, 56.2, 100} all select **idle**, 0 kg H₂.
>
> Those three points were not Pareto points. They were the degenerate corner §5 warns
> about, and they would have spent ~0.85 GPU-h training a controller to switch off.

The same scan showed every w₂ from 0.1 to 5.62 selecting the *same* optimum, so the old
spacing also spent 8 of 13 points on a single Pareto point. The new range puts **5 points in
the active band [3, 20]** where the optimum actually moves, against 2 before.

The 0.1–100 range was set on Day 0 by analogy, before any of these quantities had been
measured. It is the clearest case in the project of a Day-0 guess surviving unexamined into
a locked document — worth remembering when reading the rest of this file.

Four points cannot show the *shape* of a trade-off curve — in particular they cannot resolve
the knee, the region where a small yield sacrifice buys a large degradation reduction, which
is the paper's actual claim. Thirteen can.

**Seed count is unchanged at 5** (`train.seeds`). Replication was not traded for resolution.

Build every experiment to feed this figure.
