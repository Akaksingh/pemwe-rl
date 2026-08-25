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
bought is **more w₂ points on the same frontier over the same range** (§8), which increases
resolution rather than changing what is measured.

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

**These are the values after PR #1.** The first solve ran against the pre-PR physics; once
that PR corrected the rated-power definition, the stale observation and the hard-cap leak,
the exposure integrals moved and `baseline_naive` drifted to 4.49 µV/h — inside the ±0.5
band but 0.01 from its ceiling, with single days at 5.01. Re-solved on the corrected
physics:

| Coefficient | Day-0 stub | First solve | **Now** |
|---|---|---|---|
| `r_base_uv_per_h` | 1.5 | 2.25 | **1.5** |
| `r_idle_uv_per_h` | 2.0 | 1.0 | **2.0** |
| `k_ramp_uv_per_h` | 4.0 | 16.0 | **11.0** |
| `dv_cycle_uv` | 1.0 | 4.0 | **3.5** |
| `k_j_uv_per_h` | 25.0 | 50.0 | **20.0** |

Resulting behaviour (mean of 8 days), both Gate-G1 targets met:

| Policy | Rate | Projected life | |
|---|---|---|---|
| jittery (adversarial) | 11.08 µV/h | 1.82 yr | separation **4.50×** vs smooth ✅ |
| `baseline_naive` (ref #8) | 3.78 µV/h | **5.35 yr** | calibration target 4.0 ± 0.5 ✅ |
| `baseline_ramplimited` | 2.95 µV/h | 6.86 yr | smoothing extends life 1.28×, as ref #7 reports |
| smooth reference | 2.46 µV/h | 8.21 yr | |

`baseline_naive` at **5.35 yr** now reproduces ref #7's ~5-year PEM stack lifetime more
closely than the first solve did, and sits mid-band rather than at its edge.

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
[0.1, 0.178, 0.316, 0.562, 1.0, 1.78, 3.16, 5.62, 10.0, 17.8, 31.6, 56.2, 100.0]
```

Log spacing, quarter-decade, over the **unchanged** range 0.1–100. The original coarse
points {0.1, 1, 10, 100} remain in the list as a subset, so this is a strict refinement, not
a different experiment. Linear spacing would crowd nearly every point into the high-penalty
regime and leave the low-w₂ end unresolved.

Four points cannot show the *shape* of a trade-off curve — in particular they cannot resolve
the knee, the region where a small yield sacrifice buys a large degradation reduction, which
is the paper's actual claim. Thirteen can.

**Seed count is unchanged at 5** (`train.seeds`). Replication was not traded for resolution.

Build every experiment to feed this figure.
