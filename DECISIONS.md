# Locked Decisions (Section 8) — do not reopen after Day 1 kickoff

Resolving these was the stated prerequisite to coding. Each carries the reasoning so
the paper's Methodology can quote it directly, and so a reviewer question has an answer.

| # | Decision | Locked value |
|---|---|---|
| 1 | Episode structure | Train on **independent 24 h episodes**; evaluate additionally on a **90-day persistent-degradation rollout** |
| 2 | Control timestep | **Δt = 1 min** (1440 steps/episode) |
| 3 | Grid supplement | **Renewable-only.** `P_set(t) ≤ P_renew(t)`, hard cap |
| 4 | Algorithm | **Both SAC and PPO** (SB3). SAC primary, PPO is the stability fallback |
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

Because the two use different resources, run them concurrently. The full 40-run matrix is
under an hour of wall-clock and ~1.7 of the 90 weekly GPU-hours.

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

Build every experiment to feed this figure.
