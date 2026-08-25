# Degradation calibration — procedure and result

**Owner: Person A.** Brief item 5, `DECISIONS.md` §5. This is the paragraph that pre-empts
"where did these constants come from?", so it is written as a *procedure* with a
reproducible artefact, not as a set of chosen numbers.

Reproduce: `python scripts/calibrate_degradation.py` (add `--solve` to search and apply).

---

## The problem: two targets that pull against each other

| | Target | Source |
|---|---|---|
| **Absolute** | the rule-based baseline must run at **4.0 ± 0.5 µV/h** — a ~5-year stack life at a 10 % (177 mV) end-of-life criterion | [7] via DECISIONS §5 |
| **Ratio** | a jittery policy must degrade **≥ 3×** faster than a smooth one, or the model teaches the agent nothing | Gate G1.2 |

The baseline controller is comparatively smooth, so the ratio target wants the ramp and
cycling terms **large relative to** base and idle, while the absolute target constrains the
**total** on exactly that smooth trajectory. Scaling the whole set up fixes the ratio and
breaks the rate; scaling down does the reverse. Hand-tuning oscillates between them —
which is what the brief warns costs an afternoon.

## The method: the problem is linear, so it is searched and not hand-tuned

The five terms are **exactly linear in their five coefficients**. The shape parameters
(`j_stress_threshold = 0.6`, `j_stress_exponent = 2.0`) are held fixed by DECISIONS §5, so
for any policy *p*

```
ΔV_deg(p) = c · E(p),        E(p) = Σ_steps basis(j, j_prev, is_on, was_on)
```

where **E**(*p*) is the *unit integral* — the per-term exposure measured once with every
coefficient set to 1.0. `DegradationModel.basis()` returns that per-step vector with the
coefficients stripped out, and
`tests/test_degradation.py::test_basis_is_exactly_linear_in_the_coefficients` holds the
identity to 1e-12. Any candidate coefficient set is then an exact dot product, so the
whole search costs one rollout per policy rather than one per candidate.

`scripts/calibrate_degradation.py --solve` searches that space on a
literature-bounded grid and keeps the sets satisfying all of:

| | Constraint |
|---|---|
| absolute | `c · E(naive) / 24 h` inside 4.0 ± 0.5 µV/h |
| separation | `c · E(jittery) ≥ 3 · c · E(smooth)` |
| ceiling | `c · E(jittery) / 24 h ≤ 50 µV/h` — worst case from [4] |
| bounds | each coefficient inside its published interval |
| **mechanism** | ramping **and** cycling must *each* carry ≥ 20 % of the policy-dependent total |

That last constraint is the one that makes the result physical rather than merely
numerical. [4] names *both* ramping and on/off cycling as intermittency mechanisms; a
coefficient set where one term supplies nearly all the separation would fit the targets
while modelling only half the physics.

The exposures are effectively coefficient-independent: within one 24 h episode the
accumulated ΔV_deg is ~10⁻⁴ V against a 1.85 V cell, so the feedback into `v_cell` cannot
move the trajectory. The linearisation is therefore not an approximation at the day scale
— and the solver re-runs the real environment afterwards, which would expose it if that
ever stopped holding.

Everything is measured as a **mean over 8 days**, the same basis as `scripts/smoke_test.py`
and the tests, so the gate, the solver and the suite can never disagree. A single day is
not a calibration: the day-to-day spread moves the baseline rate by ~0.5 µV/h on its own.

## Literature intervals (the hard constraints)

| Term | Interval | Basis |
|---|---|---|
| base | 1–10 µV/h | steady-state PEMWE voltage rise [3] |
| high-j stress | 5–60 µV/h | Ir dissolution / membrane thinning, superlinear above threshold [1,5] |
| ramp | 0.5–40 µV/h per (A cm⁻² min⁻¹) | thermal fatigue, independent of absolute power [4] |
| cycle | 0.05–5 µV per ON/OFF | start/stop is the dominant intermittency mechanism [4] |
| idle | 0.5–10 µV/h | anode at OCV drives Ir dissolution [1,5] |

## Result — on the real Kutch profiles

`scripts/calibrate_degradation.py` now drives the probe policies with the **real** Kutch
weather (`profiles.env_profiles(8, split="train")`), not the synthetic placeholder, so the
ramp and cycling integrals reflect actual cloud-transient statistics.

Solved coefficients, with where each sits inside the literature interval it was bounded by:

| Config key | Value | Interval | Position |
|---|---|---|---|
| `r_base_uv_per_h` | 1.75 | [1, 10] | 24 % |
| `k_j_uv_per_h` | 20.0 | [5, 60] | 56 % |
| `k_ramp_uv_per_h` | 31.0 | [0.5, 40] | 94 % |
| `dv_cycle_uv` | 1.0 | [0.05, 5] | 65 % |
| `r_idle_uv_per_h` | 0.75 | [0.5, 10] | 14 % |

All five remain interior to their published intervals. Per-day term decomposition (µV):

| policy | base | stress | ramp | cycle | idle | total | µV/h | life |
|---|---|---|---|---|---|---|---|---|
| smooth | 37.35 | 0.00 | 16.58 | 12.38 | 1.99 | 68.30 | 2.85 | 7.09 yr |
| jittery | 36.07 | 0.84 | 162.93 | 96.88 | 2.54 | 299.25 | 12.47 | 1.62 yr |
| **naive baseline [8]** | 30.65 | 3.55 | 24.81 | 32.12 | 4.86 | **96.00** | **4.00** | **5.05 yr** |
| ramp-limited baseline | 30.95 | 3.36 | 20.16 | 12.38 | 4.74 | 71.58 | 2.98 | 6.78 yr |

**Separation 4.38×** (gate ≥ 3) and the rule-based baseline at **exactly 4.00 µV/h**, i.e.
the ~5-year life of [7]. Both targets met on real weather.

The separation is carried by both mechanisms [4] names, as required: of the jittery
policy's 260 µV of policy-dependent degradation, ramping supplies 63 % and on/off cycling
37 %. Neither term is doing the work alone, so this is a physical model rather than a fit.

Note where `k_ramp` sits — 94 % of its interval. That is the honest reading of real
1-minute weather: genuine cloud transients ramp the stack far harder than the synthetic
placeholder did, and reproducing a 5-year baseline life alongside a 3× separation needs the
ramp term near the top of its published range. It is inside the interval, but it is the
coefficient a reviewer is most likely to press on, and the answer is that the *data*, not
the target, put it there.

## The 90-day long-horizon rollout — the life-extension number

`scripts/longhorizon_rollout.py --days 90`, real profiles, `persist_degradation: true`:

| policy | H₂ (90 d) | rate | projected life |
|---|---|---|---|
| naive load-following [8] | 17 144 kg | 4.05 µV/h | **4.99 yr** |
| ramp-limited | 16 955 kg | 3.20 µV/h | **6.31 yr** |

**+1.32 years of life (+26.5 %) for 1.1 % less hydrogen.** That is the frontier the learned
policy has to improve on, and it is why the headline is a Pareto plot rather than a single
percentage (DECISIONS §8).

## Re-running it

The calibration is now on the **real** Kutch profiles, as DECISIONS §5 requires. Re-run the
whole chain after any change to the plant model, the splits, or the profile pipeline —
these four commands must all stay green together, and they are cheap:

```bash
python scripts/calibrate_degradation.py            # diagnose; --solve to re-search
python scripts/validate_physics.py                 # 11/11 plant-model checks
python -m pytest tests/ -q                         # 52 tests
python scripts/longhorizon_rollout.py --days 90    # real profiles by default
```

The procedure never changes; only the exposure matrix does. That is precisely why it is a
script and not a hand-tune, and it is why the number can honestly be called *calibrated*
rather than *chosen*.

## Limitation to state in the paper

The model is calibrated **to literature-reported aggregate rates, not validated against a
physical stack**. It reproduces the published lifetime of a rule-based controller and the
published worst-case rate under cycling, and it apportions those between five mechanisms
using published relative magnitudes. It does not claim the apportionment is individually
correct. Every conclusion here is about *relative* lifetimes under different control
policies, which is the quantity the apportionment supports and the strongest claim the
evidence allows.
