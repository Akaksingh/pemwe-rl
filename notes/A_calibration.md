# Degradation calibration — procedure and result

**Owner: Person A.** Brief item 5, `DECISIONS.md` §5. This is the paragraph that pre-empts
"where did these constants come from?", so it is written as a *procedure* with a
reproducible artefact, not as a set of chosen numbers.

Reproduce: `python scripts/calibrate_degradation.py` (add `--write` to apply).

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

## The method: it is a linear program, so solve it as one

The five terms are **linear in their five coefficients**. The shape parameters
(`j_stress_threshold = 0.6`, `j_stress_exponent = 2.0`) are held fixed by DECISIONS §5, so
for any policy *p*

```
ΔV_deg(p) = c · E(p),        E(p) = Σ_steps basis(j, j_prev, is_on, was_on)
```

exactly — `DegradationModel.basis()` returns that per-step exposure with the coefficients
stripped out, and `tests/test_degradation.py::test_basis_is_exactly_linear_in_the_coefficients`
holds it to 1e-12. One rollout per policy therefore reduces the whole calibration to a
small constrained program over **c ≥ 0**:

| | Constraint |
|---|---|
| equality | `c · E(naive) / 24 h = 4.0 µV/h` |
| inequality | `c · E(jittery) ≥ 3.3 · c · E(smooth)` — margin over the 3.0 gate |
| inequality | `c · E(jittery) / 24 h ≤ 50 µV/h` — worst-case ceiling from [4] |
| bounds | each coefficient inside its published interval |
| objective | minimise squared **log**-distance to the geometric centre of those intervals |

That objective is the part that makes this defensible rather than convenient: among all
coefficient sets satisfying the data, it returns the **least distorted** one, so every
value can be reported together with the interval it came from and its position inside it.
It also makes the answer reproducible — no solver-corner lottery.

The exposures are effectively coefficient-independent: within one 24 h episode the
accumulated ΔV_deg is ~10⁻⁴ V against a 1.85 V cell, so the feedback into `v_cell` cannot
move the trajectory. The linearisation is therefore not an approximation at the day scale.

## Literature intervals (the hard constraints)

| Term | Interval | Basis |
|---|---|---|
| base | 1–10 µV/h | steady-state PEMWE voltage rise [3] |
| high-j stress | 5–60 µV/h | Ir dissolution / membrane thinning, superlinear above threshold [1,5] |
| ramp | 0.5–40 µV/h per (A cm⁻² min⁻¹) | thermal fatigue, independent of absolute power [4] |
| cycle | 0.05–5 µV per ON/OFF | start/stop is the dominant intermittency mechanism [4] |
| idle | 0.5–10 µV/h | anode at OCV drives Ir dissolution [1,5] |

## Result

Exposure matrix (per day, coefficient-free, mean over 5 synthetic days):

| policy | base | stress | ramp | cycle | idle |
|---|---|---|---|---|---|
| naive baseline | 10.52 | 0.458 | 0.920 | 10.40 | 13.48 |
| ramp-limited baseline | 10.64 | 0.361 | 0.685 | 4.40 | 13.36 |
| smooth | 11.30 | 0.000 | 0.319 | 4.00 | 12.70 |
| jittery | 10.96 | 0.088 | 4.737 | 43.60 | 13.04 |

Solved coefficients:

| Config key | Value | Interval | Position in interval |
|---|---|---|---|
| `r_base_uv_per_h` | **2.344** | [1, 10] | 37 % |
| `k_j_uv_per_h` | **21.812** | [5, 60] | 59 % |
| `k_ramp_uv_per_h` | **7.293** | [0.5, 40] | 61 % |
| `dv_cycle_uv` | **2.847** | [0.05, 5] | 88 % |
| `r_idle_uv_per_h` | **1.857** | [0.5, 10] | 44 % |

Every coefficient is interior to its published interval. The one sitting high is the
per-cycle term at 88 %, which is exactly what [4] would predict — start/stop is named
there as the *dominant* intermittency mechanism — so the solver landing there is a mild
independent corroboration rather than a strain.

Resulting rates:

| policy | rate | projected life |
|---|---|---|
| naive baseline [8] | **4.00 µV/h** | **5.05 yr** ← the [7] calibration target |
| ramp-limited baseline | 3.13 µV/h | 6.45 yr |
| smooth | 2.66 µV/h | 7.60 yr |
| jittery | 8.77 µV/h | 2.30 yr |

**jittery / smooth = 3.30×** (gate ≥ 3.0). On the single seed the smoke test uses, the
same coefficients give **3.48×** and a baseline rate of **4.15 µV/h** — both inside the
gate and the ±0.5 band, and the spread between seeds is a useful reminder to quote these
as day-averaged rather than single-day numbers.

Over the 90-day persistent rollout the naive baseline settles at **3.63 µV/h → 5.57 yr**,
against **3.05 µV/h → 6.63 yr** for the ramp-limited one: the honest baseline already buys
**+1.06 yr (+19 %) of life for −3.4 % H₂**. That is the frontier the learned policy has to
beat, and it is why the headline figure is a Pareto plot and not a percentage (DECISIONS §8).

## What must be re-run

This calibration used **synthetic** profiles. DECISIONS §5 specifies the *real Kutch
profiles*. When Person C's `kutch_2019_1min.parquet` lands:

```bash
python scripts/calibrate_degradation.py --profiles data/processed/kutch_2019_1min.parquet --write
python scripts/validate_physics.py
python -m pytest tests/ -q
python scripts/longhorizon_rollout.py --days 90 --profiles data/processed/kutch_2019_1min.parquet
```

The procedure does not change; only the exposure matrix does. That is precisely why it is
a script and not a hand-tune — and it is the reason the number can be honestly described
in the paper as *calibrated* rather than *chosen*.

## Limitation to state in the paper

The model is calibrated **to literature-reported aggregate rates, not validated against a
physical stack**. It reproduces the published lifetime of a rule-based controller and the
published worst-case rate under cycling, and it apportions those between five mechanisms
using published relative magnitudes. It does not claim the apportionment is individually
correct. Every conclusion here is about *relative* lifetimes under different control
policies, which is the quantity the apportionment supports and the strongest claim the
evidence allows.
