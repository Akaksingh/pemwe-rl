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

## Result

The coefficients in `configs/default.yaml` are the ones solved by
`scripts/calibrate_degradation.py --solve` and recorded in `DECISIONS.md` §5:

| Config key | Value | My literature interval | Position |
|---|---|---|---|
| `r_base_uv_per_h` | 2.25 | [1, 10] | 35 % |
| `k_j_uv_per_h` | 50.0 | [5, 60] | 93 % |
| `k_ramp_uv_per_h` | 16.0 | [0.5, 40] | 79 % |
| `dv_cycle_uv` | 4.0 | [0.05, 5] | 96 % |
| `r_idle_uv_per_h` | 1.0 | [0.5, 10] | 23 % |

All five are interior to the published intervals, so the parameterisation is defensible
term by term. That solver adds a constraint worth keeping: the separation must be carried
by **both** intermittency mechanisms [4] names — ramping *and* on/off cycling — with
neither supplying less than 20 % of the policy-dependent total. A solution where one term
carries nearly all the separation is a numerical fit, not a physical model.

On the corrected plant model these give, over the 8-day gate basis:

| policy | rate | projected life |
|---|---|---|
| naive baseline [8] | **4.49 µV/h** | 4.50 yr |
| ramp-limited baseline | 3.42 µV/h | 5.91 yr |
| smooth | 2.43 µV/h | 8.30 yr |
| jittery | 12.97 µV/h | 1.56 yr |

Separation **5.33×** (gate ≥ 3) and the baseline inside 4.0 ± 0.5. **Both gates pass.**

## ⚠️ Open item: the calibration is fitted to pre-correction physics

These coefficients were solved before the plant model was validated and before three env
bugs were fixed (`P_rated` excluding BoP, the stale `obs[0]`, the ON-threshold hard-cap
violation). Those fixes changed the exposure integrals — most visibly, the naive baseline
now genuinely tracks the resource, so it spends more time at higher current density.

The consequence is that `baseline_naive` has drifted from the 4.15 µV/h recorded in
`DECISIONS.md` §5 to **4.49 µV/h — the upper edge of the ±0.5 band**, and individual days
run as high as 5.01 µV/h. Separation likewise moved from the documented 4.50× to 5.33×.

Re-running `--solve` on the corrected physics converges to:

```
r_base 1.5 · k_j 20.0 · k_ramp 11.0 · dv_cycle 3.5 · r_idle 2.0
  ->  baseline 3.78 uV/h,  separation 4.50x,  aggressive 11.1 uV/h
```

which is better centred on the target and restores the 4.50× separation `DECISIONS.md` §5
already quotes. **It has not been applied**, because it would also move the numbers written
into that shared, locked document. That is a standup decision, not a unilateral one.
Either way, `DECISIONS.md` §5's quoted rates need a refresh: the physics under them moved.

## What must be re-run

This calibration used **synthetic** profiles. DECISIONS §5 specifies the *real Kutch
profiles*. When Person C's `kutch_2019_1min.parquet` lands:

```bash
python scripts/calibrate_degradation.py --solve   # after pointing it at the real profiles
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
