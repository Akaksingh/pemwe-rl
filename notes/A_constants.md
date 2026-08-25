# Plant-model constants — validation record

**Owner: Person A.** Brief item 1 ("go through `stack:` constant by constant and either
confirm it against a published PEMWE source or correct it").
Regenerate every number here with `python scripts/validate_physics.py`.

Reference numbers `[n]` are `paper/REFERENCES.md`.

---

## What changed, and why

The Day-0 stub was described as "a first draft I did not verify". Three things were
wrong with it, and one thing was missing.

**1. `alpha_anode` was a cathode's transfer coefficient.** `α` is only meaningful through
the Tafel slope it implies at the operating temperature, `b = 2.303 RT / αF`. The stub's
`α_a = 2.0` implies **33 mV/dec** at 60 °C. OER on IrO₂ is reported at **40–60 mV/dec**
[1,5]; 33 mV/dec is a Pt-HER slope. Corrected to `α_a = 1.5` → **44 mV/dec**.
`α_c = 2.0` (33 mV/dec) is *correct* for HER on Pt and was kept.

**2. `asr_ohm_cm2 = 0.15` had no provenance**, and it is the single constant that most
strongly shapes the high-current half of the efficiency curve. It is now *derived*:

```
ASR = membrane_thickness / membrane_conductivity + r_contact
    = 0.0070 cm / 0.10 S/cm + 0.030 = 0.100 Ω·cm²
```

70 µm reinforced PFSA (between N212's 51 µm and N115's 127 µm) is the membrane class
actually used in MW-scale stacks rated near 2 A/cm²; a 183 µm N117 would put 0.37 V of
ohmic loss into the cell at rated and is not a credible choice at this current density.
Conductivity 0.10 S/cm is fully-hydrated PFSA at 60 °C (0.09–0.13 reported) [1,3].
Contact + PTL + bipolar plate at 0.030 Ω·cm² sits mid-range of the reported 0.02–0.05 [3].

**3. `j_crossover_a_cm2 = 0.03` was asserted, not derived** — and the brief flags it as
the constant to worry about most, because it sets how hard efficiency collapses at low
current density and therefore creates the entire control tension. It is now computed from
Fick's law through the *same* membrane:

```
j_cross = 2F · P_H2 · p_cathode / thickness
        = 2(96485)(2.0e-11 mol cm⁻¹s⁻¹bar⁻¹)(30 bar) / 0.0070 cm
        = 0.0165 A/cm²
```

i.e. **about half** the asserted value, and no longer a free parameter. H₂ permeability
of hydrated PFSA is reported at 1–3 × 10⁻¹¹ mol cm⁻¹ s⁻¹ bar⁻¹ [1].

> **Why this matters beyond tidiness.** Thickness now drives *both* ohmic loss and
> crossover, in opposite directions. The two ends of the efficiency curve can no longer be
> tuned independently to flatter the result — buying high-current efficiency with a thinner
> membrane necessarily costs low-current efficiency. `tests/test_stack.py::
> test_membrane_thickness_drives_both_asr_and_crossover` pins that down.

**4. `pressure_bar: 30.0` was dead config** — declared, never read. The stack is now
modelled in differential-pressure mode (cathode 30 bar, anode ambient), which is the
standard commercial arrangement, and pressure is load-bearing **twice**: a Nernst penalty
of **+48.8 mV** on `E_rev`, and the driving force for the crossover above.

---

## The table

| Constant | Value | Status | Basis |
|---|---|---|---|
| `n_cells` | 280 | locked | DECISIONS §7 |
| `area_cm2` | 1000 | locked | DECISIONS §7 |
| `j_rated_a_cm2` | 2.0 | locked | DECISIONS §7 |
| `temp_k` | 333.15 (60 °C) | locked | DECISIONS §7 |
| `pressure_bar` | 30.0 | **now used** | cathode side; differential operation |
| `pressure_anode_bar` | 1.0 | **new** | anode at ambient |
| `j_min_a_cm2` | 0.05 | confirmed | 2.5 % of rated; PEMWE turndown 5–10 % [3]. Deliberately permissive so low-load operation is rejected on efficiency grounds, not by fiat |
| `e_rev_ref_v` | 1.229 | confirmed | thermodynamic |
| `e_rev_dtemp_v_per_k` | −0.9e-3 | confirmed | −0.85…−0.90 mV/K for liquid-water electrolysis |
| `alpha_anode` | **1.5** ← 2.0 | **corrected** | ⇒ 44.1 mV/dec, inside 40–60 for IrO₂ OER [1,5] |
| `j0_anode_a_cm2` | 1.0e-7 | confirmed | IrO₂ OER spans 1e-7…1e-12 [1,5]; fast end, modern high-loading anode |
| `alpha_cathode` | 2.0 | confirmed | ⇒ 33.1 mV/dec, matches ~30 for Pt HER [1] |
| `j0_cathode_a_cm2` | 1.0e-1 | confirmed | Pt HER is fast, 1e-3…1e-1 [1] |
| `membrane_thickness_cm` | 0.0070 | **new** | 70 µm reinforced PFSA, the MW-scale class at 2 A/cm² |
| `membrane_conductivity_s_cm` | 0.10 | **new** | hydrated PFSA at 60 °C, 0.09–0.13 [1,3] |
| `r_contact_ohm_cm2` | 0.030 | **new** | PTL + plate + interfacial, 0.02–0.05 [3] |
| ⇒ `asr` (derived) | **0.100** ← 0.15 | **corrected** | no longer a free parameter |
| `j_lim_a_cm2` | 4.0 | confirmed | anode transport limits 3–8 A/cm² [3]; low end, so the high-j penalty is not understated |
| `conc_coeff_v` | 0.05 | fitted | adds only 35 mV at rated — shapes the knee, does not dominate |
| `h2_permeability_...` | 2.0e-11 | **new** | hydrated PFSA, 1e-11…3e-11 [1] |
| ⇒ `j_crossover` (derived) | **0.0165** ← 0.03 | **corrected** | Fick's law, tied to thickness and pressure |
| `bop_frac_of_rated` | **0.015** ← 0.02 | **corrected** | always-on parasitic floor only (~15 kW on 1 MW): pumps, chiller, controls, drying. Sets the *left* edge of the efficiency peak |

---

## Validation against the acceptance bands

All ten checks PASS (`scripts/validate_physics.py`).

| Quantity | Value | Band | Source of the band |
|---|---|---|---|
| V_cell at rated | **1.846 V** | 1.75–1.85 | brief |
| peak η_LHV | **70.5 %** | 65–75 % | brief (stub gave 71.9 %) |
| j at peak η | **0.729 A/cm²** | 0.50–0.80 | DECISIONS §7 |
| peak as fraction of rated | **36.5 %** | 25–40 % | DECISIONS §7 |
| η_LHV at rated | **66.3 %** | 65–70 % | brief |
| P_rated | **1049 kW** | ~1 MW | DECISIONS §7 |
| Tafel slope, anode | **44.1 mV/dec** | 40–60 | [1,5] |
| Tafel slope, cathode | **33.1 mV/dec** | 25–40 | [1] |
| ASR | **0.100 Ω·cm²** | 0.05–0.25 | [1,3] |
| j_crossover | **0.0165 A/cm²** | 0.005–0.05 | [1] |

**Polarization curve** (compare by eye against a published PEMWE curve at 60 °C — the
shape was checked against the ~1.65 V @ 0.5 A/cm² / ~1.85 V @ 2 A/cm² envelope that
[1] and [3] report for pressurised PEMWE at this temperature):

| j (A/cm²) | 0.05 | 0.10 | 0.20 | 0.40 | 0.73 | 1.00 | 1.50 | 2.00 |
|---|---|---|---|---|---|---|---|---|
| V_cell (V) | 1.507 | 1.529 | 1.559 | 1.604 | 1.661 | 1.703 | 1.775 | 1.846 |
| η_LHV (%) | 31.9 | 49.8 | 62.4 | 68.9 | **70.5** | 70.1 | 68.4 | 66.3 |
| P / P_rated (%) | 3.5 | 5.5 | 9.8 | 18.7 | 33.9 | 47.0 | 72.5 | 100 |

---

## Two consequences worth stating out loud

**P_rated moved from 1017 kW to 1049 kW**, and not only because the voltage changed. The
stub computed `p_rated` from stack power *before* the BoP attribute existed, so
`power_w(j_rated)` came out at `1.02 × p_rated` and **a commanded power fraction of 1.0
could never actually reach rated current density**. `P_rated` is now the total plant draw
at `j_rated`, parasitics included, so action `+1` maps exactly onto `j_rated`.

**Minimum ON power is 36.8 kW (3.5 % of rated)** — the stack at `j_min` *plus* the
parasitic floor. The stub gated turn-on on stack power alone, so for available power
between 21 kW and 37 kW the plant switched on and drew more than the renewable input
supplied, quietly violating the DECISIONS §3 hard cap. Pinned by
`tests/test_stack.py::test_power_never_exceeds_the_commanded_setpoint`.
