# Methodology — plant model, degradation model, calibration

**Owner: Person A** (PLAN.md: *"A writes the Methodology subsections on the plant and
degradation models… C should not be transcribing physics secondhand"*).

**For Person C:** this is draft prose for `paper/`, in the paper's voice, for you to edit
into one voice — not a hand-off of raw notes. Numbers are live against the REAL Kutch profiles and the current coefficients;
every one of them regenerates from `scripts/validate_physics.py`,
`scripts/calibrate_degradation.py` and `scripts/longhorizon_rollout.py`.
Figure: `results/figures/fig_validation.pdf` (vector, 7.0 × 2.7 in, two-column).
Citation keys `[n]` are `paper/REFERENCES.md` numbering.

---

## A. Plant model

We model a 1 MW-class PEM electrolyzer of 280 cells, each 1000 cm², rated at
*j* = 2.0 A cm⁻² and 60 °C, operated in differential-pressure mode with the cathode at
30 bar and the anode near ambient. Cell voltage follows the Amphlett form,

> **(1)**  *V*<sub>cell</sub>(*j*) = *E*<sub>rev</sub>(*T*,*p*) + *V*<sub>act</sub>(*j*) + *V*<sub>ohm</sub>(*j*) + *V*<sub>conc</sub>(*j*) + Δ*V*<sub>deg</sub>

where Δ*V*<sub>deg</sub> is the lumped degradation state of Section B. The reversible
voltage carries the standard linear temperature correction and a Nernst pressure term,
giving *E*<sub>rev</sub> = 1.246 V at the operating point — of which +48.8 mV is the
thermodynamic cost of evolving hydrogen against the 30 bar cathode.

Activation losses use the symmetric Butler–Volmer relation inverted as
η = (*RT*/α*F*) asinh(*j*/2*j*₀) at both electrodes. Rather than adopt transfer
coefficients directly, we fix them through the Tafel slope *b* = 2.303 *RT*/α*F* that they
imply at 60 °C, since that is the quantity experimental work reports: α<sub>a</sub> = 1.5
gives *b*<sub>a</sub> = 44 mV dec⁻¹, within the 40–60 mV dec⁻¹ reported for OER on IrO₂
[1,5], and α<sub>c</sub> = 2.0 gives *b*<sub>c</sub> = 33 mV dec⁻¹, consistent with the
~30 mV dec⁻¹ of HER on platinum [1]. Exchange current densities are 10⁻⁷ A cm⁻² (anode,
the fast end of the reported 10⁻⁷–10⁻¹² range) and 10⁻¹ A cm⁻² (cathode).

**Ohmic loss and hydrogen crossover are derived from a single membrane description**, and
this is deliberate. For a 70 µm reinforced PFSA membrane at 0.10 S cm⁻¹ [1,3] with
0.030 Ω cm² of contact, PTL and bipolar-plate resistance [3], the area-specific resistance
is 0.100 Ω cm². The same thickness sets the hydrogen permeation flux, which we express as
an equivalent crossover current density by Fick's law,

> **(2)**  *j*<sub>cross</sub> = 2*F* *P*<sub>H₂</sub> *p*<sub>cathode</sub> / δ = 0.0165 A cm⁻²

with *P*<sub>H₂</sub> = 2 × 10⁻¹¹ mol cm⁻¹ s⁻¹ bar⁻¹ [1], and the Faradaic efficiency
follows as η<sub>F</sub> = 1 − *j*<sub>cross</sub>/*j*. Because crossover is a
current-independent leak, it consumes a larger *fraction* of the product the lower the
current density — so a thinner membrane buys efficiency at high current and loses it at
low current. Tying both terms to one thickness means the two ends of the efficiency curve
cannot be tuned independently, and the trade-off the controller faces is a property of the
membrane rather than of our parameter choices.

Hydrogen production follows Faraday's law derated by η<sub>F</sub>. A constant
balance-of-plant parasitic of 1.5 % of rated (≈ 15 kW: circulation pumps, chiller,
controls, gas drying) is drawn whenever the stack is on, so that the reported efficiency is
a *system* rather than a stack figure. Rated plant power is therefore 1049 kW, and the plant
cannot run at all below 36.8 kW (3.5 % of rated) — the stack at *j*<sub>min</sub> plus the
parasitic floor.

**Fig. 1** shows the resulting polarization curve with its loss budget, and the system LHV
efficiency. Cell voltage rises from 1.507 V at 0.05 A cm⁻² to 1.846 V at rated. Efficiency
is **non-monotonic**: it peaks at **70.5 % at *j* = 0.73 A cm⁻²**, i.e. 36 % of rated
current and 34 % of rated power, falling to 66.3 % at rated as voltage efficiency degrades
and to 31.9 % at minimum load as crossover dominates.

That non-monotonicity is the reason a controller has anything to decide. The setpoint
maximising hydrogen **rate** is rated power; the setpoint maximising hydrogen **per joule**
is roughly one third of it. Under a hard renewable-only cap (Section …), a controller
facing a fluctuating resource must continuously choose between them, and — as Section B
makes concrete — the choice also has a cost that is not paid until years later.

## B. Degradation model

Degradation is represented as a single scalar Δ*V*<sub>deg</sub>, the cumulative
cell-voltage rise in microvolts, which enters (1) directly. Physically this lumps combined
ohmic and kinetic degradation, and it is the quantity manufacturers report; it is also the
abstraction that recent reviews support as a reasonable summary of several distinct
mechanisms [2]. Because it enters the polarization curve, accumulated degradation costs
*future yield* and not merely reward: at a fixed available power a degraded stack settles
at a lower current density and produces less hydrogen.

Five mechanisms contribute per timestep, with *j*<sub>f</sub> = *j*/*j*<sub>rated</sub>:

| Term | Form | Mechanism |
|---|---|---|
| base | *r*<sub>base</sub> Δ*t* | steady-state PEMWE degradation [3] |
| high-current stress | *k*<sub>j</sub> max(0, *j*<sub>f</sub> − 0.6)² Δ*t* | Ir dissolution and membrane thinning accelerate superlinearly at high overpotential [1,5] |
| ramp | *k*<sub>r</sub> \|d*j*/d*t*\| Δ*t* | thermal fatigue, independent of absolute power [4] |
| cycling | Δ*V*<sub>cycle</sub> per ON↔OFF | start/stop, the dominant intermittency mechanism [4] |
| idle | *r*<sub>idle</sub> Δ*t* when off | the anode at open-circuit potential drives Ir dissolution [1,5] |

The idle term is not a detail. Without it, the reward-maximising degenerate policy is to
park at idle forever — no degradation, almost no hydrogen — and a badly-weighted run will
find it. Including it is both physically correct [1,5] and a structural guard against the
most likely reward-hacking failure.

## C. Calibration

Rather than choose the five coefficients, we **solve** for them, so that the degradation
scale is anchored to published numbers rather than to invented ones.

Since the shape parameters are held fixed, the five terms are exactly linear in their five
coefficients: for a policy *p*, Δ*V*<sub>deg</sub>(*p*) = **c** · **E**(*p*), where
**E**(*p*) accumulates the coefficient-free exposure over a rollout. One rollout per policy
then reduces calibration to a small constrained program over **c** ≥ 0. We require that

1. the literature rule-based controller [8] driven over the site profiles reproduces
   **4.0 µV h⁻¹**, i.e. the ≈5-year stack life reported in [7] at a 10 % (177 mV)
   end-of-life criterion;
2. a deliberately jittery policy degrades at least 3× faster than a smooth one, so that
   the model can distinguish control strategies at all;
3. no policy exceeds the ≈50 µV h⁻¹ worst case reported under cycling [4];
4. every coefficient lies inside its published interval;

and among all admissible solutions we additionally require that the separation be carried
by **both** intermittency mechanisms identified in [4] — ramping *and* on/off cycling —
with neither contributing less than 20 % of the policy-dependent total. Without that last
condition a coefficient set can satisfy the numerical targets while modelling only half
the physics.

The solution, computed on the real Kutch profiles, places all five coefficients interior to
their published intervals (*r*<sub>base</sub> = 1.75 µV h⁻¹, *k*<sub>j</sub> = 20.0,
*k*<sub>r</sub> = 31.0, Δ*V*<sub>cycle</sub> = 1.0 µV, *r*<sub>idle</sub> = 0.75 µV h⁻¹).
The rule-based baseline lands at **4.00 µV h⁻¹** — a 5.05-year projected life against the
≈5 years reported in [7] — and the jittery/smooth separation at **4.38×** against a required
3×. Of the jittery policy's policy-dependent degradation, ramping supplies 63 % and on/off
cycling 37 %, so both of the mechanisms [4] identifies are load-bearing.

Two honest caveats belong with this. First, the model is calibrated to literature-reported
*aggregate* rates and is **not validated against a physical stack**; it reproduces a
published lifetime and a published worst case, and apportions them between mechanisms using
published relative magnitudes, but it does not claim the apportionment is individually
correct. Second, all conclusions drawn from it are about the **relative** lifetimes of
different control policies, which is the quantity the apportionment supports.

## D. Long-horizon evaluation

Training uses independent 24 h episodes (DECISIONS §1), over which the accumulated voltage
rise (~10⁻⁴ V against a 1.85 V cell) is far too small to alter the physics — within an
episode it is the *reward* that carries the degradation signal. To evaluate the lifetime
consequence we additionally run a 90-day rollout with degradation persisting across
episodes, over which the feedback into (1) becomes measurable.

On this rollout the two rule-based controllers already trace the trade the paper is about:
the naive load-following law [8] yields 17 144 kg of H₂ at 4.05 µV h⁻¹ (4.99 yr projected
life), while the ramp-limited variant yields 16 955 kg at 3.20 µV h⁻¹ (6.31 yr) — **+1.32
years of life, +26.5 %, for 1.1 % less hydrogen**. This is the frontier a learned policy must
improve on, and it is why the results are reported as a yield–degradation Pareto front
rather than a single percentage gain (DECISIONS §8).

---

## Notes for C (not for the paper)

- Figure 1 = `results/figures/fig_validation.pdf`. Vector, two panels, 8 pt type, legible
  at column width. Ready to `\includegraphics`.
- The Limitations items I own, stated above and worth keeping in the dedicated section:
  (i) calibration to literature rates, not to a physical stack; (ii) load-proportional BoP
  folded into stack losses rather than modelled separately; (iii) isothermal operation —
  no thermal dynamics, so the ramp term is a proxy for thermal fatigue rather than a
  simulation of it.
- Every number above is regenerable; if a coefficient changes after C's real profiles land,
  re-run the three scripts in `notes/A_calibration.md` and the numbers here move together.
