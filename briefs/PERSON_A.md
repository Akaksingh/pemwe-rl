# Brief — Person A · Physics & Environment

**Paste this whole file to your Claude at the start of your session.**
Read `DECISIONS.md`, `CONTRACTS.md` and `PLAN.md` first — they are short and they are binding.

---

## Your job in one sentence

You own the electrolyzer model. If it is wrong, every number in the paper is decoration.
Your deliverable is **not "an environment that runs"** — the skeleton already runs — it is
**an environment that passes Gate G1 with plotted evidence**, by end of Day 2.

## Files you own

```
src/pemwe/stack.py          polarization curve, efficiency, H2 yield
src/pemwe/degradation.py    the five-term degradation model
src/pemwe/env.py            the Gymnasium environment
configs/default.yaml        every physical constant
tests/                      yours to create
```

## Files you must NOT touch

`baselines.py`, `train.py`, `evaluate.py` (Person B) · `profiles.py`, `plots.py`, `paper/`
(Person C). If you need a change in one of those, raise it at standup.

## What is FROZEN — changing any of this breaks two other people mid-flight

From `CONTRACTS.md` §1, agreed at kickoff:

- Observation space: `Box(shape=(8,))`, the 8 fields in the stated order
- Action space: `Box(low=-1, high=1, shape=(1,))`
- **Every key in the `info` dict**, spelled exactly as listed, present at every step

Change the physics behind these freely. Never change the signatures. Person B's training
loop and Person C's plotting code are both already written against them.

---

## Day 1 — the plant model

### 1. Validate every constant in `configs/default.yaml` against literature

The stub in `stack.py` is a **first draft I did not verify**. It currently produces:

| Quantity | Stub value | Sanity range |
|---|---|---|
| V_cell at rated (j = 2.0 A/cm²) | 1.817 V | 1.75–1.85 V |
| Peak η_LHV | 71.9 % at j = 0.78 A/cm² | peak should sit at 25–40 % of rated |
| η_LHV at rated | 66.6 % | ~65–70 % |
| P_rated | 1017 kW | ~1 MW by design |

Go through `stack:` in the config constant by constant — `alpha_anode`, `j0_anode_a_cm2`,
`asr_ohm_cm2`, `j_lim_a_cm2`, `conc_coeff_v`, `j_crossover_a_cm2`, `bop_frac_of_rated` — and
for each one either confirm it against a published PEMWE source or correct it. **Leave the
citation in a comment next to the value.** You will be asked "where did this number come
from?" in review, and on Day 5 you write that subsection.

Pay particular attention to `j_crossover_a_cm2`. It sets how badly efficiency collapses at
low current density, which is what creates the entire control tension: maximum H₂ **rate**
is at maximum power, but maximum H₂ **per joule** is around 30 % of rated. If that term is
wrong, the agent is solving a different problem than the one described in the paper.

### 2. Produce the validation figure

Write `fig_validation()`-equivalent output from `stack.efficiency_curve()`: polarization
curve (V vs j) and efficiency curve (η_LHV vs power fraction), exported as **PDF, vector**.
Compare by eye against published PEMWE polarization curves and say in your notes which
source you matched against. This becomes the Methodology plant-model figure — hand the PDF
to Person C.

---

## Day 2 — degradation, and the gate

### 3. Couple degradation back into the physics

`degradation.py` already has all five terms (base, high-j stress, ramp, cycling, idle).
Confirm `dv_deg_uv` genuinely feeds back into `stack.v_cell` through `env.step` so that
accumulated degradation actually costs future yield. Without that feedback there is no
long-term/short-term tradeoff and the agent has nothing to learn.

### 4. The degradation calibration — already solved, but you own re-running it

Run it now:

```bash
./.venv/Scripts/python.exe scripts/smoke_test.py        # all three G1 checks PASS
./.venv/Scripts/python.exe scripts/calibrate_degradation.py    # the decomposition
```

`[G1.2]` and `[G1.3]` now pass: separation **4.50×** (target ≥3) and `baseline_naive` at
**4.15 µV/h → 4.87 yr**, against the ~5-year figure in ref [7]. `DECISIONS.md` §5 records
the coefficients and the reasoning.

**Your job is not to redo this — it is to (a) satisfy yourself the numbers are physical, and
(b) re-run it on real data.** The calibration used the synthetic placeholder profiles in
`env.synthetic_day`. The moment Person C lands `data/processed/kutch_2019_1min.parquet`,
re-run `scripts/calibrate_degradation.py --solve`: real cloud-transient statistics change
the ramp and cycling integrals, so the coefficients will move. **Ask C for that file on Day
1, do not wait to be told.**

You also write this up — it is a genuine methodology contribution and reviewers will ask
where the constants came from.

**The subtlety, which is why this was solved with a solver and not by hand:** you have two
calibration targets that pull against each other.

- The **ratio** target (≥3×) needs the cycling and ramp terms to be *large relative to* the
  base and idle terms.
- The **absolute** target (baseline at 4.0 ± 0.5 µV/h, see below) constrains the *total*.

The baseline controller is relatively smooth, so the way through is usually to **raise
`k_ramp_uv_per_h` and `dv_cycle_uv` while lowering `r_base_uv_per_h`**, holding the total on
the baseline profile fixed. Do not just scale everything up — that fixes the ratio and
breaks the absolute rate. Solve both together.

Keep the relative weights physically defensible: ref [4] gives ≲50 µV/h as the worst-case
rate under cycling, so that is your upper bound for an aggressive policy.

### 5. Write the calibration up — it is a methodology contribution

The anchor is: **the rule-based baseline (ref [8]), driven by real Kutch profiles, must
reproduce the ~5-year PEM stack lifetime of ref [7]**, end-of-life defined as a 10 %
cell-voltage rise (177 mV). Target 4.0 ± 0.5 µV/h; the smoke test prints the achieved rate
and implied lifetime for both baselines every run.

Written as a paragraph, this turns an arbitrary-looking parameterisation into a defensible
one and pre-empts the obvious reviewer question. `DECISIONS.md` §5 has the material — your
job is to check it, own it, and put it in the paper in your own words.

---

## Your definition of done for Day 2

```bash
./.venv/Scripts/python.exe scripts/smoke_test.py
```

must print **PASS on all three G1 checks**, and you must have:

- [ ] every constant in `configs/default.yaml` either cited or corrected
- [ ] polarization + efficiency PDF handed to Person C
- [ ] the calibration paragraph drafted
- [ ] unit tests in `tests/` covering: `j_from_power` inverts `power_w`; efficiency is
      non-monotonic; degradation is monotonically non-decreasing; a full-power constant
      policy degrades faster than a mid-power one

Then tell the other two at standup that the env is frozen.

## Day 3 onward — you change job

Hand the environment over. You then own:

1. The **90-day persistent-degradation rollout** (`env.persist_degradation: true`) that
   produces the life-extension headline number.
2. The Methodology subsections on the plant model, the degradation model, and the
   calibration — you are the only person who can write these honestly.
3. Floating help for whoever is behind. By design you are the person with slack on Days 3–4.
