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

### 4. Fix the failing gate check — this is your hardest task

Run it now:

```bash
./.venv/Scripts/python.exe scripts/smoke_test.py
```

`[G1.2]` currently **FAILS**: the jittery policy degrades only **2.06×** faster than the
smooth one, against a target of **≥ 3×**. I left it failing deliberately — it is your first
real task, not an oversight.

**The subtlety that will cost you an afternoon if you miss it:** you have two calibration
targets that pull against each other.

- The **ratio** target (≥3×) needs the cycling and ramp terms to be *large relative to* the
  base and idle terms.
- The **absolute** target (baseline at 4.0 ± 0.5 µV/h, see below) constrains the *total*.

The baseline controller is relatively smooth, so the way through is usually to **raise
`k_ramp_uv_per_h` and `dv_cycle_uv` while lowering `r_base_uv_per_h`**, holding the total on
the baseline profile fixed. Do not just scale everything up — that fixes the ratio and
breaks the absolute rate. Solve both together.

Keep the relative weights physically defensible: ref [4] gives ≲50 µV/h as the worst-case
rate under cycling, so that is your upper bound for an aggressive policy.

### 5. Calibrate against the literature — this is a methodology contribution, write it down

Do not hand-pick the coefficients. Tune them so that **the rule-based baseline, driven by
Person C's real Kutch profiles, reproduces the ~5-year PEM stack lifetime reported in
ref [7]**, with end-of-life defined as a 10 % cell-voltage rise (177 mV).

Target: **baseline mean degradation rate 4.0 ± 0.5 µV/h.** The smoke test already prints
this number and the implied lifetime for both baselines.

Write the procedure up as a paragraph as you go. It pre-empts the obvious reviewer question
and turns an arbitrary-looking parameterisation into a defensible one.

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
