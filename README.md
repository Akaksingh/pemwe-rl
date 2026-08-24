# RL Setpoint Control for a PEM Electrolyzer under Variable Renewable Input

5-day, 3-person sprint toward an IEEE two-column conference paper.

| Document | What it is |
|---|---|
| **[PLAN.md](PLAN.md)** | The 3-person / 5-day plan — roles, day-by-day, gates, risks, descope ladder |
| **[DECISIONS.md](DECISIONS.md)** | The 8 open decisions, resolved and locked, with reasoning for the paper |
| **[CONTRACTS.md](CONTRACTS.md)** | The frozen interfaces that let three people work without blocking each other |

---

## Setup (already done — verified working)

```
.venv/            Python 3.12.2 · gymnasium 1.3.0 · stable-baselines3 2.9.0 · torch 2.13.0
```

```bash
./.venv/Scripts/python.exe scripts/smoke_test.py     # skeleton runs end to end
./.venv/Scripts/python.exe scripts/fake_results.py   # results in the final schema, fake numbers
```

The smoke test currently prints **one FAIL** on gate check G1.2. That is deliberate and
correct — it is Person A's concrete first target (see below).

---

## The three lanes

Every file has exactly one owner, so merge conflicts are impossible by construction.

### Person A — Physics & Environment
`stack.py` · `degradation.py` · `env.py` · `tests/` · `configs/default.yaml`

The scientific credibility of the paper lives here. A's deliverable is not "an environment
that runs" — the skeleton already runs — it is **an environment that passes Gate G1** with
plotted evidence.

- [ ] **Day 1 AM** — validate every polarization-curve constant against literature. The stub gives V_cell = 1.817 V at rated and peak η_LHV = 71.9% at j = 0.78 A/cm²; confirm or correct.
- [ ] **Day 1 PM** — plot polarization + efficiency curves against published PEMWE curves. This becomes the Methodology plant-model figure.
- [ ] **Day 2 AM** — all five degradation terms; couple `dv_deg` back into `v_cell`.
- [ ] **Day 2 PM** — **fix G1.2**: the jittery policy currently degrades only 2.1× faster than the smooth one; the target is ≥ 3×. Then calibrate so the rule-based baseline lands at 4.0 ± 0.5 µV/h (~5-year life, ref #7) — see [DECISIONS.md §5](DECISIONS.md).
- [ ] **Day 3+** — hand off the env; take the 90-day rollout and write the Methodology physics sections.

### Person B — Agents & Experiments
`baselines.py` · `train.py` · `evaluate.py` · `scripts/run_*.sh` · all of `results/`

B owns the experimental apparatus and the compute schedule. The scarce resource is
wall-clock training time, so the real job is **keeping the machine busy overnight on Days
2, 3 and 4**.

- [ ] **Day 1 AM** — finish both baselines (stubs in `baselines.py`). Make the ramp-limited one **genuinely good** — a strawman baseline loses reviewers.
- [ ] **Day 1 PM** — SAC *and* PPO training against the stub env. TensorBoard with the three reward components logged **separately**.
- [ ] **Day 2** — seed-sweep + ablation runners. Freeze `evaluate.py`; C's plots depend on it.
- [ ] **Day 3 AM** — ⚠️ the highest-risk two hours of the sprint: coarse w₂ ∈ {0.1, 1, 10, 100} sweep at ~200k steps, read the *component* logs. Find the band where the agent neither parks at idle nor chases every fluctuation. **Do this before committing overnight compute.**
- [ ] **Day 3 PM** — queue {SAC, PPO} × 5 seeds overnight.
- [ ] **Day 4** — final runs + ablations. Freeze at end of day.

### Person C — Data, Analysis & Paper
`profiles.py` · `plots.py` · all of `paper/` · the bibliography

C is never blocked — the data pipeline, the whole plotting library, and ~60% of the paper's
text depend on no experimental result. C is also the schedule's early-warning system and
the person who calls a descope when a gate slips.

- [ ] **Day 1 AM** — Renewables.ninja token; pull full-year 2019 PV + wind at 23.25 °N, 69.00 °E (Kutch). Fallback if rate-limited: the pre-made country datasets, no token needed.
- [ ] **Day 1 PM** — 1-min upsampling (OU cloud transients on PV, turbulence on wind) preserving hourly means. Archetype days + train/test split.
- [ ] **Day 1 also** — draft **Related Work**; all 10 refs with why-cite notes are in the brief.
- [ ] **Day 2** — build **all six figures** against `scripts/fake_results.py`. They must render and export to PDF before real data exists.
- [ ] **Day 3–5** — Introduction, Experimental Setup, then editor-in-chief on the whole paper.

---

## The three gates

| Gate | When | Meaning |
|---|---|---|
| **G1** | end of Day 2 | Environment validated — efficiency non-monotonic, jittery degrades ≥3× faster, baseline calibrated, all figures render. **If G1 fails, all three people work on the environment on Day 3 morning.** |
| **G2** | end of Day 3 | A trained policy beats the naive baseline on at least one of {yield, degradation} without collapsing the other. Overnight runs launched. |
| **G3** | end of Day 4 | **EXPERIMENT FREEZE.** Every number is fixed. Day 5 is writing only. |

---

## The one thing to internalise

**RL probably will not beat the baseline on hydrogen yield** — the baseline already
harvests nearly all available energy, and RL buys degradation reduction by giving up a
little yield. A "%-gain" framing turns that into a failed paper. The headline figure is
therefore a **yield-vs-degradation Pareto plot**, decided on Day 0 so it never becomes a
Day 4 panic. Same experiments, and the result holds whichever axis wins.
See [DECISIONS.md §8](DECISIONS.md).
