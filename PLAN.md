# 5-Day / 3-Person Sprint Plan
### RL setpoint control of a PEM electrolyzer under variable renewable input

---

## The organising principle

Five days is not enough time for anyone to wait for anyone. The naive split — *A builds
the environment, then B trains on it, then C writes it up* — is a serial chain, and a
serial chain across three people is slower than one person, not faster. It also puts the
paper on Day 5, which is how 5-day papers become 8-day papers.

So the split is **interface-first**. Three seams are frozen at kickoff (`CONTRACTS.md`),
and on the far side of each seam sits a *stub that already works*:

- `src/pemwe/env.py` runs end-to-end today with placeholder physics → **B trains on Day 1 hour 1.**
- `scripts/fake_results.py` emits results in the final schema → **C's figures are finished and debugged before any real number exists.**
- The 10 references and the framing are already settled → **C writes Related Work on Day 1.**

Nobody's Day 1 depends on anybody's Day 1. When A's real physics and B's real numbers
land, downstream code does not change — only the values in it do.

**Cost of this approach:** the stub physics is throwaway and Day 1's trained agent is
meaningless. Both are fine. You are buying two days of parallelism for half a day of
scaffolding, and the scaffolding is already written.

---

## Roles

Each person **owns** their files — one author per file, so merge conflicts are impossible
by construction.

### Person A — Physics & Environment
> *Owns: `stack.py`, `degradation.py`, `env.py`, `tests/`, `configs/default.yaml`*

The scientific credibility of the paper rests here. If the electrolyzer model is wrong,
every number downstream is decoration. A's deliverable is not "an environment that runs"
— it is **an environment that passes the Day-2 validation gate**, with the evidence plotted.

A is also the person who must be able to answer *"where did these degradation constants
come from?"*, so A writes the Methodology subsections on the plant and degradation models.
A knows them best; C should not be transcribing physics secondhand.

### Person B — Agents & Experiments
> *Owns: `baselines.py`, `train.py`, `evaluate.py`, `scripts/run_*.sh`, all of `results/`*

B owns the experimental apparatus and the compute schedule. The scarce resource in this
sprint is wall-clock training time, so B's real job is **keeping the machine busy overnight
on Days 2, 3 and 4** — runs queued before bed, results by breakfast. B also owns the two
baselines and must build them *honestly*: a strawman baseline is the fastest way to lose a
reviewer, so the ramp-limited baseline should be genuinely good.

B writes the RL-method and experimental-setup sections.

### Person C — Data, Analysis & Paper
> *Owns: `profiles.py`, `plots.py`, all of `paper/`, the bibliography*

C is the one who is never blocked — the renewable data pipeline, the entire plotting
library, and roughly 60% of the paper's text depend on no experimental result. C should
finish the figures **before** the results arrive, then spend Days 4–5 as the paper's
editor-in-chief, pulling A's and B's sections into one voice.

C is also the schedule's early-warning system: C is the only person holding the whole
picture, and should be the one who calls a descope when a gate slips.

---

## Day-by-day

### Day 0 — done (this scaffold)
Decisions locked (`DECISIONS.md`), contracts frozen (`CONTRACTS.md`), runnable skeleton,
Python 3.12 venv with SB3. Nothing on Day 1 is spent on setup.

---

### Day 1 — three independent starts

**Kickoff, 60 min, all three.** Read `DECISIONS.md` aloud and argue now, not on Day 3.
Walk `CONTRACTS.md` and freeze it. Confirm the descope ladder. Set the standup time.

| | Person A | Person B | Person C |
|---|---|---|---|
| **AM** | Replace the stub polarization curve with the real Amphlett-form model: reversible + activation (asinh/Butler–Volmer) + ohmic + concentration. Faradaic efficiency with H₂ crossover. BoP parasitic load. | Both baselines from ref #8: naive load-following, and ramp-limited (this is the honest, strong one). Verify both against the stub env. | Register for the Renewables.ninja token. Pull the full 2019 year, PV and wind, at 23.25 °N 69.00 °E. |
| **PM** | Plot polarization + efficiency curves and check them against published PEMWE curves by eye. **Efficiency must peak near j ≈ 0.5–0.8 A/cm² and fall on both sides** — if it is monotonic, the crossover model is wrong. | Get SAC *and* PPO training on the stub env. Wire TensorBoard with the three reward components logged separately. Get `evaluate.py` emitting the results schema. | 1-min upsampling (OU cloud transients on PV, turbulence on wind) preserving hourly means. Pick archetype days from the year's distribution. Fix the train/test day split. |
| **Also** | — | — | Draft **Related Work** — all 10 refs are already in `DECISIONS.md` with why-cite notes. |

**Exit:** A has a physically sane efficiency curve. B has a checkpoint and a results JSON.
C has `kutch_2019_1min.parquet` and a Related Work draft.

---

### Day 2 — the gate day

| | Person A | Person B | Person C |
|---|---|---|---|
| **AM** | All five degradation terms; couple `dv_deg` back into `v_cell` so degradation actually costs yield. | Seed-sweep and ablation runners. Freeze `evaluate.py` — C's plots depend on it and it must stop changing today. | Build all six figures against `fake_results.py`. They must render, label, and export to PDF before real data exists. |
| **PM** | **Calibration** (`DECISIONS.md` §5): scale the coefficients so the rule-based baseline on real Kutch profiles reproduces the ~5-year lifetime reported in ref #7. Then run the validation checklist. | Smoke-test the real env as A lands it. Queue the first honest overnight run. | **Methodology** section from `CONTRACTS.md`. Import A's validation figure as the plant-model figure. |

> ### Gate G1 — end of Day 2. All three sign off before anyone goes home.
> - [ ] Efficiency is non-monotonic in power fraction — peaks mid-range, drops at both ends
> - [ ] Two hand-scripted policies, one smooth and one jittery, on the same profile: **the jittery one degrades measurably faster** (target ≳ 3×). If not, the degradation model cannot teach the agent anything and there is no paper.
> - [ ] Rule-based baseline runs end to end and its H₂ output tracks the renewable input shape
> - [ ] Baseline degradation rate calibrated to 4.0 ± 0.5 µV/h
> - [ ] All six figures render from fake data
>
> **If G1 fails, Day 3 morning is all three people on the environment** — not B tuning
> hyperparameters against a broken env, which is a day spent learning nothing.

---

### Day 3 — calibrate the reward, then commit compute

**A hands off the environment and changes job.** From here A owns the long-horizon rollout,
the degradation calibration against literature, and the Methodology physics sections. A is
also the floating pair of hands for whoever is behind.

**B's morning is the highest-risk two hours of the sprint: the coarse reward-weight sweep.**
Run w₂ ∈ {0.1, 1, 10, 100} short (~200k steps) and read the *component* logs, not total
reward. You are looking for the band where the agent neither parks at idle (w₂ too high)
nor chases every fluctuation to rated power (w₂ too low). Do this **before** the long runs
— it is the difference between one wasted afternoon and one wasted day.

Then queue the real runs overnight: **{SAC, PPO} × 5 seeds** at the chosen weights.

C: Introduction and Experimental Setup; first figures on real preliminary numbers.

> ### Gate G2 — end of Day 3
> - [ ] A trained policy beats the naive baseline on **at least one** of {H₂ yield, degradation} without collapsing the other
> - [ ] Training curves neither flat nor diverging
> - [ ] Overnight runs launched

---

### Day 4 — results freeze

B: final runs plus ablations — reward weights (the headline ablation), solar-heavy vs
wind-heavy, SAC vs PPO. A: the 90-day persistent-degradation rollout for the life-extension
number. C: every final figure and table, then the Results section.

> ### Gate G3 — end of Day 4: **EXPERIMENT FREEZE.**
> Every number in the paper is now fixed. No new runs on Day 5, no matter how tempting.
> This single rule is what makes a 5-day paper possible; breaking it is what makes it an
> 8-day paper. Anything discovered after this point goes into Future Work.

---

### Day 5 — writing only

- **AM** — parallel drafting into one document. A: plant model, degradation model, calibration. B: RL formulation, algorithms, experimental setup. C: Abstract, Introduction, Related Work, Results narrative, Conclusion, Limitations.
- **Midday** — C merges and does a full editing pass for one voice.
- **PM** — each person reads the *entire* paper, not only their own section. Then: IEEE template compliance, every figure referenced in the text, every figure legible at print size, bibliography complete.
- Write **Limitations** honestly: simulation only; degradation model calibrated from literature-reported rates and not validated against a physical stack; no MPC comparison. Reviewers forgive stated limitations and punish hidden ones.

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| **RL does not beat the baseline on yield** | **High — and it probably shouldn't** | The Pareto framing (`DECISIONS.md` §8). The baseline already harvests nearly all available energy; RL buys degradation reduction with a little yield. Framed as a frontier that is the *finding*, not a failure. Decided on Day 0 so it never becomes a Day 4 panic. |
| Reward hacking → degenerate policy | High | Idle term makes idling costly; components logged separately; Day 3 coarse sweep before committing compute |
| Environment physics subtly wrong | Medium | Gate G1 with plotted evidence; unit tests; polarization curve checked against published curves |
| Training too slow on CPU | Medium | 1-D action, small MLP, vectorized envs, cap at 2M steps. If tight, cut to SAC-only |
| Renewables.ninja token / rate limits | Medium | Fall back to the pre-made country datasets at renewables.ninja/downloads, which need no token. Pull on **Day 1 morning**, not Day 3 |
| Merge conflicts | Low | One owner per file, by construction |
| Someone loses a day (illness, other deadline) | Medium | A becomes free after Day 2 by design; C is the least blocked; then the descope ladder |

---

## Descope ladder — agree at kickoff, execute without debate

When a gate slips, cut from the top. Pre-deciding this is what stops a slipping sprint from
becoming an argument about what matters.

1. **PPO** — SAC only. Costs one table row.
2. **The wind-heavy / solar-heavy ablation** — keep the reward-weight sweep; it is the one that speaks to the contribution.
3. **90-day long-horizon rollout** — report degradation rate in µV/h instead of projected years. Weaker headline, same science.
4. **Seeds 5 → 3.** Do *not* go below 3. Single-seed RL results are not publishable.
5. **The idle-corrosion term** — drop to four degradation terms, note it in Future Work.

**Never cut:** the held-out test split, the reward-component logging, or the Limitations
section. Those are what make it a paper rather than a demo.

---

## Standing rules

- **15-minute standup, same time daily.** Each person: yesterday, today, blocked-on. If anyone is blocked on another person, that is the day's top priority.
- **Merge to `main` daily before standup.** A branch that lives three days is a merge conflict with interest.
- **Every run gets a `run_id`; `results/` is append-only.** Never overwrite a run to fix it.
- **Log reward components separately, always.** Total reward hides exactly the failure you are most likely to have.
- **Five seeds, mean ± std on every reported number.** A reviewer will ask.
- **Evaluate on held-out days.** If a number in the paper came from a training day, the paper is wrong.
