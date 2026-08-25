# Bibliography — 10 core references

Owner: Person C. Each entry carries a **why-cite** note: the specific job that reference
does in this paper. Do not cite any of them generically — if a sentence cites one of these
without doing the job named below, either fix the sentence or drop the citation.

The three papers the contribution is positioned *against* are marked **[POSITION]**. The
Related Work section must state, for each, what it gets right and what it does not cover.

---

**[1] Feng, Q. et al. (2017).** "A review of proton exchange membrane water electrolysis on
degradation mechanisms and mitigation strategies." *Journal of Power Sources*, 366, 33–55.
→ *Why cite:* foundational PEMWE degradation review. Use for the mechanism taxonomy in the
degradation-model subsection, and to justify the high-current-stress and idle/OCV terms.

**[2] Bernhardt, A., Lange, K. et al. (2026).** "Degradation Mechanisms in PEM Water
Electrolysis: Diagnosis and Impact." *Advanced Materials Technologies*, e02419.
→ *Why cite:* the recent comprehensive review — membrane, catalyst layer, PTL, bipolar
plate, gasket, plus diagnostics. Use in the Introduction to establish that degradation is a
live problem, and to support the claim that a lumped voltage-rise state is a reasonable
abstraction of several distinct mechanisms.

**[3] (2025).** "Proton exchange membrane water electrolyzers degradation models review:
implications for power allocation and energy management." *Electric Power Systems Research*
(S0378775325018397).
→ *Why cite:* the load-bearing quantitative source. Links degradation models to power
allocation and EMS — i.e. exactly our problem framing. Supplies membrane thinning
≈0.1 µm/1000 h and ≈20 % catalyst active-area loss over 500 h under stress. Cite for the
parameterisation of the degradation model and for the 1 MW scale being the relevant one.

**[4] (2025).** "Dynamic electrical degradation of PEM electrolyzers under renewable energy
intermittency: mechanisms, diagnostics, and mitigation strategies." *Renewable and
Sustainable Energy Reviews* (S1364032125008433).
→ *Why cite:* the single most important reference for the paper's motivation. Establishes
intermittency-specific degradation — ON/OFF cycling, ohmic resistance rise ≲50 µV/h,
thermal fatigue from ramping. This is what justifies **smart setpoint control rather than
maximum power tracking**, and supplies the upper bound used to set the relative weights of
the degradation terms. Cite in the Introduction, the degradation model, and the calibration.

**[5] Yu, H., Bonville, L., Jankovic, J., Maric, R. (2020).** "Microscopic insights on the
degradation of a PEM water electrolyzer with ultra-low catalyst loading." *Applied Catalysis
B: Environmental*, 260, 118194.
→ *Why cite:* iridium dissolution and redeposition mechanism. Grounds the high-current
stress term and, importantly, the **idle/OCV term** — the anode sitting at open-circuit
potential is not a free state. Cite in the degradation-model subsection.

**[6] (2025).** Surrogate-assisted reinforcement learning framework for zero-gap alkaline
water electrolyser operation. *Applied Energy* (S0306261925013431).
→ **[POSITION]** Closest prior art: physics-informed NN surrogate trained on COMSOL, RL
optimises temperature/flow/current density, ~400× speedup. *Differentiate on two axes:*
it is **AWE, not PEM** (different degradation physics and different dynamic response), and
it optimises efficiency **without an explicit degradation term in the reward**.

**[7] (2026).** "Automated Electrolyzer Control System for the Production, Accumulation, and
Storage of Hydrogen for Refueling Vehicles." *Hydrogen* (MDPI), doi 10.3390/hydrogen7020076.
→ **[POSITION]** The non-RL degradation-aware control comparison point: MPC/predictive
control extending PEM life from ~5 to ~7.5 years via power smoothing. *Two jobs:* (a) the
"why RL over MPC?" discussion, and (b) **the source of the ~5-year lifetime that the
degradation model is calibrated against** (see `DECISIONS.md` §5). Both uses must be cited.

**[8] (2026).** "Dynamic Control of a PV/T Electrolysis System for Hydrogen and Hot-Water
Production." *Hydrogen* (MDPI), doi 10.3390/hydrogen7020068.
→ *Why cite:* the source of the **baseline control law**, used essentially verbatim:
`P_set = min(P_renew, P_rated)` if `P_renew ≥ P_idle_threshold`, else `P_idle`. Cite wherever
the baseline is introduced. This is what makes the baseline a *literature* baseline rather
than a strawman we invented.

**[9] (2026).** "Deep Reinforcement Learning-Based Scheduling for an Electric–Hydrogen
Integrated Station Using a Data-Driven Electrolyzer Model." *Applied Sciences* (MDPI),
doi 10.3390/16073605.
→ **[POSITION]** The nearest RL + PEM work: DQN with Lagrangian relaxation for scheduling
under a CMDP with demand and carbon constraints. *Differentiate on granularity:* scheduling
(hour-ahead/day-ahead), **not real-time setpoint control**, and discrete actions.
**This reference is why the 1-minute control timestep is non-negotiable** — see
`DECISIONS.md` §1–2.

**[10] Cao, D., Hu, W., Zhao, J., Zhang, G., Zhang, B., Liu, Z., Chen, Z., Blaabjerg, F.
(2020).** "Reinforcement Learning and Its Applications in Modern Power and Energy Systems:
A Review." *Journal of Modern Power Systems and Clean Energy*, 8(6), 1029–1042.
→ *Why cite:* the RL-for-energy-systems survey. Use once, in the opening paragraph of
Related Work, to establish RL fundamentals and the PPO/SAC context. Do not lean on it
elsewhere — it is a framing citation, not evidence.

---

**[11] Pfenninger, S. & Staffell, I. (2016).** "Long-term patterns of European PV output
using 30 years of validated hourly reanalysis and satellite data." *Energy*, 114, 1251–1265.
→ *Why cite:* the methodology paper behind Renewables.ninja. **Required** by the dataset's
CC BY-NC 4.0 terms once we use that data. Cite in the Data subsection of Methodology.
(Staffell & Pfenninger 2016, the companion wind paper, if wind data is used.)

---

## The positioning paragraph

Related Work must land this explicitly. The contribution is the **intersection** of three
axes that existing work only hits two at a time:

| Axis | Prior work | This work |
|---|---|---|
| Real-time setpoint control, not offline scheduling | [9] is scheduling granularity | ✅ 1-min step-by-step control |
| PEM specifically, not AWE | [6] is zero-gap alkaline | ✅ PEM |
| Explicit degradation cost in the RL reward | [7] is degradation-aware but MPC; [6] is RL but yield/efficiency only | ✅ RL with an explicit degradation penalty |

## The "why RL over MPC?" answer — a reviewer will ask

Have this ready in the Discussion:

- RL needs no explicit dynamics or degradation model **at deployment time**.
- It can be trained offline on historical renewable profiles.
- It does not re-solve an optimisation problem at every timestep — which matters at a
  1-minute control interval under fast-changing input.

And frame it **as complementary, not superior**. Acknowledge MPC's real advantages —
interpretability and hard constraint satisfaction — as a limitation and future-work item.
A paper that claims RL simply beats MPC, without running MPC, will be marked down.
