# Brief — Person C · Data, Analysis & Paper

**Paste this whole file to your Claude at the start of your session.**
Read `DECISIONS.md`, `CONTRACTS.md`, `PLAN.md` and `paper/REFERENCES.md` first.

---

## Your job in one sentence

You are the person who is **never blocked** — the data pipeline, the entire plotting
library, and roughly 60 % of the paper's text depend on no experimental result — and from
Day 4 you are the paper's editor-in-chief.

## Files you own

```
src/pemwe/profiles.py    renewable data pipeline
src/pemwe/plots.py       all six figures
paper/                   the LaTeX source, and paper/REFERENCES.md
```

## Files you must NOT touch

`stack.py`, `degradation.py`, `env.py`, `configs/default.yaml` (Person A) · `baselines.py`,
`train.py`, `evaluate.py`, `results/` (Person B).

## You also hold a second role

You are the schedule's early-warning system. You are the only person holding the whole
picture, so **you are the one who calls a descope** when a gate slips. The ladder is in
`PLAN.md` — it is pre-agreed precisely so that invoking it is not a negotiation.

---

> ## Status: the data pipeline and the figures are already built
>
> `src/pemwe/profiles.py`, `scripts/build_profiles.py` and `src/pemwe/plots.py` exist and
> run. `data/processed/kutch_2019_1min.parquet` is committed: 365 days, 293 train / 72 test,
> month-stratified. All six figures render to vector PDF from `scripts/fake_results.py`.
>
> **Your two live tasks are (a) the provenance decision below and (b) the paper.**
> `paper/main.tex` and `paper/refs.bib` are scaffolded; Related Work, the resource-data
> subsection and Limitations are written. Everything else is marked `\TODO`.
>
> **The provenance decision, which only you can close:** the data currently comes from
> ERA5 via Open-Meteo, not Renewables.ninja, because the latter needs an account. They are
> not equivalent — Renewables.ninja supplies its own published PV and turbine models, while
> on ERA5 those conversions are *ours* and must be described and cited as such. Register
> for a token and re-run with `--source renewables_ninja` if you can; it removes two of our
> own models from the chain of custody. If not, cite ERA5 and own the conversions in
> Methodology. Either way the citation in `main.tex` §Experimental Setup has to change.

## Day 1 morning — get the data (already done, keep for reference)

Renewables.ninja, site locked in `DECISIONS.md` §6: **Kutch, Gujarat, India — 23.25 °N,
69.00 °E** (chosen because it has strong solar *and* strong wind from one coordinate, so the
solar-vs-wind ablation is not confounded by geography).

Pull the **full 2019 year**, PV and wind, capacity 1 MW each:

```
https://www.renewables.ninja/api/data/pv?lat=23.25&lon=69.00&date_from=2019-01-01&date_to=2019-12-31&dataset=merra2&capacity=1&system_loss=0.1&tracking=0&tilt=25&azim=180&format=csv
https://www.renewables.ninja/api/data/wind?lat=23.25&lon=69.00&date_from=2019-01-01&date_to=2019-12-31&capacity=1&dataset=merra2&height=100&turbine=Vestas+V80+2000&format=csv
```

Auth header: `Authorization: Token <your_token>` (free registration).
**If you hit rate limits, fall back immediately** to the pre-made country datasets at
renewables.ninja/downloads — no token needed. Do not lose a morning to this.

Pull a **full year**, not three days. The archetype days must be justified by where they sit
in the year's distribution, or a reviewer will ask why you picked those three.

Licence is CC BY-NC 4.0 — you **must** cite Pfenninger & Staffell (2016), ref [11].

## Day 1 afternoon — upsample to 1 minute

This is the step that makes the paper's central claim true. `DECISIONS.md` §1–2: at hourly
resolution this is *scheduling*, which ref [9] already occupies. **Sub-hourly resolution is
the contribution.** It is also the only way the intermittency mechanisms of ref [4] —
ON/OFF cycling, ramp-driven thermal fatigue — appear in the data at all.

- **PV**: Ornstein–Uhlenbeck cloud-transient process
- **Wind**: Kaimal-style turbulence process
- **Both**: must preserve the hourly mean exactly. Assert this in a test.

Document the method properly — it goes in the Data subsection and a reviewer will read it.

Then: select the archetype days (sunny/consistent, cloudy/intermittent, windy, calm) from
the year's distribution, and fix the **train/test day split**. Person B evaluates on
held-out days; if a paper number came from a training day, the paper is wrong.

Frozen API (`CONTRACTS.md` §2) — B and A both call these:

```python
load_profiles(path) -> pd.DataFrame       # 1-min DatetimeIndex; cols pv_w, wind_w, hybrid_w
get_day(df, date, source="hybrid") -> np.ndarray   # shape (1440,), watts
SPLITS: dict                              # {"train": [...], "test": [...], "archetypes": {...}}
```

Output: `data/processed/kutch_2019_1min.parquet`. That file is the entire coupling between
you and the rest of the project.

> **Tell Person A the moment that file exists.** The degradation model is currently
> calibrated against synthetic placeholder profiles; A has to re-run
> `scripts/calibrate_degradation.py --solve` on your real data, because real cloud-transient
> statistics change the ramp and cycling integrals. That is a Day-1/2 handoff. If it slips
> to Day 4, every degradation number in the paper was calibrated against fake weather.

## Day 1 also — draft Related Work

All 11 references with per-reference why-cite notes are in `paper/REFERENCES.md`. The
positioning table and the "why RL over MPC?" answer are there too. You do not need any
experimental result to write this section.

---

## Day 2 — build all six figures against fake data

```bash
./.venv/Scripts/python.exe scripts/fake_results.py   # 22 files in the final schema
```

Those are fake numbers in the **real** schema. Build every figure against them so that on
Day 3, when Person B's real results land, your plotting code does not change — only the
numbers in it do. This is what buys the whole team its parallelism, so do not skip ahead
and wait for real data.

| Function | Figure |
|---|---|
| `fig_pareto(results)` | **the headline** — cumulative H₂ vs cumulative degradation; baselines as points, the RL w₂-sweep as a frontier |
| `fig_trajectory(result)` | one day: renewable input, RL setpoint, baseline setpoint overlaid |
| `fig_training_curves(logdirs)` | total reward **and the three components separately**, mean ± std across seeds |
| `fig_ablation(results)` | reward-weight sweep; solar-heavy vs wind-heavy |
| `fig_longhorizon(results)` | 90-day cumulative degradation, RL vs baselines, with the end-of-life line |
| `fig_validation(env)` | polarization + efficiency curves — Person A hands you this |

**Vector PDF, not PNG.** Reviewers zoom. Everything legible at IEEE single-column width
(≈3.5 in) — that means ~8 pt minimum type, and check it by printing, not by looking at your
monitor.

### Why the Pareto figure is the headline — understand this before you plot it

The likeliest outcome of the whole project is that **RL does not beat the baseline on
hydrogen yield.** The load-following baseline already harvests nearly all available energy,
so there is little yield headroom, and the agent buys degradation reduction by giving up a
little yield.

A "%-gain" bar chart turns that into a failed paper. The same data as a frontier — H₂ on one
axis, degradation on the other, baselines as points, the RL family across the w₂ sweep as a
curve — turns it into the actual finding: *the learned policy dominates the rule-based
operating point and traces a frontier the rule-based controller cannot reach.*

This was decided on Day 0 so it never becomes a Day 4 panic. Build every figure to feed it.

Then write **Methodology** from `CONTRACTS.md`.

---

## Days 3–5

- **Day 3** — Introduction and Experimental Setup. First figures against real preliminary numbers.
- **Day 4** — every final figure and table, then the Results section. Numbers freeze at end of day.
- **Day 5** — you are editor-in-chief. A sends you the plant/degradation/calibration
  subsections, B sends the RL formulation and experimental setup. You merge them, edit for
  one voice, and write Abstract, Introduction, Related Work, Results narrative, Conclusion
  and Limitations.

### Write the Limitations section honestly

Say plainly: this is a **simulation study**; the degradation model is **calibrated from
literature-reported rates and not validated against a physical stack**; there is **no MPC
comparison**. Reviewers forgive stated limitations and punish hidden ones. On this timeline,
for a short conference paper, all three are entirely normal — but they must be stated.

### Final checks before submission

- [ ] IEEE two-column template compliance
- [ ] every figure referenced in the text, and legible at print size
- [ ] every number reported as mean ± std over ≥3 seeds
- [ ] every claim of "held-out" actually held out
- [ ] bibliography complete, and every citation doing the job listed in `paper/REFERENCES.md`
