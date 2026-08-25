# Brief — Person B · Agents & Experiments

**Paste this whole file to your Claude at the start of your session.**
Read `DECISIONS.md`, `CONTRACTS.md` and `PLAN.md` first — they are short and they are binding.

---

## Your job in one sentence

You own the experimental apparatus and the compute schedule. The scarce resource in this
sprint is **wall-clock training time**, so your real job is keeping the machine busy
overnight on Days 2, 3 and 4 — runs queued before bed, results by breakfast.

## Files you own

```
src/pemwe/baselines.py     the two baseline controllers
src/pemwe/train.py         SB3 training — you create this
src/pemwe/evaluate.py      rollout + results emission — you create this
scripts/run_*.sh           run orchestration
results/                   all experimental output
```

## Files you must NOT touch

`stack.py`, `degradation.py`, `env.py`, `configs/default.yaml` (Person A) · `profiles.py`,
`plots.py`, `paper/` (Person C).

**You are not blocked by Person A.** `env.py` already runs end to end with placeholder
physics. Train against it from hour one. When A lands the real physics on Day 2, your code
does not change — only your numbers do. Day 1's trained agent is meant to be meaningless.

## What is FROZEN

The results schema in `CONTRACTS.md` §3. Person C's entire plotting library is already
written against it. If you need to change a field, tell C at standup **before** you change it.

---

## Day 1

### 1. Finish the two baselines

`baselines.py` has both stubbed and working. Your job is to make the second one **genuinely
good**:

- `NaiveLoadFollowing` — the rule from ref [8], verbatim. Leave it alone; it is the
  literature baseline and its job is to be the literature baseline.
- `RampLimitedBaseline` — load-following plus a slew limit. **Tune `ramp_limit_frac_per_min`
  properly.** This is the honest, strong baseline.

Take this seriously. "RL beats a deliberately bad controller" is not a result, and a
reviewer who spots a strawman baseline stops reading. If the RL agent ends up only *tying*
the ramp-limited baseline on yield while beating it on degradation, that is still the Pareto
story the paper is built around, and the paper is fine.

### 2. Training pipeline — SAC and PPO

Stable-Baselines3 2.9.0 is already installed in `.venv`. Both algorithms, per
`DECISIONS.md` §4: SAC primary (1-D continuous action, sample-efficient), PPO as the
stability fallback.

**The one non-obvious requirement:** log the three reward components **separately** to
TensorBoard, not just total reward. They come through the `info` dict as `r_yield`, `r_deg`
and `r_ramp`, unweighted. You will need a custom `BaseCallback` that pulls them from
`self.locals["infos"]` and calls `self.logger.record()`.

This is not optional polish. Total reward alone cannot distinguish "learning well" from
"found a degenerate policy", and reward hacking is the single failure mode most likely to
eat a day of this sprint. You need to *see* the yield term and the degradation term move
independently.

### 3. `evaluate.py`

Rollout harness that treats scripted and learned policies identically — the baselines
already expose `.act(obs, info)` and `.reset()` to match an SB3 policy, so one code path
handles both.

It must emit exactly the schema in `CONTRACTS.md` §3. Check yourself against the reference
output:

```bash
./.venv/Scripts/python.exe scripts/fake_results.py   # 22 files in the target schema
```

Write a small validator that asserts your real output has the same keys as those files.
Include `trajectory` for **one representative episode per run only** — full trajectories for
every episode will blow up the directory.

---

## Day 2

Seed-sweep runner (seeds `[0,1,2,3,4]`) and ablation runner driven by
`--override reward.w2=...`, never by editing `configs/default.yaml`. On Day 5 you must be
able to reconstruct which config produced which number.

**Freeze `evaluate.py` by end of Day 2.** C's plots depend on it.

Smoke-test against A's real environment as it lands, then queue the first honest overnight run.

---

## Day 3 — your highest-risk two hours

**Do this before committing any overnight compute.**

Coarse reward-weight sweep: `w2 ∈ {0.1, 1, 10, 100}`, short runs (~200k steps each). Read
the **component** logs, not total reward. You are looking for the band where the agent:

- does **not** park at `P_idle` forever (w₂ too high — the classic degenerate policy), and
- does **not** chase every fluctuation straight to rated power (w₂ too low)

Two hours here versus a wasted day of long runs at a useless weighting. Then queue the real
runs overnight: **{SAC, PPO} × 5 seeds** at the chosen weights.

### Gate G2 — end of Day 3

- [ ] a trained policy beats the naive baseline on **at least one** of {H₂ yield, degradation}
      without collapsing the other
- [ ] training curves neither flat nor diverging
- [ ] overnight runs launched

---

## Day 4 — then stop

Final runs plus ablations: the reward-weight sweep (the paper's headline ablation),
solar-heavy vs wind-heavy profiles, SAC vs PPO.

> ### Gate G3 — end of Day 4: EXPERIMENT FREEZE
> Every number in the paper is fixed. **No new runs on Day 5, no matter how tempting.**
> This rule is what makes a 5-day paper possible. Anything you discover after this goes
> into Future Work.

---

## Non-negotiables

- **Evaluate on held-out days.** Person C gives you a train/test split of days. If a number
  in the paper came from a training day, the paper is wrong.
- **Five seeds, mean ± std on everything reported.** A reviewer will ask. Do not drop below
  three under any circumstances.
- **`results/` is append-only.** Every run gets a `run_id`. Never overwrite a run to fix it
  — make a new id.

## What you write on Day 5

The RL formulation (MDP definition, state, action, reward), the algorithms and
hyperparameters, and the experimental setup.
