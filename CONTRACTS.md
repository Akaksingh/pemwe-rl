# Interface Contracts — frozen at Day 1 kickoff

Three people over five days only works if nobody waits. These are the seams. Once agreed
at kickoff they are **frozen**: changing a signature here requires all three to agree in
the standup, because a silent change breaks two other people's work in progress.

The enabling trick: **the stub exists before the real thing does.** `src/pemwe/env.py`
already runs end-to-end with placeholder physics, and `scripts/fake_results.py` already
emits results files in the final schema. So Person B trains against a working env on Day 1
hour 1, and Person C's plotting code is finished and tested before a single real number
exists. When A and B land the real versions, nothing downstream changes.

---

## 1. Environment — `pemwe.env.PEMWEEnv` (owner: A)

Standard Gymnasium API. `reset(seed, options) -> (obs, info)`, `step(action) -> (obs, reward, terminated, truncated, info)`.

**Observation** — `Box(shape=(8,), dtype=float32)`, all normalized to roughly [0, 1] or [-1, 1]:

| idx | name | meaning |
|---|---|---|
| 0 | `p_renew` | available renewable power / P_rated |
| 1 | `p_renew_ma15` | 15-min trailing mean of the above (gives the agent volatility context) |
| 2 | `p_set_prev` | previous setpoint / P_rated |
| 3 | `j_frac` | current density / j_rated |
| 4 | `deg_norm` | cumulative ΔV_deg / ΔV_eol |
| 5 | `h2_rate_norm` | instantaneous H₂ rate / rate at rated |
| 6 | `tod_sin` | sin(2π · hour/24) |
| 7 | `tod_cos` | cos(2π · hour/24) |

**Action** — `Box(low=-1, high=1, shape=(1,))`, rescaled internally to `u in [0, 1]` and
applied as a **fraction of the power currently AVAILABLE**:

```
p_set = min(u * p_renew, p_rated)
```

Not a fraction of rated. That version is degenerate: the resource averages ~0.31 of rated,
so every action above ~0.31 gives an identical setpoint and ~69% of the action range is a
flat plateau with no gradient. Trained against it, PPO saturated at maximum power and
returned byte-identical policies across all 13 w2 values *and* all 5 seeds -- no frontier.

As a fraction of what is available, every action has a distinct effect whenever the
resource is non-zero, and deliberate curtailment is directly expressible. The
renewable-only hard cap (decision #3) then holds by construction, since `u <= 1`.

Scripted controllers are still written in terms of a target power fraction *of rated* --
which is how the literature states them -- and convert at the end; see
`baselines._to_action`. Both baselines reproduce byte-identical behaviour across the
change, which is what confirmed it was a reparameterisation and not a different plant.

**`info` dict — every key required at every step.** Person C's analysis reads only this:

```python
{
  "p_renew_w":    float,  # available renewable power, W
  "p_set_w":      float,  # commanded setpoint, W
  "p_stack_w":    float,  # power actually into the stack (after BoP), W
  "j":            float,  # current density, A/cm^2
  "v_cell":       float,  # cell voltage, V
  "h2_kg":        float,  # H2 produced this step, kg
  "eta_lhv":      float,  # LHV system efficiency, -
  "dv_deg_uv":    float,  # degradation ADDED this step, microvolts
  "dv_deg_total_uv": float,  # cumulative
  "is_on":        bool,
  "cycled":       bool,   # an ON<->OFF transition happened this step
  "curtailed_w":  float,  # p_renew - p_set, W
  "r_yield":      float,  # reward components, UNWEIGHTED
  "r_deg":        float,
  "r_ramp":       float,
}
```

Reward components must be logged **unweighted and separately**. Total reward alone cannot
diagnose reward hacking, and reward hacking is the failure mode most likely to eat a day.

## 2. Renewable profiles — `pemwe.profiles` (owner: C)

```python
load_profiles(path="data/processed/kutch_2019_1min.parquet") -> pd.DataFrame
# index: tz-naive DatetimeIndex at 1-min resolution
# columns: pv_w, wind_w, hybrid_w   (W, for a 1 MW-rated plant of each type)

get_day(df, date: str, source: str = "hybrid") -> np.ndarray  # shape (1440,), W
SPLITS: dict  # {"train": [dates...], "test": [dates...], "archetypes": {"sunny": date, ...}}
```

The env takes a `profile: np.ndarray` of shape `(1440,)` in W. That is the entire coupling
between A and C — one array. A uses a synthetic sine-plus-noise profile until C's data lands.

## 3. Results schema (owners: B writes, C reads)

Every evaluation run writes `results/<run_id>.json`. **This schema is the contract that
lets C finish all plotting before any real results exist.**

```jsonc
{
  "run_id": "sac_w2-1.0_seed3",
  "policy": "sac" | "ppo" | "baseline_naive" | "baseline_ramplimited",
  "seed": 3,
  "weights": {"w1": 1.0, "w2": 1.0, "w3": 0.1},
  "profile_set": "test" | "archetype_sunny" | "longhorizon_90d",
  "episodes": [
    {"date": "2019-03-14",
     "h2_kg": 412.7, "dv_deg_uv": 71.2, "mean_eta_lhv": 0.712,
     "curtailed_kwh": 118.4, "n_cycles": 6, "mean_abs_ramp": 0.031,
     "reward_total": 388.1, "r_yield": 412.7, "r_deg": 71.2, "r_ramp": 44.6}
  ],
  "aggregate": {"h2_kg_mean": 0.0, "h2_kg_std": 0.0,
                "dv_deg_uv_mean": 0.0, "dv_deg_uv_std": 0.0,
                "deg_rate_uv_per_h": 0.0, "projected_life_years": 0.0},
  "trajectory": {"t_min": [], "p_renew_w": [], "p_set_w": [], "j": [], "dv_deg_total_uv": []}
}
```

`trajectory` is written for **one representative episode per run only** (file size), and
that is what the example-day figure plots.

## 4. Figures — `pemwe.plots` (owner: C)

Each takes a list of result dicts and writes a PDF into `results/figures/`. Vector, not
PNG — IEEE reviewers zoom.

| fn | figure |
|---|---|
| `fig_pareto(results)` | **headline**: cumulative H₂ vs cumulative degradation, baselines as points, RL w₂-sweep as a frontier |
| `fig_trajectory(result)` | one day: renewable input, RL setpoint, baseline setpoint overlaid |
| `fig_training_curves(logdirs)` | reward + the three components separately, mean ± std over seeds |
| `fig_ablation(results)` | reward-weight sweep and solar-heavy vs wind-heavy |
| `fig_longhorizon(results)` | 90-day cumulative degradation, RL vs baselines, with EoL line |
| `fig_validation(env)` | polarization + efficiency curves (Methodology figure, and the Day-2 gate evidence) |

## 5. Config — `configs/default.yaml`

Single source of truth for every physical constant and reward weight. **No magic numbers
in code.** Ablations are `--override w2=2.0` on the command line, never edited files —
otherwise you cannot reconstruct which config produced which number on Day 5.

## 6. Conventions

- Branches `a/<topic>`, `b/<topic>`, `c/<topic>`; merge to `main` daily before standup. Never commit to `main` directly.
- Never commit anything under `results/` except figures. Raw data under `data/raw/` is gitignored — C shares the processed parquet.
- Every run gets a `run_id`; `results/` is append-only. Do not overwrite a run to "fix" it — make a new id.
- Random seeds: `[0, 1, 2, 3, 4]` everywhere. Five seeds is the minimum a reviewer accepts.
