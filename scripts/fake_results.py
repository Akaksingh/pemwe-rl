"""Emit results files in the FINAL schema, filled with plausible fake numbers.

This exists so Person C can build and finish all six figures on Day 1-2, before a single
real result exists. When Person B's real runs land on Day 3, the plotting code does not
change -- only the numbers in it do. That is what buys the parallelism.

    python scripts/fake_results.py        # writes results/*.json

CONTRACTS.md section 3 is the schema. If you change it here, change it there, and tell
the other two at standup.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "results"


def make_run(run_id, policy, seed, w2, h2_mu, deg_mu, rng, n_days=10):
    eps = []
    for d in range(n_days):
        h2 = float(rng.normal(h2_mu, h2_mu * 0.08))
        dv = float(rng.normal(deg_mu, deg_mu * 0.12))
        eps.append({
            "date": f"2019-03-{d+1:02d}",
            "h2_kg": h2, "dv_deg_uv": dv,
            "mean_eta_lhv": float(rng.normal(0.70, 0.01)),
            "curtailed_kwh": float(rng.normal(400, 60)),
            "n_cycles": int(rng.integers(2, 14)),
            "mean_abs_ramp": float(rng.normal(0.03, 0.01)),
            "reward_total": h2 - w2 * dv,
            "r_yield": h2, "r_deg": dv, "r_ramp": float(rng.normal(40, 8)),
        })
    h2s = np.array([e["h2_kg"] for e in eps])
    dvs = np.array([e["dv_deg_uv"] for e in eps])
    rate = float(dvs.mean() / 24.0)
    t = np.arange(1440)
    renew = np.clip(np.sin((t / 60 - 6) / 12 * np.pi), 0, None) ** 1.3 * 1.0e6
    # cloud transients, so the baseline and the RL policy have something to differ ABOUT.
    # Without these the two trajectories are identical and the overlay figure silently
    # tests nothing.
    gust = np.zeros(1440)
    for i in range(1, 1440):
        gust[i] = 0.93 * gust[i - 1] + rng.normal(0, 0.30)
    renew = np.clip(renew * (1 + 0.22 * gust), 0, 1.0e6)
    if policy.startswith("baseline"):
        p_set = renew.copy()                      # chases every transient (ref [8] law)
    else:
        # RL: smoothed, and it declines to follow the top of the envelope
        k = 41
        p_set = np.convolve(renew, np.ones(k) / k, mode="same") * 0.88
    return {
        "run_id": run_id, "policy": policy, "seed": seed,
        "weights": {"w1": 1.0, "w2": w2, "w3": 0.1},
        "profile_set": "test",
        "episodes": eps,
        "aggregate": {
            "h2_kg_mean": float(h2s.mean()), "h2_kg_std": float(h2s.std()),
            "dv_deg_uv_mean": float(dvs.mean()), "dv_deg_uv_std": float(dvs.std()),
            "deg_rate_uv_per_h": rate,
            "projected_life_years": 177000.0 / rate / 8760.0,
        },
        "trajectory": {
            "t_min": t.tolist(),
            "p_renew_w": renew.tolist(),
            "p_set_w": p_set.tolist(),
            "j": (p_set / 1.0e6 * 1.7).tolist(),
            "dv_deg_total_uv": np.cumsum(np.full(1440, dvs.mean() / 1440)).tolist(),
        },
    }


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    runs = []
    # baselines: high yield, high degradation
    runs.append(make_run("baseline_naive", "baseline_naive", 0, 1.0, 126.0, 65.0, rng))
    runs.append(make_run("baseline_ramplimited", "baseline_ramplimited", 0, 1.0, 123.0, 59.0, rng))
    # RL family across the w2 sweep -- this is what traces the Pareto frontier
    for w2, h2, dv in [(0.1, 127.0, 55.0), (1.0, 124.0, 38.0),
                       (10.0, 118.0, 26.0), (100.0, 96.0, 15.0)]:
        for seed in range(5):
            runs.append(make_run(f"sac_w2-{w2}_seed{seed}", "sac", seed, w2, h2, dv, rng))
    for r in runs:
        (OUT / f"{r['run_id']}.json").write_text(json.dumps(r))
    print(f"wrote {len(runs)} fake result files to {OUT}")
    print("Person C: build all six figures against these. Do not wait for real runs.")


if __name__ == "__main__":
    main()
