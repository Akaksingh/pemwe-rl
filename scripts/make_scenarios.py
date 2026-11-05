"""Renewable-variability scenarios for the sensitivity study (Experiment B).

    python scripts/make_scenarios.py [--out-dir data/scenarios] [--metrics results_hpc/renewable_scenarios.csv]

WHAT THIS DOES
--------------
The shipped profile (data/processed/kutch_2019_1min.parquet) is ERA5 hourly resource upsampled to
1 minute with Ornstein-Uhlenbeck noise (profiles.upsample). Every 1-minute dynamic in the study --
ramping, cycling, curtailment -- comes from that synthetic noise, and no measured 1-minute data
exists here to validate it. This script therefore regenerates the 1-minute series with the noise
amplitude varied, keeping each hourly mean EXACTLY (same projection as the shipped pipeline), so
conclusions can be checked for sensitivity to the one thing that is assumed rather than measured.

The hourly series are recovered from the shipped file (hourly means are preserved by
construction to ~1e-12), so the raw weather cache is not needed.

SCENARIOS (one-factor changes to the shipped amplitudes; correlation times are unchanged)
    original         PV amplitude x1.0, wind turbulence intensity 0.12   (shipped profile, seed 0)
    low_variability  PV amplitude x0.4, wind TI 0.06
    high_solar       PV amplitude x1.6, wind TI 0.12
    high_wind        PV amplitude x1.0, wind TI 0.24
    combined_high    PV amplitude x1.6, wind TI 0.24
Each is generated for REALISATIONS independent noise seeds, so run-to-run noise variation can be
separated from the scenario effect. These multipliers are design choices, NOT fitted to
measurements.

SELF-CHECK: scenario `original`, realisation 0 must reproduce the shipped file; the script aborts
otherwise.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pemwe import profiles as P                       # noqa: E402
from pemwe.config import load_config                  # noqa: E402
from pemwe.stack import StackModel                    # noqa: E402

SCENARIOS = {
    "original":        dict(pv_amp=1.0, wind_ti=0.12),
    "low_variability": dict(pv_amp=0.4, wind_ti=0.06),
    "high_solar":      dict(pv_amp=1.6, wind_ti=0.12),
    "high_wind":       dict(pv_amp=1.0, wind_ti=0.24),
    "combined_high":   dict(pv_amp=1.6, wind_ti=0.24),
}
REALISATIONS = (0, 1, 2)
PV_TAU, WIND_TAU = 12.0, 3.0                          # minutes, as in profiles.upsample


def upsample_scaled(hourly: pd.Series, kind: str, seed: int, rated_w: float,
                    pv_amp: float = 1.0, wind_ti: float = 0.12) -> pd.Series:
    """profiles.upsample with the noise amplitude exposed. Identical RNG use, so
    (pv_amp=1, wind_ti=0.12) reproduces the shipped series."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(hourly.index[0], periods=len(hourly) * 60, freq="min")
    base = pd.Series(hourly.values, index=hourly.index).reindex(
        hourly.index.union(idx)).interpolate("time").reindex(idx).values
    base = np.nan_to_num(base, nan=0.0)
    if kind == "pv":
        tau = PV_TAU
        env = pd.Series(hourly.values).rolling(25, center=True, min_periods=1).max().values
        env_min = np.repeat(np.maximum(env, 1e-9), 60)
        depth = np.clip(1.0 - base / env_min, 0.0, 1.0)
        amp = 0.55 * pv_amp * depth * base
    elif kind == "wind":
        tau = WIND_TAU
        amp = wind_ti * base
    else:
        raise ValueError(kind)
    fine = base + amp * P._ou_noise(len(idx), rng, tau, 1.0)
    fine = P._rescale_to_hourly_mean(fine, hourly.values.astype(float), cap=rated_w)
    return pd.Series(fine, index=idx, name=hourly.name)


def _runs(mask: np.ndarray) -> np.ndarray:
    """Lengths of consecutive True runs."""
    m = np.concatenate([[False], mask, [False]])
    d = np.diff(m.astype(np.int8))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return ends - starts


def metrics(df: pd.DataFrame, p_min_w: float, p_rated_w: float, hourly_ref: dict) -> dict:
    p = df["hybrid_w"].to_numpy()
    n_days = len(p) / 1440.0
    d1 = np.abs(np.diff(p)) / 1e6                         # per-minute change, fraction of 1 MW
    d15 = np.abs(p[15:] - p[:-15]) / 1e6
    d10 = np.abs(p[10:] - p[:-10]) / 1e6
    hourly_mean = p.reshape(-1, 60).mean(axis=1)
    fl = p - np.repeat(hourly_mean, 60)                   # fluctuation about the hourly mean

    def acf(l):
        return float(np.mean(fl[:-l] * fl[l:]) / max(np.mean(fl * fl), 1e-12))

    hrs = p.reshape(-1, 60)
    ok = hrs.mean(axis=1) > 50e3
    cv = float(np.mean(hrs[ok].std(axis=1) / hrs[ok].mean(axis=1)))
    low = p < p_min_w
    low10 = p < 0.10 * p_rated_w
    low_runs, low10_runs = _runs(low), _runs(low10)
    # "major fluctuation": 10-minute change above 20% of 1 MW; consecutive minutes count once
    major = _runs(d10 > 0.20)
    out = {
        "cf_of_1MW": float(p.mean() / 1e6),
        "cf_of_plant_rating": float(p.mean() / p_rated_w),
        "ramp1_mean": float(d1.mean()), "ramp1_p50": float(np.percentile(d1, 50)),
        "ramp1_p95": float(np.percentile(d1, 95)), "ramp1_p99": float(np.percentile(d1, 99)),
        "ramp1_max": float(d1.max()),
        "ramp15_p95": float(np.percentile(d15, 95)),
        "major_events_per_day": float(len(major) / n_days),
        "cv_within_hour_mean": cv,
        "frac_time_below_pmin": float(low.mean()),
        "low_intervals_per_day": float(len(low_runs) / n_days),
        "low_mean_duration_min": float(low_runs.mean()) if len(low_runs) else 0.0,
        "frac_time_below_10pct_rated": float(low10.mean()),
        "acf_fluct_lag1": acf(1), "acf_fluct_lag5": acf(5), "acf_fluct_lag15": acf(15),
    }
    for kind in ("pv", "wind"):
        got = df[f"{kind}_w"].to_numpy().reshape(-1, 60).mean(axis=1)
        out[f"{kind}_hourly_mean_max_abs_err_W"] = float(np.abs(got - hourly_ref[kind]).max())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "scenarios"))
    ap.add_argument("--metrics", default=str(ROOT / "results_hpc" / "renewable_scenarios.csv"))
    ap.add_argument("--realisations", type=int, default=len(REALISATIONS))
    ap.add_argument("--no-write-profiles", action="store_true", help="metrics only")
    a = ap.parse_args()

    cfg = load_config()
    stack = StackModel(cfg)
    base = P.load_profiles(cfg["data"]["profile_path"])
    pv_h = base["pv_w"].resample("h").mean()
    wind_h = base["wind_w"].resample("h").mean()
    hourly_ref = {"pv": pv_h.to_numpy(), "wind": wind_h.to_numpy()}
    print(f"hourly series recovered: {len(pv_h)} h; p_min_on={stack.p_min_on/1e3:.1f} kW, "
          f"p_rated={stack.p_rated/1e3:.1f} kW", flush=True)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    # observed reference row: the shipped file itself
    m0 = metrics(base, stack.p_min_on, stack.p_rated, hourly_ref)
    rows.append({"scenario": "shipped_file", "realisation": "-", "pv_amp": "-", "wind_ti": "-", **m0})

    for name, prm in SCENARIOS.items():
        for r in range(a.realisations):
            pv_seed, wind_seed = 10 * r, 10 * r + 1       # r=0 -> (0, 1): the shipped seeds
            pv = upsample_scaled(pv_h, "pv", pv_seed, P.PV_RATED_W, prm["pv_amp"], prm["wind_ti"])
            wind = upsample_scaled(wind_h, "wind", wind_seed, P.WIND_RATED_W, prm["pv_amp"], prm["wind_ti"])
            df = pd.DataFrame({"pv_w": pv, "wind_w": wind})
            df["hybrid_w"] = np.minimum(df["pv_w"] + df["wind_w"], 1.0e6)
            df = df.loc[df.index.normalize().astype(str).isin(
                base.index.normalize().astype(str).unique())]

            if name == "original" and r == 0:             # self-check against the shipped file
                err = max(np.abs(df["pv_w"].to_numpy() - base["pv_w"].to_numpy()).max(),
                          np.abs(df["wind_w"].to_numpy() - base["wind_w"].to_numpy()).max())
                print(f"self-check original/r0 vs shipped file: max abs diff {err:.3e} W", flush=True)
                if err > 1.0:
                    sys.exit("ABORT: scenario generator does not reproduce the shipped profile")

            m = metrics(df, stack.p_min_on, stack.p_rated, hourly_ref)
            rows.append({"scenario": name, "realisation": r, **prm, **m})
            if not a.no_write_profiles:
                df.to_parquet(out_dir / f"kutch_2019_1min__{name}__r{r}.parquet")
            print(f"{name:16s} r{r}: CF={m['cf_of_1MW']:.3f}  ramp1_p99={m['ramp1_p99']:.3f}  "
                  f"major/day={m['major_events_per_day']:.2f}  low-int/day={m['low_intervals_per_day']:.1f}  "
                  f"max hourly-mean err pv/wind={m['pv_hourly_mean_max_abs_err_W']:.1e}/"
                  f"{m['wind_hourly_mean_max_abs_err_W']:.1e} W", flush=True)

    Path(a.metrics).parent.mkdir(parents=True, exist_ok=True)
    with open(a.metrics, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", a.metrics, len(rows), "rows")


if __name__ == "__main__":
    main()
