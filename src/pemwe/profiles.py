"""Renewable generation profiles for Kutch, Gujarat. OWNER: Person C.

Frozen API (CONTRACTS.md section 2) -- Person A and Person B both call these:

    load_profiles(path) -> pd.DataFrame     1-min index; columns pv_w, wind_w, hybrid_w
    get_day(df, date, source="hybrid")      -> np.ndarray shape (1440,), watts
    SPLITS                                  -> {"train": [...], "test": [...], "archetypes": {...}}

Site is locked by DECISIONS.md section 6: 23.25 N, 69.00 E.

TWO BACKENDS
------------
`renewables_ninja`  the locked primary (DECISIONS.md 6). Returns PV and wind capacity
                    factors directly. Needs a free API token; set NINJA_TOKEN.
`open_meteo`        keyless fallback on ERA5 reanalysis. Returns raw *weather*, so the PV
                    and turbine conversions below are ours rather than the provider's.

The fallback exists because the pipeline, the calibration and the training runs all block
on this file, and Renewables.ninja needs an account nobody had at the time. Which backend
produced a file is recorded in `df.attrs["source"]` and in the sidecar JSON, so no result
can silently lose its provenance. If the paper ships on ERA5, the Methodology and the
citation change with it -- see notes in build_profiles.py.

WHY 1-MINUTE
------------
Both providers are hourly. DECISIONS.md 1-2 requires a 1-minute control step, because at
hourly resolution this is scheduling (ref [9] territory) and the intermittency mechanisms
of ref [4] -- ON/OFF cycling, ramp-driven thermal fatigue -- are invisible. `upsample()`
adds sub-hourly structure with a documented stochastic model and **preserves every hourly
mean exactly**, so no energy is invented.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
DEFAULT_PATH = PROCESSED / "kutch_2019_1min.parquet"

LAT, LON, YEAR = 23.25, 69.00, 2019
STEPS_PER_DAY = 1440

# --- plant ratings: 1 MW of each technology (CONTRACTS.md section 2) ---------
PV_RATED_W = 1.0e6
WIND_RATED_W = 1.0e6


# =============================================================================
# 1. Fetch
# =============================================================================

def fetch_open_meteo(lat=LAT, lon=LON, year=YEAR, cache=True) -> pd.DataFrame:
    """Hourly ERA5 reanalysis. No API key. Returns raw weather, not power."""
    import requests

    RAW.mkdir(parents=True, exist_ok=True)
    cache_file = RAW / f"open_meteo_{lat}_{lon}_{year}.json"
    if cache and cache_file.exists():
        payload = json.loads(cache_file.read_text())
    else:
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat, "longitude": lon,
                "start_date": f"{year}-01-01", "end_date": f"{year}-12-31",
                "hourly": ("shortwave_radiation,direct_normal_irradiance,"
                           "diffuse_radiation,temperature_2m,wind_speed_100m"),
                "timezone": "UTC",
            }, timeout=180)
        r.raise_for_status()
        payload = r.json()
        if cache:
            cache_file.write_text(json.dumps(payload))

    h = payload["hourly"]
    df = pd.DataFrame({
        "ghi": h["shortwave_radiation"],
        "dni": h["direct_normal_irradiance"],   # DNI, not beam-on-horizontal
        "dhi": h["diffuse_radiation"],
        "temp_c": h["temperature_2m"],
        "wind_ms": np.array(h["wind_speed_100m"], dtype=float) / 3.6,  # km/h -> m/s
    }, index=pd.to_datetime(h["time"]))
    return df.astype(float).interpolate(limit_direction="both")


def fetch_renewables_ninja(lat=LAT, lon=LON, year=YEAR, token=None) -> pd.DataFrame:
    """Locked primary source (DECISIONS.md 6). Needs NINJA_TOKEN.

    Returns capacity factors directly, so no PV or turbine model of ours is involved --
    which is exactly why it is the preferred source for the paper.
    """
    import requests

    token = token or os.environ.get("NINJA_TOKEN")
    if not token:
        raise RuntimeError(
            "No Renewables.ninja token. Register free at renewables.ninja, then\n"
            "  export NINJA_TOKEN=...\n"
            "or run the ERA5 fallback: python scripts/build_profiles.py --source open_meteo")

    RAW.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers = {"Authorization": f"Token {token}"}
    common = dict(lat=lat, lon=lon, date_from=f"{year}-01-01", date_to=f"{year}-12-31",
                  dataset="merra2", capacity=1, format="csv", local_time="false")

    out = {}
    for kind, extra in (("pv", dict(system_loss=0.1, tracking=0, tilt=25, azim=180)),
                        ("wind", dict(height=100, turbine="Vestas V80 2000"))):
        cache_file = RAW / f"ninja_{kind}_{lat}_{lon}_{year}.csv"
        if cache_file.exists():
            txt = cache_file.read_text()
        else:
            r = s.get(f"https://www.renewables.ninja/api/data/{kind}",
                      params={**common, **extra}, timeout=300)
            r.raise_for_status()
            txt = r.text
            cache_file.write_text(txt)
        from io import StringIO
        body = txt[txt.index("time,"):] if "time," in txt else txt
        d = pd.read_csv(StringIO(body), index_col=0, parse_dates=True)
        out[kind] = d["electricity"].astype(float)

    return pd.DataFrame({"pv_cf": out["pv"], "wind_cf": out["wind"]})


# =============================================================================
# 2. Weather -> power  (only needed for the ERA5 backend)
# =============================================================================

def _solar_position(idx: pd.DatetimeIndex, lat: float, lon: float):
    """NOAA low-precision solar position. Good to ~0.1 deg, ample for a POA transposition."""
    doy = idx.dayofyear.values.astype(float)
    hour = idx.hour.values + idx.minute.values / 60.0
    g = np.radians((360 / 365.25) * (doy - 81))
    eot = 9.87 * np.sin(2 * g) - 7.53 * np.cos(g) - 1.5 * np.sin(g)      # minutes
    solar_time = hour + (4 * lon + eot) / 60.0
    omega = np.radians(15.0 * (solar_time - 12.0))                        # hour angle
    delta = np.radians(23.45) * np.sin(np.radians(360 / 365.25 * (doy - 81)))
    phi = np.radians(lat)
    cos_z = np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.cos(omega)
    return np.clip(cos_z, -1, 1), delta, omega, phi


def pv_power_w(wx: pd.DataFrame, rated_w=PV_RATED_W, tilt_deg=25.0, azim_deg=180.0,
               system_loss=0.10, gamma=-0.0035, noct=45.0) -> pd.Series:
    """PVWatts-style model: POA irradiance, cell temperature, then a linear power derate.

    Fixed tilt 25 deg / due south, matching the Renewables.ninja parameters in the brief so
    the two backends stay comparable. Isotropic sky transposition.
    """
    cos_z, delta, omega, phi = _solar_position(wx.index, LAT, LON)
    beta, az = np.radians(tilt_deg), np.radians(azim_deg)
    zen = np.arccos(cos_z)
    # solar azimuth measured from north, clockwise
    sin_az = np.cos(delta) * np.sin(omega) / np.maximum(np.sin(zen), 1e-6)
    cos_az = ((np.sin(delta) * np.cos(phi) - np.cos(delta) * np.sin(phi) * np.cos(omega))
              / np.maximum(np.sin(zen), 1e-6))
    # atan2 of these two already gives azimuth measured from NORTH clockwise: at solar
    # noon cos_az -> -1, so solar_az -> pi = due south, matching azim_deg=180 for the
    # panel. Adding another pi (as a first draft did) points the array due north and
    # costs ~20% of the beam component.
    solar_az = np.arctan2(sin_az, cos_az)
    cos_aoi = np.clip(np.cos(zen) * np.cos(beta)
                      + np.sin(zen) * np.sin(beta) * np.cos(solar_az - az), 0, 1)

    dni, dhi, ghi = wx["dni"].values, wx["dhi"].values, wx["ghi"].values
    poa = (dni * cos_aoi                                   # beam
           + dhi * (1 + np.cos(beta)) / 2                  # isotropic sky diffuse
           + ghi * 0.2 * (1 - np.cos(beta)) / 2)           # ground reflected, albedo 0.2
    poa = np.where(cos_z > 0, poa, 0.0)

    t_cell = wx["temp_c"].values + (noct - 20.0) / 800.0 * poa
    p = rated_w * (poa / 1000.0) * (1 + gamma * (t_cell - 25.0)) * (1 - system_loss)
    return pd.Series(np.clip(p, 0, rated_w), index=wx.index, name="pv_w")


# Vestas V80-2.0 MW power curve (m/s -> fraction of rated), the turbine named in the brief.
_V80_U = np.array([0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                   20, 25, 25.001, 40])
_V80_P = np.array([0, 0, .020, .075, .154, .259, .397, .565, .748, .888, .965, .995,
                   1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])


def wind_power_w(wx: pd.DataFrame, rated_w=WIND_RATED_W, availability=0.97) -> pd.Series:
    """Turbine power curve applied to the 100 m wind speed. Cut-out at 25 m/s is a real
    discontinuity and is left in -- it is exactly the kind of event the controller must
    survive."""
    cf = np.interp(wx["wind_ms"].values, _V80_U, _V80_P)
    return pd.Series(np.clip(cf * rated_w * availability, 0, rated_w),
                     index=wx.index, name="wind_w")


# =============================================================================
# 3. Hourly -> 1 minute
# =============================================================================

def _ou_noise(n: int, rng: np.random.Generator, tau_min: float, sigma: float) -> np.ndarray:
    """Ornstein-Uhlenbeck process, unit time step = 1 minute, zero mean."""
    a = np.exp(-1.0 / tau_min)
    x = np.zeros(n)
    eps = rng.normal(0.0, sigma * np.sqrt(1 - a * a), n)
    for i in range(1, n):
        x[i] = a * x[i - 1] + eps[i]
    return x


def _rescale_to_hourly_mean(fine: np.ndarray, hourly: np.ndarray,
                            cap: float, iters: int = 100) -> np.ndarray:
    """Force each 60-minute block to its exact hourly mean while respecting 0 <= x <= cap.

    A single multiplicative rescale is not enough: clipping to the plant rating afterwards
    pulls the mean back off target, which matters for wind, where hours sit close to rated
    and the turbulence term pushes samples through the ceiling. (PV rarely saturates, which
    is why the naive version looked correct until wind was added.)

    This is an alternating projection onto {mean == target} and {0 <= x <= cap}: scale to
    the target, clip, then push the remaining residual only into samples that are still
    strictly inside the box. Converges in a handful of passes; the assertion in
    scripts/build_profiles.py checks it did.
    """
    blocks = fine.reshape(-1, 60).astype(float)
    want = hourly.astype(float)

    have = blocks.mean(axis=1)
    scale = np.divide(want, have, out=np.ones_like(want), where=have > 1e-9)
    blocks = np.clip(blocks * scale[:, None], 0.0, cap)

    for _ in range(iters):
        err = want - blocks.mean(axis=1)
        if np.abs(err).max() < 1e-12:
            break
        free = (blocks > 1e-12) & (blocks < cap - 1e-12)     # samples with headroom
        n_free = free.sum(axis=1)
        # if a block is fully saturated, spread over everything and let the clip settle it
        spread = np.where(n_free > 0, n_free, 60)
        delta = (err * 60.0 / spread)[:, None]
        blocks = np.clip(blocks + np.where(free | (n_free == 0)[:, None], delta, 0.0),
                         0.0, cap)

    blocks[want <= 1e-12] = 0.0          # a zero hour stays exactly zero
    return blocks.reshape(-1)


def upsample(hourly: pd.Series, kind: str, seed: int = 0,
             rated_w: float = 1.0e6) -> pd.Series:
    """Hourly -> 1-minute with documented sub-hourly structure.

    PV   : cloud transients. Ornstein-Uhlenbeck on the clear-sky index, correlation time
           ~12 min, amplitude scaled by how far below the day's envelope the hour already
           sits -- a clear hour flickers little, a broken-cloud hour flickers hard.
    WIND : turbulence. Faster OU (tau ~3 min) with amplitude set by turbulence intensity,
           which is what a Kaimal spectrum delivers in the frequency band that matters at
           a 1-minute step.

    Every hourly mean is preserved exactly (see `_rescale_to_hourly_mean`).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(hourly.index[0], periods=len(hourly) * 60, freq="min")
    # smooth interpolation of the hourly series onto the minute grid
    base = pd.Series(hourly.values, index=hourly.index).reindex(
        hourly.index.union(idx)).interpolate("time").reindex(idx).values
    base = np.nan_to_num(base, nan=0.0)

    if kind == "pv":
        tau, sigma = 12.0, 1.0
        env = pd.Series(hourly.values).rolling(25, center=True, min_periods=1).max().values
        env_min = np.repeat(np.maximum(env, 1e-9), 60)
        # broken cloud => hour well below its local envelope => large fluctuation
        depth = np.clip(1.0 - base / env_min, 0.0, 1.0)
        amp = 0.55 * depth * base
    elif kind == "wind":
        tau, sigma = 3.0, 1.0
        ti = 0.12                                   # turbulence intensity, onshore ~10-15%
        amp = ti * base
    else:
        raise ValueError(kind)

    fine = base + amp * _ou_noise(len(idx), rng, tau, sigma)
    fine = _rescale_to_hourly_mean(fine, hourly.values.astype(float), cap=rated_w)
    return pd.Series(fine, index=idx, name=hourly.name)


# =============================================================================
# 4. Frozen public API (CONTRACTS.md section 2)
# =============================================================================

def load_profiles(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    """1-min DatetimeIndex; columns pv_w, wind_w, hybrid_w in watts."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it first:\n"
            f"  python scripts/build_profiles.py --source open_meteo")
    df = pd.read_parquet(path)
    meta = path.with_suffix(".json")
    if meta.exists():
        df.attrs.update(json.loads(meta.read_text()))
    return df


def get_day(df: pd.DataFrame, date: str, source: str = "hybrid") -> np.ndarray:
    """One day as a (1440,) float array in watts -- the array PEMWEEnv consumes."""
    col = {"pv": "pv_w", "wind": "wind_w", "hybrid": "hybrid_w"}[source]
    day = df.loc[date, col]
    if len(day) != STEPS_PER_DAY:
        raise ValueError(f"{date}: {len(day)} rows, expected {STEPS_PER_DAY}")
    return day.to_numpy(dtype=float)


def day_stats(df: pd.DataFrame, source: str = "hybrid") -> pd.DataFrame:
    """Per-day energy and intermittency. Archetypes are picked from these, so the choice
    is justified by where a day sits in the year rather than by eye."""
    col = {"pv": "pv_w", "wind": "wind_w", "hybrid": "hybrid_w"}[source]
    g = df[col].groupby(df.index.date)
    energy = g.sum() / 60.0 / 1000.0                                  # kWh
    ramp = g.apply(lambda s: np.abs(np.diff(s.to_numpy())).mean())    # W/min
    peak = g.max()
    out = pd.DataFrame({"energy_kwh": energy, "mean_abs_ramp_w": ramp, "peak_w": peak})
    out["intermittency"] = out["mean_abs_ramp_w"] / out["peak_w"].clip(lower=1.0)
    out.index = pd.to_datetime(out.index).strftime("%Y-%m-%d")
    return out


def build_splits(df: pd.DataFrame, test_fraction: float = 0.2, seed: int = 0) -> dict:
    """Month-stratified train/test split, plus the four archetype days.

    Stratified by month so the test set cannot land entirely in one season -- Kutch has a
    strong monsoon/dry split and a random split would be a real confound.
    """
    stats = day_stats(df)
    rng = np.random.default_rng(seed)
    dates = np.array(stats.index)
    months = pd.to_datetime(dates).month

    test = []
    for m in range(1, 13):
        pool = dates[months == m]
        k = max(1, int(round(len(pool) * test_fraction)))
        test.extend(rng.choice(pool, size=k, replace=False).tolist())
    test = sorted(test)
    train = sorted(set(dates) - set(test))

    hi_e = stats["energy_kwh"] > stats["energy_kwh"].quantile(0.6)
    pv = day_stats(df, "pv"); wd = day_stats(df, "wind")
    archetypes = {
        # high energy, low intermittency -> the easy day
        "sunny_consistent": stats[hi_e]["intermittency"].idxmin(),
        # high energy, high intermittency -> the day that punishes a naive controller
        "cloudy_intermittent": stats[hi_e]["intermittency"].idxmax(),
        "windy": wd["energy_kwh"].idxmax(),
        "calm": wd["energy_kwh"].idxmin(),
        "solar_heavy": (pv["energy_kwh"] - wd["energy_kwh"]).idxmax(),
        "wind_heavy": (wd["energy_kwh"] - pv["energy_kwh"]).idxmax(),
    }
    return {"train": train, "test": test, "archetypes": archetypes}


def _load_splits() -> dict:
    f = PROCESSED / "splits.json"
    return json.loads(f.read_text()) if f.exists() else {"train": [], "test": [],
                                                         "archetypes": {}}


SPLITS = _load_splits()


def profiles_array(df: pd.DataFrame, dates: list[str], source: str = "hybrid") -> np.ndarray:
    """(n_days, 1440) array in watts -- what PEMWEEnv(profiles=...) expects."""
    return np.stack([get_day(df, d, source) for d in dates])


def env_profiles(n_days: int = 8, split: str = "train", source: str = "hybrid",
                 path: str | Path = DEFAULT_PATH) -> np.ndarray | None:
    """Deterministic (n_days, 1440) array for calibration and the G1 gates.

    Returns None when the processed file does not exist yet, so callers can fall back to
    the synthetic placeholder instead of crashing. Always takes days from the TRAIN split:
    the calibration is a modelling choice and must not see held-out weather.
    """
    try:
        df = load_profiles(path)
    except FileNotFoundError:
        return None
    sp = build_splits(df)
    days = sp[split][:n_days]
    return profiles_array(df, days, source)
