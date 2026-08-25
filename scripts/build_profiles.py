"""Build data/processed/kutch_2019_1min.parquet. OWNER: Person C.

    python scripts/build_profiles.py --source open_meteo      # keyless, ERA5
    NINJA_TOKEN=... python scripts/build_profiles.py --source renewables_ninja

PROVENANCE -- read before using the output in the paper
-------------------------------------------------------
DECISIONS.md section 6 locks Renewables.ninja (MERRA-2), cited via Pfenninger & Staffell
2016, ref [11]. That needs a free account, so this script also supports an ERA5 fallback
through Open-Meteo, which needs no key.

They are not equivalent:

  renewables_ninja  gives PV and wind CAPACITY FACTORS. The PV and turbine models are the
                    provider's, published and peer-reviewed. Cite ref [11].
  open_meteo        gives raw WEATHER. The PV model (PVWatts-style, fixed 25 deg tilt) and
                    the V80 turbine curve in profiles.py are OURS. Citing ref [11] for this
                    would be wrong -- cite ERA5 (Hersbach et al. 2020) and describe the
                    conversion in Methodology.

Which one produced a file is written into the sidecar JSON and into df.attrs, so a run can
always be traced back. Get a token before the paper is written if you can: it removes two
of our own models from the chain of custody.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from pemwe import profiles as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["open_meteo", "renewables_ninja"],
                    default="open_meteo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--year", type=int, default=P.YEAR)
    args = ap.parse_args()

    print(f"site   : {P.LAT} N, {P.LON} E  (Kutch, Gujarat -- DECISIONS.md 6)")
    print(f"year   : {args.year}")
    print(f"source : {args.source}\n")

    if args.source == "open_meteo":
        wx = P.fetch_open_meteo(year=args.year)
        print(f"fetched {len(wx)} hourly weather rows")
        pv_h = P.pv_power_w(wx)
        wind_h = P.wind_power_w(wx)
        note = ("ERA5 via Open-Meteo. PV = PVWatts-style model (25 deg fixed tilt, "
                "10% system loss, -0.35%/K); wind = Vestas V80-2.0MW curve at 100 m. "
                "Both conversions are ours -- cite ERA5, NOT Pfenninger & Staffell.")
    else:
        cf = P.fetch_renewables_ninja(year=args.year)
        print(f"fetched {len(cf)} hourly capacity-factor rows")
        pv_h = (cf["pv_cf"] * P.PV_RATED_W).rename("pv_w")
        wind_h = (cf["wind_cf"] * P.WIND_RATED_W).rename("wind_w")
        note = ("Renewables.ninja (MERRA-2). Provider PV and turbine models. "
                "Cite Pfenninger & Staffell 2016, ref [11]. CC BY-NC 4.0.")

    print(f"  PV   mean CF {pv_h.mean()/P.PV_RATED_W:6.3f}   peak {pv_h.max()/1e3:7.1f} kW")
    print(f"  wind mean CF {wind_h.mean()/P.WIND_RATED_W:6.3f}   peak {wind_h.max()/1e3:7.1f} kW")

    print("\nupsampling to 1-minute (OU cloud transients / turbulence)...")
    pv = P.upsample(pv_h, "pv", seed=args.seed, rated_w=P.PV_RATED_W)
    wind = P.upsample(wind_h, "wind", seed=args.seed + 1, rated_w=P.WIND_RATED_W)

    df = pd.DataFrame({"pv_w": pv, "wind_w": wind})
    # hybrid plant: 1 MW PV + 1 MW wind feeding one 1 MW electrolyzer, so the combined
    # input is genuinely capped and curtailment is a real decision rather than an artifact
    df["hybrid_w"] = np.minimum(df["pv_w"] + df["wind_w"], 1.0e6)

    # --- verification: hourly means must survive upsampling exactly ---------
    for name, fine, coarse in (("pv", pv, pv_h), ("wind", wind, wind_h)):
        got = fine.resample("h").mean().to_numpy()
        want = coarse.to_numpy(dtype=float)
        n = min(len(got), len(want))
        err = np.abs(got[:n] - want[:n]).max()
        rel = err / max(want[:n].max(), 1.0)
        print(f"  {name:5s} hourly-mean preservation: max abs err {err:.3e} W "
              f"({rel:.2e} of peak)  {'OK' if rel < 1e-9 else 'FAIL'}")
        assert rel < 1e-9, f"{name}: upsampling did not preserve hourly means"

    full_days = df.groupby(df.index.date).size()
    keep = [str(d) for d, n in full_days.items() if n == P.STEPS_PER_DAY]
    df = df.loc[df.index.normalize().astype(str).isin(keep)]
    print(f"\n{len(keep)} complete days, {len(df)} minute rows")

    P.PROCESSED.mkdir(parents=True, exist_ok=True)
    out = P.PROCESSED / f"kutch_{args.year}_1min.parquet"
    df.to_parquet(out)

    splits = P.build_splits(df, seed=args.seed)
    (P.PROCESSED / "splits.json").write_text(json.dumps(splits, indent=2))
    out.with_suffix(".json").write_text(json.dumps(
        {"source": args.source, "note": note, "lat": P.LAT, "lon": P.LON,
         "year": args.year, "seed": args.seed,
         "pv_rated_w": P.PV_RATED_W, "wind_rated_w": P.WIND_RATED_W}, indent=2))

    print(f"\nwrote {out}  ({out.stat().st_size/1e6:.1f} MB)")
    print(f"      {P.PROCESSED/'splits.json'}   "
          f"{len(splits['train'])} train / {len(splits['test'])} test days")
    print("\narchetype days (chosen from the year's distribution, not by eye):")
    st = P.day_stats(df)
    for k, d in splits["archetypes"].items():
        r = st.loc[d]
        print(f"  {k:20s} {d}   {r['energy_kwh']:7.0f} kWh   "
              f"intermittency {r['intermittency']:.4f}")

    print(f"\nPROVENANCE: {note}")
    print("\nTell Person A this file exists -- the degradation calibration must be re-run")
    print("on it (DECISIONS.md 5): python scripts/calibrate_degradation.py --solve")


if __name__ == "__main__":
    main()
