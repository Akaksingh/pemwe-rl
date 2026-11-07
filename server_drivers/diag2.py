import json, numpy as np
ref = {e["date"]: e for e in json.load(open("results_pilot/no_threshold_const_plus1.json"))["episodes"]}
rh = np.mean([e["h2_kg"] for e in ref.values()]); rv = np.mean([e["dv_deg_uv"] for e in ref.values()])
print(f"reference constant +1 (train days): H2 {rh:.1f} kg/d, dV {rv:.1f} uV/d")
for n in ("pilot_utd1_w2-0.1_seed0", "pilot_utd1_g0.99_seed0", "pilot_utd1_g0.95_seed0"):
    try: ep = json.load(open(f"results_pilot/{n}.json"))["episodes"]
    except Exception as ex: print(n, "missing", ex); continue
    dh = np.array([e["h2_kg"] - ref[e["date"]]["h2_kg"] for e in ep])
    dd = np.array([e["dv_deg_uv"] - ref[e["date"]]["dv_deg_uv"] for e in ep])
    r = np.array([(e["r_yield"] - 0.1 * e["r_deg"]) - (ref[e["date"]]["r_yield"] - 0.1 * ref[e["date"]]["r_deg"]) for e in ep])
    print(f"{n:26s} H2 {np.mean([e['h2_kg'] for e in ep]):6.1f}  dV {np.mean([e['dv_deg_uv'] for e in ep]):6.1f}  "
          f"cycles {np.mean([e['n_cycles'] for e in ep]):5.1f}  curt {np.mean([e['curtailed_kwh'] for e in ep]):5.0f} kWh | "
          f"vs const+1: dReward {r.mean():+7.0f}/day, SAC better on {(r>0).mean():.0%} of {len(ep)} days")
