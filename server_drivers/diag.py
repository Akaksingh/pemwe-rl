import json, numpy as np
PMIN = 36.8e3
names = ["pilot_utd1_w2-0.1_seed0", "pilot_utd1_w2-0.1_seed1", "no_threshold_const_plus1",
         "naive_as_published", "ramplimited"]
print(f"{'run':28s}{'H2/d':>7}{'curt kWh/d':>11}{'cyc/d':>7} | rep-day: {'u|avail':>8}{'frac u<.95':>11}{'OFF|avail':>10}{'mean j':>8}")
for n in names:
    d = json.load(open(f"results_pilot/{n}.json"))
    ep = d["episodes"]
    t = d["trajectory"]
    pr, ps, j = (np.array(t[k]) for k in ("p_renew_w", "p_set_w", "j"))
    av = pr > PMIN
    u = ps[av] / pr[av]
    off = (j[av] == 0).mean()
    print(f"{n:28s}{np.mean([e['h2_kg'] for e in ep]):7.1f}{np.mean([e['curtailed_kwh'] for e in ep]):11.0f}"
          f"{np.mean([e['n_cycles'] for e in ep]):7.1f} | {u.mean():8.3f}{(u<0.95).mean():11.3f}{off:10.3f}{j[av].mean():8.3f}")
# per-episode paired comparison vs constant +1 (same days)
ref = {e["date"]: e for e in json.load(open("results_pilot/no_threshold_const_plus1.json"))["episodes"]}
for n in names[:2]:
    ep = json.load(open(f"results_pilot/{n}.json"))["episodes"]
    dh = np.array([e["h2_kg"] - ref[e["date"]]["h2_kg"] for e in ep])
    dd = np.array([e["dv_deg_uv"] - ref[e["date"]]["dv_deg_uv"] for e in ep])
    r = np.array([ (e["r_yield"] - 0.1*e["r_deg"]) - (ref[e["date"]]["r_yield"] - 0.1*ref[e["date"]]["r_deg"]) for e in ep])
    print(f"{n}: paired vs const+1 over {len(ep)} days: dH2 mean {dh.mean():+.1f} kg (days better: {(dh>0).mean():.0%}), "
          f"dDeg {dd.mean():+.1f} uV, d(reward) mean {r.mean():+.0f}  (days SAC reward higher: {(r>0).mean():.0%})")
