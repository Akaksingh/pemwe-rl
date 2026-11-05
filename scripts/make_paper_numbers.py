"""Generate paper/numbers_track2.tex from the raw benchmark CSVs.

    python scripts/make_paper_numbers.py [--dir results_hpc] [--out paper/numbers_track2.tex]

Run scripts/analyze_hpc.py first (it writes hpc_summary.csv). Every macro here is computed from
a CSV; a macro whose source file or row is missing is simply not written, so the manuscript
keeps showing a red TODO for it instead of a stale or invented value.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sig(x: float, n: int = 3) -> str:
    """Round to n significant figures, as a siunitx \\num{} argument."""
    if x == 0 or not np.isfinite(x):
        return "0"
    d = n - int(np.floor(np.log10(abs(x)))) - 1
    return f"{round(x, d):.{max(d, 0)}f}"


def num(x: float, n: int = 3) -> str:
    return r"\num{" + sig(x, n) + "}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "results_hpc"))
    ap.add_argument("--out", default=str(ROOT / "paper" / "numbers_track2.tex"))
    a = ap.parse_args()
    d = Path(a.dir)
    M: dict[str, str] = {}

    sp = d / "hpc_summary.csv"
    if sp.exists():
        t = pd.read_csv(sp)
        st = t[(t.experiment == "raw_env_strong_scaling") & (t.placement == "unpinned")].set_index("n_workers")
        if 1 in st.index:
            M["NumSimOne"] = num(st.loc[1, "steps_per_s"])
            M["NumSimDaysPerSecOne"] = sig(st.loc[1, "sim_days_per_wall_s"], 2)
        for n, key in ((8, "Eight"), (16, "Sixteen")):
            if n in st.index:
                M[f"NumSim{key}"] = num(st.loc[n, "steps_per_s"])
                M[f"NumSpeed{key}"] = sig(st.loc[n, "speedup"], 2)
                M[f"NumEff{key}"] = sig(st.loc[n, "efficiency"], 2)
        smt = t[(t.experiment == "raw_env_strong_scaling") & (t.placement == "smt_pairs")].set_index("n_workers")
        if 2 in smt.index:
            M["NumSmtTwo"] = sig(smt.loc[2, "speedup"], 3)
        if 8 in st.index and 16 in st.index:
            M["NumSmtGain"] = sig((st.loc[16, "steps_per_s"] / st.loc[8, "steps_per_s"] - 1.0) * 100.0, 2)
        allp = t[t.experiment == "raw_env_strong_scaling"]
        if len(allp):
            best = allp.sim_days_per_wall_s.max()
            M["NumSimDaysPerSecMax"] = sig(best, 2)
            M["NumYearSeconds"] = sig(365.0 / best, 2)

    dp = d / "hpc_diag_summary.csv"
    if dp.exists():
        q = pd.read_csv(dp)
        s = q[q["mode"] == "same_work"].set_index("n_workers")
        if 1 in s.index and 8 in s.index:
            M["NumFreqOneGHz"] = sig(s.loc[1, "mean_freq_mhz"] / 1000.0, 2)
            M["NumFreqEightGHz"] = sig(s.loc[8, "mean_freq_mhz"] / 1000.0, 2)
            M["NumPerWorkerOne"] = num(s.loc[1, "per_worker_steps_per_s"])
            M["NumPerWorkerEight"] = num(s.loc[8, "per_worker_steps_per_s"])
            M["NumPerGhzRatioEight"] = sig(s.loc[8, "steps_per_ghz_s"] / s.loc[1, "steps_per_ghz_s"], 2)
            M["NumSlowFastSameWork"] = sig(s["slow_over_fast"].max(), 3)

    import json
    gm = ROOT / "results_e6" / "grid_baselines_test.meta.json"
    if gm.exists():
        j = json.loads(gm.read_text())
        M["NumGridUnits"] = str(j["units"])
        M["NumGridSteps"] = r"\num{" + str(j["env_steps"]) + "}"
        M["NumGridSeconds"] = sig(j["wall_s"], 3)
        M["NumGridWorkers"] = str(j["workers"])
        M["NumGridDaysPerSec"] = sig(j["sim_days_per_wall_s"], 2)

    sp2 = d / "degradation_sensitivity.csv"
    if sp2.exists():
        s = pd.read_csv(sp2)
        nv = s[s.controller == "naive_as_published"].set_index("variant").life_yr
        rl = s[s.controller == "ramp_limited"].set_index("variant").life_yr
        nt = s[s.controller == "no_threshold"].set_index("variant").life_yr
        if len(nv):
            M["NumLifeMin"] = sig(nv.min(), 2)
            M["NumLifeMax"] = sig(nv.max(), 2)
            M["NumLifeSpread"] = sig(nv.max() / nv.min(), 2)
            M["NumLifeNominalNaive"] = sig(nv["nominal"], 2)
            M["NumLifeNominalRamp"] = sig(rl["nominal"], 2)
            M["NumLifeNominalNoThr"] = sig(nt["nominal"], 2)
            order = lambda v: tuple(s_.sort_values(ascending=False).index for s_ in [v])[0]
            ref_order = tuple(pd.DataFrame({"n": nv, "t": nt, "r": rl}).loc["nominal"].sort_values(ascending=False).index)
            tab = pd.DataFrame({"n": nv, "t": nt, "r": rl})
            flips = sum(tuple(tab.loc[v].sort_values(ascending=False).index) != ref_order for v in tab.index)
            M["NumRankFlips"] = str(int(flips))
            M["NumVariants"] = str(int(len(tab)))
    rp = d / "renewable_scenarios_results.csv"
    if rp.exists():
        r = pd.read_csv(rp)
        g = r[r.controller == "naive_as_published"].groupby("scenario").life_yr.mean()
        if len(g):
            M["NumScenLifeMin"] = sig(g.min(), 2)
            M["NumScenLifeMax"] = sig(g.max(), 2)
        h = r.groupby(["scenario", "controller"]).h2_kg_day.mean().unstack()
        if len(h):
            M["NumScenHydrogenSpreadPct"] = sig(((h.max(axis=1) - h.min(axis=1)) / h.max(axis=1) * 100).max(), 2)

    lp = sorted(d.glob("hpc_learner*.csv"))
    if lp:
        L = pd.concat([pd.read_csv(p) for p in lp], ignore_index=True)
        g = L[(L.backend == "cuda") & (L.note == "batch=512")].steps_per_s
        c4 = L[(L.backend == "cpu") & (L.n_workers == 4) & (L.note == "batch=512")].steps_per_s
        c8 = L[(L.backend == "cpu") & (L.n_workers == 8) & (L.note == "batch=512")].steps_per_s
        if len(g):
            M["NumGpuUps"] = num(g.mean(), 3)
        if len(c4):
            M["NumCpuUpsFour"] = num(c4.mean(), 2)
        if len(c8):
            M["NumCpuUpsEight"] = num(c8.mean(), 2)
        if len(g) and len(c8):
            M["NumGpuCpuRatio"] = sig(g.mean() / c8.mean(), 2)

    mp_ = d / "hpc_multi.csv"
    if mp_.exists():
        m = pd.read_csv(mp_)
        agg = m.groupby("case").aggregate_updates_per_s.mean()
        for case, key in (("gpu_x1", "NumMultiGpuOne"), ("gpu_x2", "NumMultiGpuTwo"), ("gpu_x4", "NumMultiGpuFour"),
                          ("cpu4t_x1", "NumMultiCpuOne"), ("cpu4t_x4", "NumMultiCpuFour"),
                          ("mixed_gpu2_cpu4t_x3", "NumMultiMixed")):
            if case in agg.index:
                M[key] = num(agg[case], 3)
        if len(agg):
            M["NumMultiBest"] = num(agg.max(), 3)
            M["NumMultiBestCase"] = r"\texttt{" + agg.idxmax().replace("_", r"\_") + "}"

    lines = ["% GENERATED by scripts/make_paper_numbers.py -- do not edit by hand."]
    for k, v in M.items():
        lines.append(rf"\newcommand{{\{k}}}{{{v}}}")
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {a.out}: {len(M)} macros")
    for k, v in M.items():
        print(f"  {k:24s} {v}")


if __name__ == "__main__":
    main()
