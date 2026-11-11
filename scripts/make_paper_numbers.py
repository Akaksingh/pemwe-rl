"""Generate paper/numbers_track2.tex from the raw benchmark CSVs.

    python scripts/make_paper_numbers.py [--dir results_hpc] [--out paper/numbers_track2.tex]

Run scripts/analyze_hpc.py first (it writes hpc_summary.csv). Every macro here is computed from
a CSV; a macro whose source file or row is missing is simply not written, so the manuscript
keeps showing a red TODO for it instead of a stale or invented value.
"""
from __future__ import annotations

import argparse
import re
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

    def neg(x, n=2):                      # a real minus sign for LaTeX text mode
        return sig(x, n).replace("-", "$-$")

    # ---- evaluation-harness throughput: rules only vs rules + learned policies (same 14 workers)
    for meta, pre in ((ROOT / "results_e5" / "grid_slew_matched.meta.json", "NumEvalRule"),
                      (ROOT / "results_e5" / "grid_e5_test.meta.json", "NumEvalSac")):
        if meta.exists():
            j2 = json.loads(meta.read_text())
            M[pre + "StepsPerSec"] = num(j2["steps_per_s"], 3)
            M[pre + "Units"] = str(j2["units"])
            M[pre + "Steps"] = r"\num{" + str(j2["env_steps"]) + "}"
            M[pre + "Seconds"] = sig(j2["wall_s"], 4)
            M[pre + "Workers"] = str(j2["workers"])

    # ---- E3: update-ratio study
    e3 = d / "e3_quality.csv"
    if e3.exists():
        q = pd.read_csv(e3)
        fin = q[q.steps >= q.steps.max() - 1000].groupby("utd").agg(gap=("gap_pct", "mean"), wall=("wall_min", "mean"),
                                                                    upd=("updates", "mean"))
        first = q[q.steps == q.steps.min()].groupby("utd").agg(gap=("gap_pct", "mean"), wall=("wall_min", "mean"))
        lo_k, hi_k = fin.index.astype(float).argmin(), fin.index.astype(float).argmax()
        lo, hi = fin.index[lo_k], fin.index[hi_k]
        M["NumEThreeShortfallMin"] = sig(abs(fin.gap).min(), 2)
        M["NumEThreeShortfallMax"] = sig(abs(fin.gap).max(), 2)
        M["NumEThreeFirstShortfall"] = sig(abs(first.loc[lo, "gap"]), 2)
        M["NumEThreeFirstWallLow"] = sig(first.loc[lo, "wall"], 2)
        M["NumEThreeWallLow"] = sig(fin.loc[lo, "wall"], 2)
        M["NumEThreeWallHigh"] = sig(fin.loc[hi, "wall"], 2)
        M["NumEThreeWallRatio"] = sig(fin.loc[hi, "wall"] / fin.loc[lo, "wall"], 2)
        M["NumEThreeUpdLow"] = num(fin.loc[lo, "upd"], 3)
        M["NumEThreeUpdHigh"] = num(fin.loc[hi, "upd"], 3)

    # ---- measured update-to-data ratios, from the update counts the library itself logged
    def measured_utd(tag):
        rs = []
        for s_ in (0, 1):
            f_ = ROOT / "results_e3" / f"e3_utd{tag}_s{s_}.csv"
            if f_.exists():
                last = pd.read_csv(f_).iloc[-1]
                rs.append(last.n_updates / (last.timesteps - 10_000))      # learning_starts = 10,000 steps
        return float(np.mean(rs)) if rs else None
    lowr, highr = measured_utd("0.03125"), measured_utd("1")
    if lowr is not None:
        M["NumUtdMeasuredLow"] = sig(lowr, 2)
    if highr is not None:
        M["NumUtdMeasuredHigh"] = sig(highr, 2)

    # ---- does the cheap update ratio still reach the frontier policies at high weights? (training days)
    ck = ROOT / "results_check" / "grid_check_train.csv"
    if ck.exists():
        g_ = pd.read_csv(ck)
        g_ = g_[(g_.scenario == "original") & (g_.variant == "nominal")]
        per = g_.groupby("controller").agg(h2=("h2_kg", "mean"), dv=("dv_deg_uv", "mean"))
        for w_, tag in (("12.9", "Twelve"), ("20.0", "Twenty")):
            for pre, lab in ((f"chk_utd32_w2-{w_}", "Low"), (f"e5_w2-{w_}", "High")):
                s_ = per[per.index.str.startswith(pre)]
                if len(s_):
                    M[f"NumChk{tag}{lab}Hydrogen"] = sig(s_.h2.mean(), 3)
                    M[f"NumChk{tag}{lab}Deg"] = sig(s_.dv.mean(), 3)

    # ---- wall-clock of the full-ratio frontier runs (four runs shared the GPU)
    walls = []
    for f_ in (ROOT / "results_e5").glob("e5_w2-*_s*.csv"):
        walls.append(pd.read_csv(f_).iloc[-1].wall_s / 60.0)
    if walls:
        M["NumEFiveWallMin"] = sig(float(np.mean(walls)), 2)

    # ---- E5: corrected SAC against the best simple rule at the same degradation
    vp = d / "e5_vs_rules.csv"
    if vp.exists():
        v = pd.read_csv(vp).set_index("w2")
        for w, tag in ((1.41, "One"), (5.32, "Five"), (12.9, "Twelve"), (20.0, "Twenty")):
            if w in v.index:
                r_ = v.loc[w]
                M[f"NumSac{tag}HydrogenKg"] = sig(r_.sac_h2, 3)
                M[f"NumSac{tag}Degradation"] = sig(r_.sac_dv, 3)
                M[f"NumRule{tag}HydrogenKg"] = sig(r_.rule_h2_at_same_dv, 3)
                M[f"NumAdv{tag}Kg"] = neg(r_.advantage_kg, 2) if r_.advantage_kg < 0 else sig(r_.advantage_kg, 2)
                M[f"NumAdv{tag}Pct"] = neg(r_.advantage_pct, 2) if r_.advantage_pct < 0 else sig(r_.advantage_pct, 2)
                if "advantage_pct_slewcap_only" in r_.index:       # advantage against slew-limit/power-cap rules only
                    M[f"NumAdv{tag}PctOldRules"] = sig(r_.advantage_pct_slewcap_only, 2)

    # ---- Kaggle follow-up: intermediate update ratios (training days) and extra seeds at ratio 1/4 (test days)
    kd = ROOT / "results_kaggle"
    if (kd / "ratio_per_seed.csv").exists():
        rr = pd.read_csv(kd / "ratio_per_seed.csv")
        gg = rr.groupby("setting").agg(h2=("h2", "mean"), dv=("dv", "mean"))
        full_ = gg.loc["utd 1 (e5)"]
        for key, tag in (("kg_utd0.125", "Eighth"), ("kg_utd0.25", "Quarter")):
            if key in gg.index:
                M[f"NumRatio{tag}Hydrogen"] = sig(gg.loc[key, "h2"], 3)
                M[f"NumRatio{tag}Deg"] = sig(gg.loc[key, "dv"], 3)
                M[f"NumRatio{tag}DegRel"] = sig(gg.loc[key, "dv"] / full_.dv, 3)
                M[f"NumRatio{tag}HydrogenRel"] = sig(gg.loc[key, "h2"] / full_.h2, 3)
    if (kd / "seeds_summary.csv").exists():
        ss = pd.read_csv(kd / "seeds_summary.csv")
        for w, tag in ((12.9, "Twelve"), (20.0, "Twenty")):
            for grp, g_ in (("utd 1/4", "Quarter"), ("utd 1 (", "Full")):
                q = ss[(ss.w2 == w) & ss.group.str.startswith(grp)]
                if not len(q):
                    continue
                q = q.iloc[0]
                M[f"NumKg{g_}{tag}Seeds"] = str(int(q.n_seeds))
                M[f"NumKg{g_}{tag}HydrogenKg"] = sig(q.h2_mean, 3)
                M[f"NumKg{g_}{tag}Deg"] = sig(q.dv_mean, 3)
                M[f"NumKg{g_}{tag}DegMin"] = sig(q.dv_min, 3)
                M[f"NumKg{g_}{tag}DegMax"] = sig(q.dv_max, 3)
                M[f"NumKg{g_}{tag}AdvMean"] = sig(q.adv_pct_mean, 2)
                M[f"NumKg{g_}{tag}AdvMin"] = sig(q.adv_pct_min, 2)
                M[f"NumKg{g_}{tag}AdvMax"] = sig(q.adv_pct_max, 2)
                M[f"NumKg{g_}{tag}Positive"] = str(int(q.n_positive))
    # training wall time of the Kaggle runs (one job at a time on a T4; a different platform from the workstation)
    walls_k = {}
    for f_ in (kd / "run2" / "curves").glob("kg_utd*_w2-20.0_s*.csv"):
        m_ = re.match(r"kg_utd([\d.]+)_", f_.name)
        if m_:
            walls_k.setdefault(m_[1], []).append(pd.read_csv(f_).iloc[-1].wall_s / 60.0)
    for key, tag in (("0.125", "Eighth"), ("0.25", "Quarter")):
        if key in walls_k:
            M[f"NumKgWall{tag}Min"] = sig(float(np.mean(walls_k[key])), 2)

    # ---- robustness: tuned slew limit and matched slew limits
    sm = d / "sac_sensitivity_summary.csv"
    if sm.exists():
        t = pd.read_csv(sm)
        for w, tag in ((12.9, "Twelve"), (20.0, "Twenty")):
            for kind, k in (("coefficient variant", "Coef"), ("resource scenario", "Scen")):
                r_ = t[(t.series == f"SAC w2={w:g}") & (t.kind == kind)]
                if len(r_):
                    r_ = r_.iloc[0]
                    M[f"NumTuned{tag}{k}Out"] = str(int(r_.sac_outlives_rule))
                    M[f"NumTuned{tag}{k}Rows"] = str(int(r_.rows))
                    M[f"NumTuned{tag}{k}MinRatio"] = sig(r_.life_ratio_min, 2)
                    M[f"NumTuned{tag}{k}HydrogenPct"] = sig(abs(r_.h2_vs_rule_pct_mean), 2)
    mm = d / "sac_matched_summary.csv"
    if mm.exists():
        t = pd.read_csv(mm)
        for (w, rule), tag in (((12.9, "slew_0.0015"), "Twelve"), ((20.0, "slew_0.0005"), "Twenty")):
            for kind, k in (("coefficient variant", "Coef"), ("resource scenario", "Scen")):
                r_ = t[(t.sac == f"SAC w2={w:g}") & (t.rule == rule) & (t.kind == kind)]
                if len(r_):
                    r_ = r_.iloc[0]
                    M[f"NumMatch{tag}{k}Out"] = str(int(r_.sac_outlives))
                    M[f"NumMatch{tag}{k}Rows"] = str(int(r_.rows))
                    M[f"NumMatch{tag}{k}MinRatio"] = sig(r_.life_ratio_min, 2)
                    M[f"NumMatch{tag}{k}MedRatio"] = sig(r_.life_ratio_median, 2)
                    M[f"NumMatch{tag}{k}HydrogenPct"] = sig(r_.h2_vs_rule_pct, 2)
                    M[f"NumMatch{tag}RuleDv"] = sig(r_.nominal_rule_dv, 3)
                    M[f"NumMatch{tag}RuleHydrogen"] = sig(r_.nominal_rule_h2, 3)

    # ---- what the policies do (probe on a day subset)
    pp = d / "policy_probe.csv"
    if pp.exists():
        pr = pd.read_csv(pp)
        def row(pol, lo, hi):
            x = pr[(pr.policy == pol) & (abs(pr.bin_lo - lo) < 1e-9) & (abs(pr.bin_hi - hi) < 1e-9)]
            return x.iloc[0] if len(x) else None
        for pol, tag in (("e3_utd1_s0", "E"), ("e5_w2-20.0_s0", "Twenty")):
            a0 = row(pol, 0.035, 1.0)
            if a0 is not None:
                M[f"NumProbe{tag}MeanU"] = sig(a0.mean_u, 2)
                M[f"NumProbe{tag}Off"] = sig(a0.share_off * 100, 2)
        lo_ = row("e3_utd1_s0", 0.035, 0.05)
        hi_ = row("e3_utd1_s0", 0.7, 1.0)
        if lo_ is not None:
            M["NumProbeEOffLow"] = sig(lo_.share_off * 100, 2)
        if hi_ is not None:
            M["NumProbeEUHigh"] = sig(hi_.mean_u, 3)

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

    # ---- derived: learned-policy evaluation slowdown, and where the earlier SAC runs stopped
    if "NumEvalRuleStepsPerSec" in M:
        jr = json.loads((ROOT / "results_e5" / "grid_slew_matched.meta.json").read_text())["steps_per_s"]
        js = json.loads((ROOT / "results_e5" / "grid_e5_test.meta.json").read_text())["steps_per_s"]
        M["NumEvalRatio"] = sig(jr / js, 2)
    import glob as _glob
    import re as _re
    legacy = {}
    for f in _glob.glob(str(ROOT / "results" / "sac_w2-*_seed*.json")):
        w = float(_re.search(r"w2-([\d.]+)_seed", f).group(1))
        legacy.setdefault(w, []).append(json.load(open(f))["aggregate"]["dv_deg_uv_mean"])
    if legacy:
        M["NumLegacyMinDv"] = sig(min(float(np.mean(v)) for v in legacy.values()), 3)

    lines = ["% GENERATED by scripts/make_paper_numbers.py -- do not edit by hand."]
    for k, v in M.items():
        lines.append(rf"\newcommand{{\{k}}}{{{v}}}")
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {a.out}: {len(M)} macros")

    # ---- fallbacks: any \Num... macro the manuscript uses but the data did not supply shows as a red TODO
    tex_path = ROOT / "paper" / "main_track2.tex"
    if tex_path.exists():
        tex = tex_path.read_text(encoding="utf-8")
        bad = sorted(set(_re.findall(r"\\(Num[A-Za-z]*\d[A-Za-z0-9]*)", tex)))
        if bad:
            print("!! macro names containing digits (LaTeX cannot parse these):", bad)
        used = set(_re.findall(r"\\(Num[A-Za-z]+)", tex))
        missing = sorted(used - set(M))
        fb = ["% GENERATED by scripts/make_paper_numbers.py: red-TODO fallbacks for macros used in the text but not generated."]
        fb += [rf"\providecommand{{\{k}}}{{\TODO{{{k}}}}}" for k in missing]
        (ROOT / "paper" / "numbers_fallback.tex").write_text("\n".join(fb) + "\n", encoding="utf-8")
        print(f"fallbacks written for {len(missing)} macro(s) not yet generated: {missing[:10]}")
        unused = sorted(set(M) - used)
        print(f"generated but unused in the text: {len(unused)}")


if __name__ == "__main__":
    main()
