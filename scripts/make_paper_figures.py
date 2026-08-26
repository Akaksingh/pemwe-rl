"""Regenerate every paper figure and print the numbers the text quotes. OWNER: Person C.

    python scripts/make_paper_figures.py

Reads results/*.json (CONTRACTS.md section 3) plus the TensorBoard logs, writes the six
PDFs into results/figures/, and prints a RESULTS SUMMARY block with the figures the
Results section cites -- so the prose and the plots cannot drift apart. Every number the
paper states about performance should be copied from this output, not retyped from memory.

Runs on whatever exists: if only one algorithm has finished, it reports that one and says
so rather than failing or silently plotting an incomplete matrix as if it were complete.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from pemwe import plots, load_config, PEMWEEnv

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(profile_set=None, policy_prefix=None):
    out = []
    for p in sorted(RESULTS.glob("*.json")):
        if p.name.startswith("pipe_"):          # pipeline-check runs are never paper data
            continue
        d = json.loads(p.read_text())
        if profile_set and d.get("profile_set") != profile_set:
            continue
        if policy_prefix and not str(d.get("policy", "")).startswith(policy_prefix):
            continue
        out.append(d)
    return out


def agg(runs, key):
    v = np.array([r["aggregate"][key] for r in runs], dtype=float)
    return v.mean(), v.std()


def frontier(results, policy):
    """(w2, h2_mean, h2_std, dv_mean, dv_std, n_seeds) per sweep point, ascending w2."""
    by = {}
    for r in results:
        if r["policy"] == policy:
            by.setdefault(r["weights"]["w2"], []).append(r)
    rows = []
    for w in sorted(by):
        h, hs = agg(by[w], "h2_kg_mean")
        d, ds = agg(by[w], "dv_deg_uv_mean")
        rows.append((w, h, hs, d, ds, len(by[w])))
    return rows


def pareto_dominated(px, py, qx, qy):
    """q dominates p if q has >= yield and <= degradation, strictly better in one."""
    return (qy >= py and qx <= px) and (qy > py or qx < px)


def main():
    test = load(profile_set="test")
    if not test:
        print("No test-split results yet. Run: python -m pemwe.evaluate --sweep")
        return

    algos = sorted({r["policy"] for r in test if r["policy"] in ("sac", "ppo")})
    bases = sorted({r["policy"] for r in test if r["policy"].startswith("baseline")})
    print(f"loaded {len(test)} test-split runs   algorithms: {algos or 'none'}   "
          f"baselines: {bases or 'none'}\n")

    # ---------------- figures ----------------
    print("figures")
    if any(r["policy"] == "sac" for r in test):
        plots.fig_pareto(test)
        plots.fig_ablation(test)
    elif any(r["policy"] == "ppo" for r in test):
        # SAC is the configured primary; if only PPO exists, plot PPO and say so loudly
        print("  NOTE: no SAC results -- plotting the PPO frontier instead.")
        print("        The paper must then name PPO as the primary algorithm, or state")
        print("        explicitly that the SAC arm is incomplete. Do not imply otherwise.")
        ppo_as_primary = [dict(r, policy="sac") if r["policy"] == "ppo" else r for r in test]
        plots.fig_pareto(ppo_as_primary)
        plots.fig_ablation(ppo_as_primary)

    rl = [r for r in test if r["policy"] in algos and r.get("trajectory")]
    bl = [r for r in test if r["policy"] == "baseline_naive" and r.get("trajectory")]
    if rl:
        plots.fig_trajectory(rl[len(rl) // 2], bl[0] if bl else None)

    lh = load(profile_set="longhorizon_90d")
    if lh:
        series = {d["policy"]: np.cumsum([e["dv_deg_uv"] for e in d["episodes"]]) for d in lh}
        plots.fig_longhorizon(series)

    plots.fig_validation(PEMWEEnv(load_config(), seed=0))

    # ---------------- numbers the text quotes ----------------
    print("\n" + "=" * 78)
    print("RESULTS SUMMARY  --  copy these into the paper, do not retype from memory")
    print("=" * 78)

    print(f"\n{'policy':<24}{'H2 kg/day':>14}{'dV uV/day':>14}{'runs':>7}")
    print("-" * 78)
    base_pts = {}
    for b in bases:
        runs = [r for r in test if r["policy"] == b]
        h, hs = agg(runs, "h2_kg_mean"); d, ds = agg(runs, "dv_deg_uv_mean")
        base_pts[b] = (d, h)
        print(f"{b:<24}{h:>9.1f}+-{hs:<4.1f}{d:>9.1f}+-{ds:<4.1f}{len(runs):>7}")

    for a in algos:
        rows = frontier(test, a)
        if not rows:
            continue
        seeds = {r[5] for r in rows}
        print(f"\n{a.upper()} frontier -- {len(rows)} w2 points, "
              f"{'/'.join(str(s) for s in sorted(seeds))} seeds each")
        print(f"{'w2':>8}{'H2 kg/day':>14}{'dV uV/day':>14}{'dominates':>28}")
        print("-" * 78)
        for w, h, hs, d, ds, n in rows:
            dom = [b for b, (bx, by) in base_pts.items() if pareto_dominated(bx, by, d, h)]
            print(f"{w:>8.3g}{h:>9.1f}+-{hs:<4.1f}{d:>9.1f}+-{ds:<4.1f}"
                  f"{(', '.join(dom) if dom else '--'):>28}")

        hs_ = [r[1] for r in rows]; ds_ = [r[3] for r in rows]
        print(f"\n  frontier span: H2 {min(hs_):.1f}-{max(hs_):.1f} kg/day "
              f"({(max(hs_)-min(hs_))/max(hs_):.1%}), "
              f"degradation {min(ds_):.1f}-{max(ds_):.1f} uV/day "
              f"({(max(ds_)-min(ds_))/max(ds_):.1%})")
        n_dom = sum(1 for w, h, hs2, d, ds2, n in rows
                    if any(pareto_dominated(bx, by, d, h) for bx, by in base_pts.values()))
        print(f"  {n_dom}/{len(rows)} frontier points Pareto-dominate at least one baseline")
        if n_dom == 0:
            print("  -> NO point dominates a baseline. Say so plainly in the Results; the")
            print("     frontier claim is about reachable operating points, not dominance.")

    if lh:
        print("\n90-day persistent-degradation rollout")
        print("-" * 78)
        for d in sorted(lh, key=lambda x: x["policy"]):
            a2 = d["aggregate"]
            print(f"{d['policy']:<24}{a2.get('deg_rate_uv_per_h', float('nan')):>8.2f} uV/h"
                  f"{a2.get('projected_life_years', float('nan')):>10.2f} yr")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
