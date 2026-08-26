"""End-to-end pipeline check: train -> evaluate -> schema -> Pareto. OWNER: Person B.

This is NOT the paper's experiment. It is the proof that the apparatus works before ~5.5
GPU-hours are committed to the real matrix, and it is what the brief means by
"sanity-check a single learning curve before launching the full matrix".

It runs a small w2 sweep at a reduced step count, evaluates every checkpoint on the
HELD-OUT test split through the same evaluate.py the real runs use, and checks three
things that would each waste an overnight run if they were wrong:

  1. train.py and evaluate.py actually compose, and the results validate against the
     frozen CONTRACTS.md 3 schema.
  2. The reward components move in OPPOSITE directions as w2 rises. If r_yield and r_deg
     both fall, the agent is not trading -- it is collapsing toward idle.
  3. Neither degenerate policy appears: parked at idle (w2 too high) or pinned at rated
     (w2 too low). These are the two failure modes the Day-3 sweep exists to locate.

    python scripts/pipeline_check.py                     # ~5 min on a laptop
    python scripts/pipeline_check.py --steps 200000      # the brief's coarse-sweep size

Runs are written with a `pipe_` prefix so they can never be confused with real results.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from pemwe import load_config
from pemwe.train import main as train_main
from pemwe.evaluate import evaluate, validate, SB3Policy, _reference

OUT = ROOT / "results" / "pipeline_check"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="sac", choices=["sac", "ppo"])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--n-envs", type=int, default=4)
    # Endpoints and a mid-active point of the CURRENT sweep range (0.1-20). The old
    # default topped out at 100, which is past the w2 ~= 20.2 crossover where idling
    # outscores running -- so the check would have trained a shut-down policy and
    # reported its flatness as a spread failure. See DECISIONS.md 8.
    ap.add_argument("--w2", type=float, nargs="+", default=[0.1, 5.32, 20.0])
    ap.add_argument("--eval-days", type=int, default=10)
    ap.add_argument("--train-days", type=int, default=32)
    # Default to the CONFIGURED device, not "cpu". Hard-coding cpu here meant a run
    # submitted through `gpurun -g 1` held a GPU and then trained on the CPU anyway --
    # burning quota, and producing a throughput number that was mistaken for the GPU rate.
    ap.add_argument("--device", default=None,
                    help="override; defaults to configs/default.yaml train.<algo>.device")
    ap.add_argument("--keep", action="store_true", help="keep the throwaway checkpoints")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"pipeline check: {args.algo.upper()}, {len(args.w2)} w2 values, "
          f"{args.steps:,} steps each, eval on {args.eval_days} HELD-OUT days\n")

    rows = []
    for w2 in args.w2:
        run_id = f"pipe_{args.algo}_w2-{w2}"
        train_main([
            "--algo", args.algo, "--seed", "0", "--run-id", run_id,
            "--total-timesteps", str(args.steps), "--n-envs", str(args.n_envs),
            "--train-days", str(args.train_days),
            *(("--device", args.device) if args.device else ()),
            "--no-subproc", "--override", f"reward.w2={w2}",
        ])
        cfg = load_config(overrides=[f"reward.w2={w2}"])
        pol = SB3Policy(ROOT / "models" / f"{run_id}.zip", args.algo)
        res = evaluate(cfg, pol, args.algo, run_id, seed=0, split="test",
                       max_days=args.eval_days)
        problems = validate(res, _reference())
        (OUT / f"{run_id}.json").write_text(json.dumps(res))
        rows.append((w2, res, problems))

    print(f"\n{'w2':>8}{'H2 kg/day':>11}{'uV/h':>8}{'life yr':>9}"
          f"{'r_yield':>10}{'r_deg':>9}{'duty':>7}{'schema':>8}")
    print("-" * 70)
    for w2, res, problems in rows:
        a = res["aggregate"]
        ry = float(np.mean([e["r_yield"] for e in res["episodes"]]))
        rd = float(np.mean([e["r_deg"] for e in res["episodes"]]))
        duty = float(np.mean([e["mean_eta_lhv"] > 0 for e in res["episodes"]]))
        print(f"{w2:8.3g}{a['h2_kg_mean']:11.1f}{a['deg_rate_uv_per_h']:8.2f}"
              f"{a['projected_life_years']:9.2f}{ry:10.0f}{rd:9.0f}{duty:7.2f}"
              f"{'PASS' if not problems else 'FAIL':>8}")
    print("-" * 70)

    # --- the three checks -------------------------------------------------------------
    ok = True
    schema_ok = all(not p for _, _, p in rows)
    print(f"\n1. train -> evaluate -> frozen schema : {'PASS' if schema_ok else 'FAIL'}")
    ok &= schema_ok

    h2 = np.array([r[1]["aggregate"]["h2_kg_mean"] for r in rows])
    dv = np.array([r[1]["aggregate"]["deg_rate_uv_per_h"] for r in rows])
    # A monotone direction is NOT evidence on its own: a policy that ignores w2 entirely
    # satisfies it by chance half the time. Demand a spread big enough to be a real
    # response, and say INCONCLUSIVE rather than PASS when it is not there.
    MIN_SPREAD = 0.05          # 5% across the w2 range
    spread_h2 = float(np.ptp(h2) / max(h2.mean(), 1e-9))
    spread_dv = float(np.ptp(dv) / max(dv.mean(), 1e-9))
    directional = h2[-1] <= h2[0] and dv[-1] <= dv[0]
    responsive = max(spread_h2, spread_dv) >= MIN_SPREAD
    verdict = "PASS" if (directional and responsive) else "INCONCLUSIVE"
    print(f"2. components trade as w2 rises      : {verdict}")
    print(f"     H2 {h2[0]:.1f} -> {h2[-1]:.1f} kg/day (spread {spread_h2*100:.1f}%), "
          f"degradation {dv[0]:.2f} -> {dv[-1]:.2f} uV/h (spread {spread_dv*100:.1f}%)")
    if not responsive:
        print(f"     The policy barely responds to w2 (<{MIN_SPREAD*100:.0f}% spread), so the")
        print("     direction means nothing yet. Expected while undertrained: with")
        print(f"     learning_starts=10,000 only {max(args.steps-10000,0):,} steps actually")
        print("     train. It is a REAL red flag if it persists at the full 2M steps --")
        print("     that would mean no Pareto frontier and no paper.")
    elif not directional:
        print("     Spread is there but the direction is wrong -- inspect the component")
        print("     logs before committing compute.")

    idle = [w2 for w2, r, _ in rows if r["aggregate"]["h2_kg_mean"] < 10]
    pinned = [w2 for w2, r, _ in rows
              if r["aggregate"]["deg_rate_uv_per_h"] > 8.0]
    print(f"3. no degenerate policy               : "
          f"{'PASS' if not idle and not pinned else 'WARN'}")
    if idle:
        print(f"     parked at idle at w2={idle} -- w2 too high, or the idle term is weak")
    if pinned:
        print(f"     pinned near rated at w2={pinned} -- w2 too low to bite")

    if not args.keep:
        for w2 in args.w2:
            for suf in (".zip", ".json"):
                (ROOT / "models" / f"pipe_{args.algo}_w2-{w2}{suf}").unlink(missing_ok=True)
            shutil.rmtree(ROOT / "results" / "tb" / f"pipe_{args.algo}_w2-{w2}",
                          ignore_errors=True)

    print(f"\nresults in {OUT}")
    print("These are NOT paper numbers -- the step count is a fraction of "
          f"{load_config()['train']['total_timesteps']:,}.")
    print("The real matrix is 13 w2 x 5 seeds x 2 algos on the server "
          "(~83 min, ~5.5 GPU-h).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
