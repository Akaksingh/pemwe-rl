"""Enforce the experiment invariants that are easy to break silently.

Run before any sweep, and after anyone edits configs/default.yaml:

    python scripts/validate_experiment_config.py

Every check corresponds to a decision in DECISIONS.md that a well-meaning edit could
quietly undo -- most dangerously swapping the primary SAC run to gradient_steps=8, which
looks like a 2x speedup and is actually a different algorithm.
"""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "configs" / "default.yaml", encoding="utf-8"))

fails, warns = [], []


def check(ok, msg, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        fails.append(msg)


tr, sw = CFG["train"], CFG["sweep"]

print("PPO -- must be CPU-only (DECISIONS.md 4)")
check(tr["ppo"]["device"] == "cpu", "train.ppo.device == cpu", tr["ppo"]["device"])
check(tr["n_envs"] == 32, "n_envs == 32 (measured optimum)", tr["n_envs"])

print("\nSAC -- must preserve the 1:1 update-to-step ratio (DECISIONS.md 4)")
sac = tr["sac"]
check(sac["device"] == "cuda", "train.sac.device == cuda", sac["device"])
check(sac["train_freq"] == tr["n_envs"], "train_freq == n_envs",
      f"{sac['train_freq']} vs {tr['n_envs']}")
check(sac["gradient_steps"] == sac["train_freq"], "gradient_steps == train_freq",
      f"{sac['gradient_steps']} vs {sac['train_freq']}")
if sac["gradient_steps"] != sac["train_freq"]:
    print("         ^ gradient_steps=8 is 4x fewer updates per env step: a DIFFERENT")
    print("           algorithm, valid only as a separate labelled experiment validated")
    print("           at equal ENV STEPS. Never the primary/headline SAC run.")

print("\nSweep -- Pareto frontier resolution (DECISIONS.md 8)")
w2 = sw["w2"]
check(12 <= len(w2) <= 15, "12-15 w2 points", f"{len(w2)} points")
check(min(w2) == 0.1 and max(w2) <= 20.0, "range 0.1 .. 20 (top end is degenerate)",
      f"{min(w2)} .. {max(w2)}")
# Above w2 ~= 20 idling outscores running, so any point up there trains a shut-down
# policy rather than a Pareto point. Measured, not assumed -- see configs/default.yaml.
check(max(w2) < 20.3, "no sweep point in the idle-degenerate regime (w2 >= 20.2)",
      f"max={max(w2)}")
check(w2 == sorted(w2), "w2 ascending")
check(len(set(w2)) == len(w2), "no duplicate w2 values")

print("\nReplication -- must not be traded for sweep points")
check(len(tr["seeds"]) == 5, "5 seeds", str(tr["seeds"]))
check(tr["seeds"] == [0, 1, 2, 3, 4], "seeds are [0,1,2,3,4] as per CONTRACTS.md 6")

print("\nReward -- only w2 is swept")
r = CFG["reward"]
check(r["w1"] == 1.0, "w1 fixed at 1.0", r["w1"])
check(r["w3"] == 0.1, "w3 fixed at 0.1", r["w3"])

print("\nLaunch scripts -- guards present")
ppo_sh = (ROOT / "scripts" / "run_ppo_sweep.sh").read_text(encoding="utf-8")
sac_sh = (ROOT / "scripts" / "run_sac_sweep.sh").read_text(encoding="utf-8")
check("GPU_BROKER_JOB" in ppo_sh, "PPO script refuses to run under gpurun")
check('CUDA_VISIBLE_DEVICES=""' in ppo_sh, "PPO script clears CUDA_VISIBLE_DEVICES")
check("gpurun" not in ppo_sh.split("# ---- guard")[-1].split("if [ \"${1:-}\"")[0]
      or "nohup gpurun" not in ppo_sh, "PPO script never invokes gpurun for training")
check("gpurun -g 1" in sac_sh, "SAC script requests one GPU via gpurun -g 1")
check('"$GS" != "$TF"' in sac_sh, "SAC script rejects gradient_steps != train_freq")

print("\nNo Ollama dependency")
srv = (ROOT / "SERVER.md").read_text(encoding="utf-8")
check("Do not add `-L 11434" in srv, "SERVER.md tells you NOT to open the 11434 tunnel")
check("11434" not in ppo_sh and "11434" not in sac_sh,
      "no launch script references port 11434")

n = len(fails)
print(f"\n{'ALL CHECKS PASSED' if n == 0 else str(n) + ' CHECK(S) FAILED'}")
for f in fails:
    print(f"  - {f}")
sys.exit(1 if n else 0)
