#!/usr/bin/env bash
# PPO Pareto sweep -- CPU ONLY, never through the GPU broker.
#
# Measured: PPO is ~30% SLOWER on an H200 than on the 128-core CPU (BENCHMARK.md).
# The workload is bound by Python env stepping, not matmuls. And `gpurun` bills wall-clock
# time a job HOLDS a GPU regardless of utilisation, so submitting PPO through the broker
# would be both slower AND quota-burning. Hence the hard guard below.
#
#   ./scripts/run_ppo_sweep.sh          # dry run: print the plan, launch nothing
#   ./scripts/run_ppo_sweep.sh --go     # actually launch
set -euo pipefail
cd "$(dirname "$0")/.."

# `python -m pemwe.train` cannot import the package unless src/ is on the path:
# the repo is a plain source tree, not an installed distribution. Both sweeps
# dry-run by default, so this only surfaced the first time one was run for real.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# ---- guard: refuse to run under the GPU broker -----------------------------
if [ -n "${GPU_BROKER_JOB:-}" ]; then
  echo "REFUSING TO RUN." >&2
  echo "This script was submitted through gpurun (GPU_BROKER_JOB=${GPU_BROKER_JOB})." >&2
  echo "PPO is CPU-only: ~30% slower on the H200, and the broker bills held wall-clock." >&2
  echo "Run it over plain SSH instead:  ./scripts/run_ppo_sweep.sh --go" >&2
  exit 1
fi
# Belt and braces: even if a GPU is visible, PPO must not take it.
export CUDA_VISIBLE_DEVICES=""

PY=.venv/bin/python
[ -x "$PY" ] || PY=./.venv/Scripts/python.exe

read -r W2S SEEDS NENVS STEPS DEVICE <<EOF
$($PY - <<'PYEOF'
import yaml
c = yaml.safe_load(open("configs/default.yaml"))
print(",".join(str(w) for w in c["sweep"]["w2"]),
      ",".join(str(s) for s in c["train"]["seeds"]),
      c["train"]["n_envs"], c["train"]["total_timesteps"],
      c["train"]["ppo"]["device"])
PYEOF
)
EOF

if [ "$DEVICE" != "cpu" ]; then
  echo "CONFIG ERROR: train.ppo.device is '$DEVICE', expected 'cpu'. Aborting." >&2
  exit 1
fi

IFS=',' read -ra W2 <<< "$W2S"
IFS=',' read -ra SEED <<< "$SEEDS"
TOTAL=$(( ${#W2[@]} * ${#SEED[@]} ))
CONCURRENCY=4                      # 4 x n_envs=32 saturates 128 cores

echo "PPO sweep  --  CPU only, no gpurun"
echo "  w2 values   : ${#W2[@]}  (${W2[0]} .. ${W2[-1]})"
echo "  seeds       : ${#SEED[@]}  ($SEEDS)"
echo "  n_envs      : $NENVS      steps/run: $STEPS"
echo "  runs        : $TOTAL      concurrency: $CONCURRENCY"
echo "  est. wall   : ~$(( TOTAL * 34 / 10 / CONCURRENCY )) min at the measured 9,712 steps/s"
echo

if [ ! -f src/pemwe/train.py ]; then
  echo "NOTE: src/pemwe/train.py does not exist yet (Person B owns it)."
  echo "      This script is the launch convention it should satisfy."
  echo
fi

if [ "${1:-}" != "--go" ]; then
  echo "Dry run. Nothing launched. Re-run with --go to execute."
  exit 0
fi

mkdir -p logs
for w in "${W2[@]}"; do
  for s in "${SEED[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$CONCURRENCY" ]; do wait -n; done
    id="ppo_w2-${w}_seed${s}"
    echo "launch $id"
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" \
      nohup $PY -m pemwe.train --algo ppo --seed "$s" \
        --override "reward.w2=$w" --run-id "$id" \
        > "logs/${id}.log" 2>&1 &
  done
done
wait
echo "PPO sweep complete: $TOTAL runs."
