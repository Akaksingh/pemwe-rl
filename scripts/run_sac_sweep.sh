#!/usr/bin/env bash
# SAC Pareto sweep -- one H200 per run, through the gpurun broker.
#
# Measured (BENCHMARK.md): properly batched SAC is 6,520 steps/s on an H200 vs 2,450 on CPU
# (2.7x), and 26x faster than SB3's single-env default. The GPU genuinely earns its place
# here, unlike for PPO.
#
#   ./scripts/run_sac_sweep.sh          # dry run: print the plan, launch nothing
#   ./scripts/run_sac_sweep.sh --go     # actually launch
set -euo pipefail
cd "$(dirname "$0")/.."

# `python -m pemwe.train` cannot import the package unless src/ is on the path:
# the repo is a plain source tree, not an installed distribution. Both sweeps
# dry-run by default, so this only surfaced the first time one was run for real.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# Do not nest: each run requests its own GPU through gpurun below.
if [ -n "${GPU_BROKER_JOB:-}" ]; then
  echo "REFUSING TO RUN: already inside a gpurun job (GPU_BROKER_JOB=${GPU_BROKER_JOB})." >&2
  echo "This script submits its own gpurun jobs. Run it over plain SSH." >&2
  exit 1
fi

PY=.venv/bin/python
[ -x "$PY" ] || PY=./.venv/Scripts/python.exe

read -r W2S SEEDS NENVS STEPS DEVICE TF GS <<EOF
$($PY - <<'PYEOF'
import yaml
c = yaml.safe_load(open("configs/default.yaml"))
s = c["train"]["sac"]
print(",".join(str(w) for w in c["sweep"]["w2"]),
      ",".join(str(x) for x in c["train"]["seeds"]),
      c["train"]["n_envs"], c["train"]["total_timesteps"],
      s["device"], s["train_freq"], s["gradient_steps"])
PYEOF
)
EOF

# ---- guard: the primary SAC run must preserve the 1:1 update-to-step ratio ----
if [ "$DEVICE" != "cuda" ]; then
  echo "CONFIG ERROR: train.sac.device is '$DEVICE', expected 'cuda'. Aborting." >&2
  exit 1
fi
if [ "$GS" != "$TF" ] || [ "$GS" != "$NENVS" ]; then
  echo "CONFIG ERROR: primary SAC must keep gradient_steps == train_freq == n_envs." >&2
  echo "  got n_envs=$NENVS train_freq=$TF gradient_steps=$GS" >&2
  echo "  gradient_steps=8 is 4x fewer updates per env step -- a DIFFERENT algorithm," >&2
  echo "  valid only as a separate labelled experiment validated at equal env steps." >&2
  echo "  See DECISIONS.md 4. Aborting." >&2
  exit 1
fi

IFS=',' read -ra W2 <<< "$W2S"
IFS=',' read -ra SEED <<< "$SEEDS"
TOTAL=$(( ${#W2[@]} * ${#SEED[@]} ))
CONCURRENCY=4                      # 4 GPUs

echo "SAC sweep  --  gpurun -g 1 per run"
echo "  w2 values   : ${#W2[@]}  (${W2[0]} .. ${W2[-1]})"
echo "  seeds       : ${#SEED[@]}  ($SEEDS)"
echo "  n_envs      : $NENVS  train_freq: $TF  gradient_steps: $GS   (1:1 ratio preserved)"
echo "  runs        : $TOTAL  concurrency: $CONCURRENCY"
echo "  est. wall   : ~$(( TOTAL * 51 / 10 / CONCURRENCY )) min at the measured 6,520 steps/s"
echo "  est. quota  : ~$(( TOTAL * 51 / 10 / 60 )) GPU-h of the 90/week"
echo
gpurun --status 2>/dev/null || echo "(gpurun not on PATH -- are you on the server?)"
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
    id="sac_w2-${w}_seed${s}"
    echo "launch $id"
    OMP_NUM_THREADS=8 nohup gpurun -g 1 $PY -m pemwe.train --algo sac --seed "$s" \
        --override "reward.w2=$w" --run-id "$id" \
        > "logs/${id}.log" 2>&1 &
  done
done
wait
echo "SAC sweep complete: $TOTAL runs."
