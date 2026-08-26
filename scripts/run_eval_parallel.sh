#!/usr/bin/env bash
# Evaluate every checkpoint in models/ on the held-out split, N at a time.
#
#   ./scripts/run_eval_parallel.sh [concurrency]
#
# `python -m pemwe.evaluate --sweep` is single-threaded and does one model at a time:
# 72 held-out days x 1440 steps is ~103,000 policy forward passes per checkpoint, so a
# 65-run matrix takes the better part of an hour while 120 cores sit idle. Evaluation is
# embarrassingly parallel across checkpoints -- there is no shared state, each writes its
# own results file.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

CONC="${1:-8}"
PY=.venv/bin/python
[ -x "$PY" ] || PY=./.venv/Scripts/python.exe
mkdir -p logs results

# one thread each: many small independent jobs, not a few big ones
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

shopt -s nullglob
models=(models/*.zip)
# never evaluate scratch runs as if they were paper data
keep=()
for m in "${models[@]}"; do
  b="$(basename "$m" .zip)"
  case "$b" in pipe_*|verif_*|smoke_*|rate_*) continue ;; esac
  keep+=("$m")
done

echo "evaluating ${#keep[@]} checkpoints, $CONC at a time"
[ "${#keep[@]}" -eq 0 ] && { echo "nothing to evaluate"; exit 0; }

start=$(date +%s)
for m in "${keep[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do wait -n; done
  id="$(basename "$m" .zip)"
  algo="ppo"; case "$id" in sac_*) algo="sac" ;; esac
  $PY -m pemwe.evaluate --model "$m" --policy "$algo" --run-id "$id" --split test \
      > "logs/eval_${id}.log" 2>&1 &
done
wait

n=$(ls results/*_seed*.json 2>/dev/null | wc -l)
echo "done: $n result files in $(( ($(date +%s) - start) / 60 )) min"
