#!/usr/bin/env bash
# Release every GPU this PROJECT is holding, and nothing else.
#
#   ./scripts/release_gpus.sh          # report + release
#   ./scripts/release_gpus.sh --check  # report only
#
# SCOPED ON PURPOSE. This account also runs unrelated work (an LLM server, other
# training), and the box is shared with ~30 other users. Matching on "gpurun" alone would
# kill a colleague's job or the account owner's own. Every pattern here contains `pemwe`,
# so only this project's processes can ever be signalled.
#
# Written because a stray `gpurun nvidia-smi` of mine sat holding a GPU for 11 minutes,
# and a driver that died on SSH teardown left four preempted jobs behind it.
set -uo pipefail

PATTERNS=("pemwe.train" "pemwe.evaluate" "pemwe-rl/scripts" "pipeline_check.py"
          "calibrate_degradation.py" "longhorizon_rollout.py" "reward_landscape.py")

check_only=0
[ "${1:-}" = "--check" ] && check_only=1

echo "GPUs held by this project:"
found=0
for p in "${PATTERNS[@]}"; do
  while read -r pid args; do
    [ -z "${pid:-}" ] && continue
    echo "  pid $pid  ${args:0:88}"
    found=$((found + 1))
  done < <(pgrep -fa "$p" 2>/dev/null | grep -i gpurun || true)
done
[ "$found" -eq 0 ] && echo "  (none)"

if [ "$check_only" -eq 1 ]; then
  echo
  gpurun --status 2>/dev/null | head -8
  exit 0
fi

if [ "$found" -gt 0 ]; then
  echo "releasing..."
  for p in "${PATTERNS[@]}"; do
    pkill -f "$p" 2>/dev/null || true
  done
  sleep 4
  for p in "${PATTERNS[@]}"; do
    pkill -9 -f "$p" 2>/dev/null || true
  done
  sleep 2
fi

echo
echo "after release:"
gpurun --status 2>/dev/null | head -8
echo
still=$(pgrep -fa "pemwe" 2>/dev/null | grep -ci gpurun || true)
if [ "${still:-0}" -eq 0 ]; then
  echo "OK: this project holds no GPUs."
else
  echo "WARNING: $still project process(es) still hold a GPU -- inspect before leaving."
fi
