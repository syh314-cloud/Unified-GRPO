#!/usr/bin/env bash
# Wait for usable shared-GPU memory, then run Stage-3 training.
# OOM exits return to the wait loop; all other failures stop immediately.
set -euo pipefail

OUTPUT_ROOT=${OUTPUT_ROOT:?Set OUTPUT_ROOT for checkpoints and logs.}
RESOURCE_GPU=${RESOURCE_GPU:-4}
MIN_FREE_MIB=${MIN_FREE_MIB:-80000}
CHECK_INTERVAL_SECONDS=${CHECK_INTERVAL_SECONDS:-60}
STABLE_CHECKS=${STABLE_CHECKS:-3}

TRAIN_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_stage3_joint_video_action.sh"
mkdir -p "$OUTPUT_ROOT/attempt_logs"

wait_for_memory() {
  local ready=0

  while true; do
    echo "===== GPU memory check: $(date) ====="
    nvidia-smi --query-gpu=index,memory.free,memory.total,utilization.gpu \
      --format=csv,noheader,nounits

    if nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | \
        awk -v required="$RESOURCE_GPU" -v minimum="$MIN_FREE_MIB" '
          NR > required { extra=1 }
          {
            gsub(/ /, "", $1)
            if ($1 < minimum) insufficient=1
            count += 1
          }
          END { exit (count == required && !extra && !insufficient) ? 0 : 1 }
        '
    then
      ready=$((ready + 1))
      echo "Memory condition met: ${ready}/${STABLE_CHECKS}"
      if [ "$ready" -ge "$STABLE_CHECKS" ]; then
        return
      fi
    else
      ready=0
      echo "Need ${RESOURCE_GPU} visible GPUs with at least ${MIN_FREE_MIB} MiB free each."
    fi

    sleep "$CHECK_INTERVAL_SECONDS"
  done
}

attempt=0
while true; do
  wait_for_memory

  attempt=$((attempt + 1))
  attempt_log="$OUTPUT_ROOT/attempt_logs/attempt_$(printf '%03d' "$attempt").log"
  echo "===== Starting attempt ${attempt}: $(date) ====="

  set +e
  bash "$TRAIN_SCRIPT" 2>&1 | tee "$attempt_log"
  status=${PIPESTATUS[0]}
  set -e

  if [ "$status" -eq 0 ]; then
    echo "Training completed successfully."
    exit 0
  fi

  if grep -qE 'torch\.OutOfMemoryError|CUDA out of memory|out of memory' "$attempt_log"; then
    echo "Attempt ${attempt} ended with CUDA OOM; returning to memory wait loop."
    continue
  fi

  echo "Attempt ${attempt} failed without CUDA OOM; stopping. See: $attempt_log" >&2
  exit "$status"
done
