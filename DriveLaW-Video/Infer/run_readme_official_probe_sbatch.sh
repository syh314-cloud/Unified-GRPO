#!/bin/bash
#SBATCH -J video_readme
#SBATCH -p gpu_h200,gpu_h100,gpu_a800
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 02:00:00
#SBATCH -o /data/run01/scyb955/sunyuhan/projects/SD/DriveLaW-Video/Infer/video_readme.%j.out
#SBATCH -e /data/run01/scyb955/sunyuhan/projects/SD/DriveLaW-Video/Infer/video_readme.%j.err

set -euo pipefail

source /data/apps/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate drivelaw_video

INFER_ROOT=/data/run01/scyb955/sunyuhan/projects/SD/DriveLaW-Video/Infer
MODEL_ROOT=${MODEL_ROOT:-/data/home/scyb955/run/sunyuhan/projects/models/DriveLaW_weights/DriveLaW-Video}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/home/scyb955/run/sunyuhan/projects/exp/drivelaw_video_readme_official}

CONDITION_VIDEO="${INFER_ROOT}/demo/condition_videos/scene_0001_conditioning.mp4"
PROMPT="${INFER_ROOT}/demo/prompts/scene_0001_prompt.txt"
OUTPUT_VIDEO="${OUTPUT_ROOT}/scene_0001_future_24frames.mp4"
STATS_FILE="${OUTPUT_ROOT}/frame_statistics.txt"

# These are the exact values in DriveLaW-Video/Infer/README.md.
NUM_FRAMES=33
CONDITION_FRAMES=9
FRAME_RATE=8
NUM_INFERENCE_STEPS=30

count_video_frames() {
  local video_path=$1
  local count
  count=$(ffprobe -v error -count_frames -select_streams v:0 \
    -show_entries stream=nb_read_frames \
    -of default=noprint_wrappers=1:nokey=1 "${video_path}")
  if [[ -z "${count}" || "${count}" == "N/A" ]]; then
    count=$(ffprobe -v error -select_streams v:0 \
      -show_entries stream=nb_frames \
      -of default=noprint_wrappers=1:nokey=1 "${video_path}")
  fi
  printf '%s\n' "${count}"
}

for required_path in \
  "${CONDITION_VIDEO}" \
  "${PROMPT}" \
  "${MODEL_ROOT}/model_index.json" \
  "${MODEL_ROOT}/vae/config.json"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Required official inference input is missing: ${required_path}" >&2
    exit 1
  fi
done

command -v ffprobe >/dev/null || {
  echo "ffprobe is required to count frames" >&2
  exit 1
}

mkdir -p "${OUTPUT_ROOT}"
INPUT_FRAME_COUNT=$(count_video_frames "${CONDITION_VIDEO}")

TEMPORAL_RATIO=$(python - "${MODEL_ROOT}/vae/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    config = json.load(handle)
print(config["temporal_compression_ratio"])
PY
)

MAIN_LATENT_FRAMES=$(( (NUM_FRAMES - 1) / TEMPORAL_RATIO + 1 ))
CONDITION_LATENT_FRAMES=$(( (CONDITION_FRAMES - 1) / TEMPORAL_RATIO + 1 ))

cat >"${STATS_FILE}" <<EOF
DriveLaW-Video official README inference frame statistics
=========================================================
Condition video file: ${CONDITION_VIDEO}
Input frames measured by ffprobe: ${INPUT_FRAME_COUNT}
Pipeline requested pixel frames before trimming: ${NUM_FRAMES}
VAE temporal compression ratio: ${TEMPORAL_RATIO}
Main latent temporal frames: ${MAIN_LATENT_FRAMES}
Condition latent temporal frames: ${CONDITION_LATENT_FRAMES}
Condition frames removed by official infer.py: ${CONDITION_FRAMES}
Expected saved future frames: $((NUM_FRAMES - CONDITION_FRAMES))
EOF

cd "${INFER_ROOT}"

# Keep this invocation aligned with the minimal command in Infer/README.md.
python infer.py \
  --condition_video "${CONDITION_VIDEO}" \
  --prompt "${PROMPT}" \
  --output_path "${OUTPUT_VIDEO}" \
  --model_path "${MODEL_ROOT}" \
  --height 704 \
  --width 1280 \
  --num_frames "${NUM_FRAMES}" \
  --condition_frames "${CONDITION_FRAMES}" \
  --frame_rate "${FRAME_RATE}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --device cuda

OUTPUT_FRAME_COUNT=$(count_video_frames "${OUTPUT_VIDEO}")
cat >>"${STATS_FILE}" <<EOF
Output frames measured by ffprobe: ${OUTPUT_FRAME_COUNT}
Output video: ${OUTPUT_VIDEO}
EOF

echo
cat "${STATS_FILE}"

if [[ "${INPUT_FRAME_COUNT}" != "${CONDITION_FRAMES}" ]]; then
  echo "Input frame count differs from README condition_frames=${CONDITION_FRAMES}" >&2
  exit 1
fi
if [[ "${OUTPUT_FRAME_COUNT}" != "$((NUM_FRAMES - CONDITION_FRAMES))" ]]; then
  echo "Output frame count differs from README expectation" >&2
  exit 1
fi
