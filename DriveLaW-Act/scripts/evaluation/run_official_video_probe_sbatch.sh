#!/bin/bash
#SBATCH -J official_video
#SBATCH -p gpu_h200,gpu_h100,gpu_a800
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 02:00:00
#SBATCH -o /data/run01/scyb955/sunyuhan/projects/SD/DriveLaW-Act/official_video.%j.out
#SBATCH -e /data/run01/scyb955/sunyuhan/projects/SD/DriveLaW-Act/official_video.%j.err

set -euo pipefail

source /data/apps/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate drivelaw_video

PROJECT_ROOT=/data/run01/scyb955/sunyuhan/projects/SD
VIDEO_CODE_ROOT="${PROJECT_ROOT}/DriveLaW-Video/Infer"
MODEL_ROOT=/data/home/scyb955/run/sunyuhan/projects/models/DriveLaW_weights/DriveLaW-Video
PROBE_ROOT=/data/home/scyb955/run/sunyuhan/projects/exp/joint_video_trajectory_probe_2s_h200_navtrain

TOKEN="${TOKEN:-0000be0b1dc65be3}"
INPUT_DIR="${PROBE_ROOT}/input_frames/${TOKEN}"
OUTPUT_ROOT="${PROBE_ROOT}/official_video_comparison/${TOKEN}"
CONDITION_VIDEO="${OUTPUT_ROOT}/condition_9frames_8fps.mp4"
CONDITION_FRAMES_DIR="${OUTPUT_ROOT}/condition_frames"
FULL_OUTPUT_VIDEO="${OUTPUT_ROOT}/official_drivelaw_video_full_33frames.mp4"
OUTPUT_VIDEO="${OUTPUT_ROOT}/official_drivelaw_video_future_24frames.mp4"

if [[ ! -f "${MODEL_ROOT}/model_index.json" ]]; then
  echo "DriveLaW-Video model_index.json not found: ${MODEL_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "NAVSIM input-frame directory not found: ${INPUT_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

python "${VIDEO_CODE_ROOT}/prepare_navsim_condition_video.py" \
  --input-dir "${INPUT_DIR}" \
  --output-video "${CONDITION_VIDEO}" \
  --output-frames-dir "${CONDITION_FRAMES_DIR}" \
  --fps 8

cd "${VIDEO_CODE_ROOT}"

python infer.py \
  --condition_video "${CONDITION_VIDEO}" \
  --prompt "A high-quality, photorealistic dashboard camera view of autonomous driving. Based on the past one second, predict and generate a realistic driving continuation. Maintain temporal consistency, stable camera perspective, natural motion flow without jitter or artifacts, clear details, and realistic physics." \
  --output_path "${FULL_OUTPUT_VIDEO}" \
  --model_path "${MODEL_ROOT}" \
  --height 704 \
  --width 1280 \
  --num_frames 33 \
  --condition_frames 0 \
  --frame_rate 8 \
  --seed 42 \
  --num_inference_steps 30 \
  --device cuda \
  --noise_reinjection_enabled \
  --noise_reinjection_beta 1.0 \
  --noise_reinjection_sigma 0.3 \
  --noise_reinjection_steps 3

ffmpeg -y -loglevel error \
  -i "${FULL_OUTPUT_VIDEO}" \
  -vf "select='gte(n,9)',setpts=N/(8*TB)" \
  -an \
  -c:v mpeg4 \
  -q:v 2 \
  -pix_fmt yuv420p \
  "${OUTPUT_VIDEO}"

echo "Condition video: ${CONDITION_VIDEO}"
echo "Official full video: ${FULL_OUTPUT_VIDEO}"
echo "Official future video: ${OUTPUT_VIDEO}"
