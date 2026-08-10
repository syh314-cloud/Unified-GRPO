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
BUNDLED_DIFFUSERS="${INFER_ROOT}/diffusers/src/diffusers"

# These are the exact values in DriveLaW-Video/Infer/README.md.
NUM_FRAMES=33
CONDITION_FRAMES=9
FRAME_RATE=8
NUM_INFERENCE_STEPS=30

count_video_frames() {
  local video_path=$1
  (cd /tmp && python - "${video_path}" <<'PY'
import sys
from diffusers.utils import load_video

print(len(load_video(sys.argv[1])))
PY
  )
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

if [[ -d "${BUNDLED_DIFFUSERS}/models" ]]; then
  echo "Bundled diffusers models directory unexpectedly exists; remove the compatibility setup." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
INPUT_FRAME_COUNT=$(count_video_frames "${CONDITION_VIDEO}")

readarray -t INSTALLED_DIFFUSERS_INFO < <(
  cd /tmp
  python - <<'PY'
import os
import diffusers

print(diffusers.__version__)
print(os.path.dirname(diffusers.__file__))
PY
)
INSTALLED_DIFFUSERS_VERSION=${INSTALLED_DIFFUSERS_INFO[0]}
INSTALLED_DIFFUSERS_ROOT=${INSTALLED_DIFFUSERS_INFO[1]}
INSTALLED_MODELS="${INSTALLED_DIFFUSERS_ROOT}/models"

if [[ ! -d "${INSTALLED_MODELS}" ]]; then
  echo "Installed diffusers models directory not found: ${INSTALLED_MODELS}" >&2
  exit 1
fi

BUNDLED_DIFFUSERS_VERSION=$(python - "${BUNDLED_DIFFUSERS}/__init__.py" <<'PY'
import ast
import sys

tree = ast.parse(open(sys.argv[1], "r", encoding="utf-8").read())
for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__version__":
                print(ast.literal_eval(node.value))
                raise SystemExit
raise RuntimeError("Bundled diffusers __version__ was not found")
PY
)

# The official repository accidentally excludes src/diffusers/models via its
# root models/ gitignore rule. Build a temporary package that keeps every
# bundled source file unchanged and supplies only that missing directory from
# the complete diffusers installation in this environment.
RUNTIME_ROOT=$(mktemp -d "/tmp/drivelaw-video-${SLURM_JOB_ID:-manual}.XXXXXX")
trap 'rm -rf "${RUNTIME_ROOT}"' EXIT
mkdir -p "${RUNTIME_ROOT}/diffusers/src/diffusers"
ln -s "${INFER_ROOT}/infer.py" "${RUNTIME_ROOT}/infer.py"

for source_path in "${BUNDLED_DIFFUSERS}"/*; do
  entry_name=$(basename "${source_path}")
  ln -s "${source_path}" "${RUNTIME_ROOT}/diffusers/src/diffusers/${entry_name}"
done
ln -s "${INSTALLED_MODELS}" "${RUNTIME_ROOT}/diffusers/src/diffusers/models"

python - "${RUNTIME_ROOT}" <<'PY'
import sys
from pathlib import Path

runtime_root = Path(sys.argv[1])
sys.path.insert(0, str(runtime_root / "diffusers" / "src"))
from diffusers import LTXConditionPipeline
from diffusers.models import AutoencoderKLLTXVideo, LTXVideoTransformer3DModel

print("Temporary official pipeline import: OK")
print("Pipeline:", LTXConditionPipeline)
print("VAE:", AutoencoderKLLTXVideo)
print("Transformer:", LTXVideoTransformer3DModel)
PY

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
Input frames measured by official diffusers load_video: ${INPUT_FRAME_COUNT}
Pipeline requested pixel frames before trimming: ${NUM_FRAMES}
VAE temporal compression ratio: ${TEMPORAL_RATIO}
Bundled diffusers version: ${BUNDLED_DIFFUSERS_VERSION}
Installed diffusers models version: ${INSTALLED_DIFFUSERS_VERSION}
Main latent temporal frames: ${MAIN_LATENT_FRAMES}
Condition latent temporal frames: ${CONDITION_LATENT_FRAMES}
Condition frames removed by official infer.py: ${CONDITION_FRAMES}
Expected saved future frames: $((NUM_FRAMES - CONDITION_FRAMES))
EOF

cd "${RUNTIME_ROOT}"

# Keep this invocation aligned with the minimal command in Infer/README.md.
python "${RUNTIME_ROOT}/infer.py" \
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
Output frames measured by official diffusers load_video: ${OUTPUT_FRAME_COUNT}
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
