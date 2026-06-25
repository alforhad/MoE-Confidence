#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# =============================================================================
# Edit paths here (leave empty for X-TAIL defaults under this directory)
# =============================================================================
DATA_ROOT=""
CHECKPOINT_ROOT=""
ALL_CLASSES=""
EVAL_LOG=""               # (default: ${ROOT}/output_eval_clean.txt)

GPU=""
NUM_TASKS=""
# =============================================================================

XTAIL_DATA_ROOT="${DATA_ROOT:-${XTAIL_DATA_ROOT:-${ROOT}/data}}"
XTAIL_CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${XTAIL_CHECKPOINT_ROOT:-${ROOT}/checkpoints}}"
export XTAIL_DATA_ROOT XTAIL_CHECKPOINT_ROOT

if [[ -n "${ALL_CLASSES}" ]]; then
  export XTAIL_ALL_CLASSES="${ALL_CLASSES}"
fi

export XTAIL_EVAL_LOG="${EVAL_LOG:-${XTAIL_EVAL_LOG:-${ROOT}/output_eval_clean.txt}}"

: > "${XTAIL_EVAL_LOG}"

GPU="${GPU:-0}"
NUM_TASKS="${NUM_TASKS:-11}"

DATASETS=(
  Aircraft Caltech101 CIFAR100 DTD EuroSAT Flowers Food MNIST
  OxfordPet StanfordCars SUN397
)

for ((learned = 0; learned < NUM_TASKS; learned++)); do
  for dataset_idx in "${!DATASETS[@]}"; do
    dataset="${DATASETS[dataset_idx]}"
    CUDA_VISIBLE_DEVICES="${GPU}" python3 -m src.main_eval --eval-only \
      --eval-datasets="${dataset}" \
      --data-location="${XTAIL_DATA_ROOT}" \
      --checkpoint-dir="${XTAIL_CHECKPOINT_ROOT}" \
      --experts_num "${dataset_idx}" \
      --tid "${learned}"
  done
done

python3 -m src.xtail_accuracy --log-path "${XTAIL_EVAL_LOG}" --num-tasks "${NUM_TASKS}"
