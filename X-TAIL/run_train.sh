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

GPU=""
ITERATIONS=""
# =============================================================================

XTAIL_DATA_ROOT="${DATA_ROOT:-${XTAIL_DATA_ROOT:-${ROOT}/data}}"
XTAIL_CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${XTAIL_CHECKPOINT_ROOT:-${ROOT}/checkpoints}}"
export XTAIL_DATA_ROOT XTAIL_CHECKPOINT_ROOT

if [[ -n "${ALL_CLASSES}" ]]; then
  export XTAIL_ALL_CLASSES="${ALL_CLASSES}"
fi

GPU="${GPU:-0}"
ITERATIONS="${ITERATIONS:-1000}"

DATASETS=(
  Aircraft Caltech101 CIFAR100 DTD EuroSAT Flowers Food MNIST
  OxfordPet StanfordCars SUN397
)
LR=(5e-3 1e-3 5e-3 1e-3 1e-4 1e-3 1e-3 1e-4 1e-3 1e-3 1e-3)

for ((i=0; i<${#DATASETS[@]}; i++)); do
  dataset="${DATASETS[i]}"
  echo "=== Training task ${i}: ${dataset} ==="
  CUDA_VISIBLE_DEVICES="${GPU}" python3 -m src.main_train \
    --train-dataset="${dataset}" \
    --data-location="${XTAIL_DATA_ROOT}" \
    --checkpoint-dir="${XTAIL_CHECKPOINT_ROOT}" \
    --lr="${LR[i]}" \
    --ls 0.2 \
    --iterations "${ITERATIONS}" \
    --tid "${i}"
done
