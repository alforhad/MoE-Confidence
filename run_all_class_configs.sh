#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

TORCH_NPROC="${TORCH_NPROC:-1}"
if [[ -n "${PYTHON_CMD:-}" ]]; then
    PYTHON_CMD=(${PYTHON_CMD})
elif [[ "${TORCH_NPROC}" -gt 1 ]]; then
    PYTHON_CMD=(conda run --no-capture-output -n MoE_Confidence torchrun --standalone --nproc_per_node="${TORCH_NPROC}")
else
    PYTHON_CMD=(conda run --no-capture-output -n MoE_Confidence python)
fi

ENTRYPOINT="${ENTRYPOINT:-main_ddp.py}"
EXTRA_ARGS=("$@")

run_config() {
    local config_name="$1"
    local class_order="$2"

    echo "Running ${config_name}"
    "${PYTHON_CMD[@]}" "${ENTRYPOINT}" \
        --config-path configs/class \
        --config-name "${config_name}" \
        class_order="${class_order}" \
        "${EXTRA_ARGS[@]}"
}

run_config "cifar100_2-2.yaml" "class_orders/cifar100.yaml"
run_config "cifar100_5-5.yaml" "class_orders/cifar100.yaml"
run_config "cifar100_10-10.yaml" "class_orders/cifar100.yaml"

run_config "imagenet_r_20-20.yaml" "class_orders/imagenet_R_order.yaml"

run_config "tinyimagenet_100-5.yaml" "class_orders/tinyimagenet.yaml"
run_config "tinyimagenet_100-10.yaml" "class_orders/tinyimagenet.yaml"
run_config "tinyimagenet_100-20.yaml" "class_orders/tinyimagenet.yaml"


