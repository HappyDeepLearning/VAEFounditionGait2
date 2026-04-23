#!/usr/bin/env bash

set -euo pipefail

GPUS="${CUDA_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}"
NPROC="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-9381}"
STAGE="${STAGE:-head}"
CFG_PATH="${CFG_PATH:-}"

if [ -z "${CFG_PATH}" ]; then
    case "${STAGE}" in
        head)
            CFG_PATH="./configs/deepgaitv2/DeepGaitV2_gait3d_last_ft_head.yaml"
            ;;
        full)
            CFG_PATH="./configs/deepgaitv2/DeepGaitV2_gait3d_last_ft_full.yaml"
            ;;
        *)
            echo "Unsupported STAGE: ${STAGE}. Use head or full, or set CFG_PATH directly."
            exit 1
            ;;
    esac
fi

echo "Using config: ${CFG_PATH}"
echo "CUDA_VISIBLE_DEVICES=${GPUS}"
echo "NPROC_PER_NODE=${NPROC}"
echo "MASTER_PORT=${MASTER_PORT}"

CUDA_VISIBLE_DEVICES="${GPUS}" python -m torch.distributed.launch \
    --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" opengait/main.py \
    --cfgs "${CFG_PATH}" \
    --phase train --log_to_file
