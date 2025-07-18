#!/bin/bash
set -e

IMAGE_PATH="/home/unhe/factorySim/factorySim.sif"
INSTANCE_NAME="ray-worker"
HEAD_NODE_IP="10.54.129.113"


# Start the Apptainer instance
apptainer instance start  --nv --writable-tmpfs "$IMAGE_PATH" "$INSTANCE_NAME"


# Start Ray in the foreground (blocking)

apptainer exec \
#Disable P2P Conenction between GPUs for NCCL, as it is not supported in the cluster.
    --env NCCL_P2P_DISABLE=1 \
    --env CUDA_VISIBLE_DEVICES=0,1 \
    --env NCCL_P2P_LEVEL=LOC \
# Disable shared memory for NCCL, as it is not supported in the cluster. Would need P2P to be enabled.
    --env NCCL_SHM_DISABLE=1 \
    instance://$INSTANCE_NAME ray start \
        --address="${HEAD_NODE_IP}:6379" \
        --metrics-export-port=8266 \
        --block