#!/bin/bash
set -e

IMAGE_PATH="/home/unhe/factorySim/factorySim.sif"
INSTANCE_NAME="ray-worker"
HEAD_NODE_IP="${RAY_HEAD_IP}"

# Start the Apptainer instance
apptainer instance start --nv --writable-tmpfs "$IMAGE_PATH" "$INSTANCE_NAME"


# Start Ray in the foreground (blocking)
apptainer exec \
    --env NCCL_P2P_DISABLE=1 \
    --env NCCL_DEBUG=INFO \
    --env NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo.xml \
    --env CUDA_VISIBLE_DEVICES=0,1 \instance://$INSTANCE_NAME ray start \
    --address="${HEAD_NODE_IP}:6379" \
    --metrics-export-port=8266\
    --block