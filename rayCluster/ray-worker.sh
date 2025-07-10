#!/bin/bash
set -e

IMAGE_PATH="/home/unhe/factorySim/factorySim.sif"
INSTANCE_NAME="ray-worker"
HEAD_NODE_IP="${RAY_HEAD_IP}"

# Start the Apptainer instance
apptainer instance start --writable-tmpfs "$IMAGE_PATH" "$INSTANCE_NAME"

# Graceful shutdown trap
cleanup() {
    echo "Received SIGTERM. Shutting down Ray worker gracefully..."
    apptainer exec instance://$INSTANCE_NAME ray stop || true
    apptainer instance stop "$INSTANCE_NAME" || true
    exit 0
}
trap cleanup SIGTERM SIGINT

# Start Ray in the foreground (blocking)
apptainer exec instance://$INSTANCE_NAME ray start \
    --address="${HEAD_NODE_IP}:6379" \
    --block