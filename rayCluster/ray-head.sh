#!/bin/bash
set -e

IMAGE_PATH="/home/unhe/factorySim/factorySim.sif"
INSTANCE_NAME="ray-head"
PORT=6379
DASHBOARD_PORT=8265

# Start the Apptainer instance
apptainer instance start --writable-tmpfs "$IMAGE_PATH" "$INSTANCE_NAME"

# Graceful shutdown trap
cleanup() {
    echo "Received SIGTERM. Shutting down Ray gracefully..."
    apptainer exec instance://$INSTANCE_NAME ray stop || true
    apptainer instance stop "$INSTANCE_NAME" || true
    exit 0
}
trap cleanup SIGTERM SIGINT

# Start Ray in the foreground — this blocks
apptainer exec instance://$INSTANCE_NAME ray start \
    --head \
    --port=$PORT \
    --dashboard-host=0.0.0.0 \
    --block