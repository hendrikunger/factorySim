#!/bin/bash
set -e

IMAGE_PATH="/home/unhe/factorySim/factorySim.sif"
INSTANCE_NAME="ray-head"
PORT=6379

# Start Apptainer instance
apptainer instance start --writable-tmpfs "$IMAGE_PATH" "$INSTANCE_NAME"

# Let ray handle shutdown via --block (it handles SIGTERM)
apptainer exec instance://$INSTANCE_NAME ray start \
    --head \
    --port=$PORT \
    --dashboard-host=0.0.0.0 \
    --metrics-export-port=10001\
    --block