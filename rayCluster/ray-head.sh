#!/bin/bash
set -e

# Path to your Apptainer image
IMAGE_PATH="/home/unhe/factorySim/factorySim.sif"
INSTANCE_NAME="ray-head"

# Ray startup parameters
PORT=6379
DASHBOARD_PORT=8265

# Start the instance (does nothing until script inside is called)
apptainer instance start --writable-tmpfs "$IMAGE_PATH" "$INSTANCE_NAME"


# Start Ray head
apptainer exec instance://$INSTANCE_NAME  ray start \
    --head \
    --port=6379 \
    --dashboard-host=0.0.0.0