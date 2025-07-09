#!/bin/bash
set -e

# Path to your Apptainer image
IMAGE_PATH="/home/unhe/factorysim/factorySim.sif"
LOG_FILE="/var/log/ray-head.log"

# Ray startup parameters
PORT=6379
DASHBOARD_PORT=8265

# Start Ray head
apptainer exec "$IMAGE_PATH" ray start \
    --head \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    >> "$LOG_FILE" 2>&1