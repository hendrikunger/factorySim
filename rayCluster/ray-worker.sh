#!/bin/bash
set -e

# Path to Apptainer image
IMAGE_PATH="/home/unhe/factorysim/factorySim.sif"

# Head node IP (override via env or systemd)
HEAD_NODE_IP=${RAY_HEAD_IP:-"10.54.129.113"}
LOG_FILE="/var/log/ray-worker.log"

# Retry settings
MAX_RETRIES=10
RETRY_DELAY=5  # seconds

echo "[$(date)] Starting Ray worker. Head: ${HEAD_NODE_IP}" >> "$LOG_FILE"

for i in $(seq 1 $MAX_RETRIES); do
    echo "[$(date)] Attempt $i to start worker..." >> "$LOG_FILE"

    if apptainer exec "$IMAGE_PATH" ray start --address="${HEAD_NODE_IP}:6379" >> "$LOG_FILE" 2>&1; then
        echo "[$(date)] Worker started successfully." >> "$LOG_FILE"
        exit 0
    else
        echo "[$(date)] Worker start failed. Retrying in $RETRY_DELAY seconds..." >> "$LOG_FILE"
        sleep $RETRY_DELAY
    fi
done

echo "[$(date)] Failed to start worker after $MAX_RETRIES attempts." >> "$LOG_FILE"
exit 1