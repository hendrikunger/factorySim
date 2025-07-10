#!/bin/bash
set -e

# Path to Apptainer image
IMAGE_PATH="/home/unhe/factorySim/factorySim.sif"
INSTANCE_NAME="ray-worker01"

# Head node IP (override via env or systemd) start 
HEAD_NODE_IP=${RAY_HEAD_IP:-"10.54.129.113"}


# Retry settings
MAX_RETRIES=10
RETRY_DELAY=5  # seconds


# Start the instance (does nothing until script inside is called)
apptainer instance start --writable-tmpfs "$IMAGE_PATH" "$INSTANCE_NAME"

echo "[$(date)] Starting Ray worker. Head: ${HEAD_NODE_IP}"

for i in $(seq 1 $MAX_RETRIES); do
    echo "[$(date)] Attempt $i to start worker..." 

    if apptainer exec instance://$INSTANCE_NAME ray start --address="${HEAD_NODE_IP}:6379"; then
        echo "[$(date)] Worker started successfully."
        exit 0
    else
        echo "[$(date)] Worker start failed. Retrying in $RETRY_DELAY seconds..."
        sleep $RETRY_DELAY
    fi
done

echo "[$(date)] Failed to start worker after $MAX_RETRIES attempts."
exit 1