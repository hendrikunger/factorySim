#!/bin/bash
set -e

INSTANCES=("ray-head" "ray-worker")

for INSTANCE in "${INSTANCES[@]}"; do
    if apptainer instance list | grep -q "$INSTANCE"; then
        echo "Stopping Ray inside instance: $INSTANCE"
        apptainer exec instance://$INSTANCE ray stop || true

        echo "Waiting for Ray to clean up..."
        sleep 10  # You can adjust this up to 10–15s if needed

        echo "Stopping Apptainer instance: $INSTANCE"
        apptainer instance stop "$INSTANCE"
    else
        echo "Instance $INSTANCE is not running."
    fi
done