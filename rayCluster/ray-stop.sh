#!/bin/bash
set -e

# List of instance names to stop
INSTANCES=("ray-head" "ray-worker")

for INSTANCE in "${INSTANCES[@]}"; do
    if apptainer instance list | grep -q "$INSTANCE"; then
        echo "Stopping Apptainer instance: $INSTANCE"
        apptainer instance stop "$INSTANCE"
    else
        echo "Instance $INSTANCE is not running."
    fi
done