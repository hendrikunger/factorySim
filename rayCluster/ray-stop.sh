#!/bin/bash
set -e

# List of instance names to shut down gracefully
INSTANCES=("ray-head" "ray-worker")

for INSTANCE in "${INSTANCES[@]}"; do
    if apptainer instance list | grep -q "$INSTANCE"; then
        echo "Stopping Ray inside instance: $INSTANCE"

        # Gracefully shut down Ray (ignore errors if Ray isn't running)
        apptainer exec instance://$INSTANCE ray stop || true

        echo "Stopping Apptainer instance: $INSTANCE"
        apptainer instance stop "$INSTANCE"
    else
        echo "Instance $INSTANCE is not running."
    fi
done