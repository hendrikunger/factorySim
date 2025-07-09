#!/bin/bash
set -e

# Path to your Apptainer image
IMAGE_PATH="/home/unhe/factorySim/factorySim.sif"

# IP address of the head node (replace this with the actual IP)
HEAD_IP="10.54.129.113"
PORT=6379

apptainer exec "$IMAGE_PATH" ray start --address="${HEAD_IP}:${PORT}"