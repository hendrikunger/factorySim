#!/bin/bash
set -e

ray job submit \
  --runtime-env-json "{\"env_vars\": {\"WANDB_API_KEY\": \"$WANDB_API_KEY\"}}" \
  -- python train.py