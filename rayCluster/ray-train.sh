#!/bin/bash
set -e

ray job submit \
  --runtime-env-json "{\"env_vars\": {\"WANDB_API_KEY\": \"$WANDB_API_KEY\"}}" \
  -- python train.py "$@"



# ray job submit \
#   --runtime-env-json "{\"env_vars\": {\"WANDB_API_KEY\": \"$WANDB_API_KEY\"}}" \
#   -- python pong_test.py --num-env-runners 100 --num-learners 2 