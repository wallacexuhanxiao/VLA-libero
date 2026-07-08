#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source configs/env.sh
mkdir -p "$EVAL_ROOT"

# Official sanity check: verify LIBERO env + official policy can run.
# Start with only task 0 and 3 episodes to debug cheaply.
lerobot-eval \
  --output_dir="$EVAL_ROOT/pi05_spatial_task0" \
  --policy.path=lerobot/pi05_libero_finetuned \
  --policy.n_action_steps=10 \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.task_ids='[0]' \
  --eval.batch_size=1 \
  --eval.n_episodes=3 \
  --env.max_parallel_tasks=1
