#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source configs/env.sh
mkdir -p "$EVAL_ROOT"
TASK_IDS=${TASK_IDS:-"[0]"}
EPISODES=${EPISODES:-3}
OUTPUT_DIR=${OUTPUT_DIR:-"$EVAL_ROOT/pi05_spatial_task0"}
POLICY_PATH=${POLICY_PATH:-"lerobot/pi05_libero_finetuned"}
lerobot-eval \
  --output_dir="$OUTPUT_DIR" \
  --policy.path="$POLICY_PATH" \
  --policy.n_action_steps=10 \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.task_ids="$TASK_IDS" \
  --eval.batch_size=1 \
  --eval.n_episodes="$EPISODES" \
  --eval.recording=true \
  --env.max_parallel_tasks=1
