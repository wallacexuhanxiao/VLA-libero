#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source configs/env.sh

CHECKPOINT=${CHECKPOINT:-"$OUTPUT_ROOT/smolvla_libero_spatial/checkpoints/last/pretrained_model"}
TASK_IDS=${TASK_IDS:-"[0]"}
EPISODES=${EPISODES:-10}

mkdir -p "$EVAL_ROOT"

lerobot-eval \
  --output_dir="$EVAL_ROOT/smolvla_spatial_eval" \
  --policy.path="$CHECKPOINT" \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.task_ids="$TASK_IDS" \
  --eval.batch_size=1 \
  --eval.n_episodes="$EPISODES" \
  --env.max_parallel_tasks=1
