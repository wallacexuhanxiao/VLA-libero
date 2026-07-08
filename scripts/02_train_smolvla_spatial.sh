#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source configs/env.sh
mkdir -p "$OUTPUT_ROOT"

# 5090-friendly baseline. If OOM, reduce --batch_size to 2.
# Important: use --policy.path to fine-tune the pretrained SmolVLA policy,
# including its pretrained action expert, rather than initializing a fresh policy.
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --env.type=libero \
  --env.task=libero_spatial \
  --output_dir="$OUTPUT_ROOT/smolvla_libero_spatial" \
  --steps=20000 \
  --batch_size=4 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --env_eval_freq=1000
