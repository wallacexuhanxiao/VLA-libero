#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source configs/env.sh
mkdir -p "$OUTPUT_ROOT"

# Structure-innovation baseline:
# - load pretrained VLM weights
# - initialize a fresh vanilla action expert from config
# - train the action expert on official LIBERO data/protocol
# If OOM on a single 5090, reduce --batch_size to 2.
lerobot-train \
  --policy.type=smolvla \
  --policy.repo_id="$HF_USER/smolvla-libero-spatial-scratch-expert" \
  --policy.load_vlm_weights=true \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --env.type=libero \
  --env.task=libero_spatial \
  --output_dir="$OUTPUT_ROOT/smolvla_libero_spatial_scratch_expert" \
  --steps=20000 \
  --batch_size=4 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --env_eval_freq=1000
