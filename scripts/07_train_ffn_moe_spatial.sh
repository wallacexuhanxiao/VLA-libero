#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source configs/env.sh
mkdir -p "$OUTPUT_ROOT"
STEPS=${STEPS:-20000}
BATCH_SIZE=${BATCH_SIZE:-16}
EVAL_EPISODES=${EVAL_EPISODES:-1}
ENV_EVAL_FREQ=${ENV_EVAL_FREQ:-5000}
SAVE_FREQ=${SAVE_FREQ:-5000}
OUTPUT_DIR=${OUTPUT_DIR:-"$OUTPUT_ROOT/smolvla_libero_spatial_ffn_moe"}

lerobot-train \
  --policy.type=smolvla \
  --policy.repo_id="$HF_USER/smolvla-libero-spatial-ffn-moe" \
  --policy.load_vlm_weights=true \
  --policy.use_action_ffn_moe=true \
  --policy.action_ffn_moe_num_experts=4 \
  --policy.action_ffn_moe_aux_loss_coef=0.01 \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.root=/root/autodl-tmp/datasets/HuggingFaceVLA_libero \
  --dataset.streaming=true \
  --env.type=libero \
  --env.task=libero_spatial \
  --output_dir="$OUTPUT_DIR" \
  --steps="$STEPS" \
  --batch_size="$BATCH_SIZE" \
  --eval.batch_size=1 \
  --eval.n_episodes="$EVAL_EPISODES" \
  --env_eval_freq="$ENV_EVAL_FREQ" \
  --save_freq="$SAVE_FREQ" \
  --policy.push_to_hub=false
