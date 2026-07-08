#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source configs/env.sh
mkdir -p "$OUTPUT_ROOT"

lerobot-train \
  --policy.type=smolvla \
  --policy.repo_id="$HF_USER/smolvla-libero-spatial-moe" \
  --policy.load_vlm_weights=true \
  --policy.use_action_moe_adapter=true \
  --policy.action_moe_num_experts=4 \
  --policy.action_moe_hidden_multiplier=2.0 \
  --policy.action_moe_residual_scale=1.0 \
  --policy.action_moe_aux_loss_coef=0.01 \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --env.type=libero \
  --env.task=libero_spatial \
  --output_dir="$OUTPUT_ROOT/smolvla_libero_spatial_moe" \
  --steps=20000 \
  --batch_size=4 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --env_eval_freq=1000
