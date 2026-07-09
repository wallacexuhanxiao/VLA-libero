#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source configs/env.sh
mkdir -p "$OUTPUT_ROOT"
STEPS=${STEPS:-20000}
BATCH_SIZE=${BATCH_SIZE:-8}
EVAL_EPISODES=${EVAL_EPISODES:-1}
ENV_EVAL_FREQ=${ENV_EVAL_FREQ:-5000}
SAVE_FREQ=${SAVE_FREQ:-5000}
OUTPUT_DIR=${OUTPUT_DIR:-"$OUTPUT_ROOT/smolvla_libero_spatial_pamae"}

# Ver4 is memory-heavier than Ver1/Ver3 because it keeps multiple action expert
# Transformer copies. Start with BATCH_SIZE=8 and tune upward only after a
# successful sanity run.
lerobot-train \
  --policy.type=smolvla \
  --policy.repo_id="$HF_USER/smolvla-libero-spatial-pamae" \
  --policy.load_vlm_weights=true \
  --policy.use_pamae_action_expert=true \
  --policy.pamae_num_experts=4 \
  --policy.pamae_top_k=2 \
  --policy.pamae_num_phases=4 \
  --policy.pamae_max_chunk_size=50 \
  --policy.pamae_phase_loss_coef=0.05 \
  --policy.pamae_routing_alignment_coef=0.05 \
  --policy.pamae_balance_loss_coef=0.01 \
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
