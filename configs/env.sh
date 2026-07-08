#!/usr/bin/env bash

# Reuse the image-provided conda/PyTorch stack; do not redownload CUDA/Torch.
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}

export PROJECT_ROOT=${PROJECT_ROOT:-/root/VLA-libero}
export LEROBOT_DIR=${LEROBOT_DIR:-$PROJECT_ROOT/third_party/lerobot}
export HF_USER=${HF_USER:-wallacexuhanxiao}

# Keep heavy files on the data disk.
export OUTPUT_ROOT=${OUTPUT_ROOT:-/root/autodl-tmp/outputs/VLA-libero}
export EVAL_ROOT=${EVAL_ROOT:-/root/autodl-tmp/eval_logs/VLA-libero}
export HF_HOME=${HF_HOME:-/root/autodl-tmp/cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/root/autodl-tmp/cache/huggingface/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/root/autodl-tmp/cache/huggingface/transformers}
export DATASETS_CACHE=${DATASETS_CACHE:-/root/autodl-tmp/cache/huggingface/datasets}
export TORCH_HOME=${TORCH_HOME:-/root/autodl-tmp/cache/torch}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/root/autodl-tmp/cache/pip}
export TMPDIR=${TMPDIR:-/root/autodl-tmp/cache/tmp}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-/root/autodl-tmp/datasets}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/root/autodl-tmp/cache/huggingface/datasets}
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" "$TMPDIR" "$OUTPUT_ROOT" "$EVAL_ROOT"
