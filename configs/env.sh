#!/usr/bin/env bash

# Use EGL on headless AutoDL/cloud servers.
export MUJOCO_GL=${MUJOCO_GL:-egl}

# Project paths.
export PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
export LEROBOT_DIR=${LEROBOT_DIR:-$PROJECT_ROOT/third_party/lerobot}

# Change this if you want to push checkpoints to a different HF namespace.
export HF_USER=${HF_USER:-wallacexuhanxiao}

# Default output locations.
export OUTPUT_ROOT=${OUTPUT_ROOT:-$PROJECT_ROOT/outputs}
export EVAL_ROOT=${EVAL_ROOT:-$PROJECT_ROOT/eval_logs}
