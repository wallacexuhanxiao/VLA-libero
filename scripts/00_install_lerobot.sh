#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p third_party
if [ ! -d third_party/lerobot/.git ]; then
  git clone https://github.com/huggingface/lerobot.git third_party/lerobot
fi

cd third_party/lerobot

python -m pip install --upgrade pip setuptools wheel
pip install -e ".[smolvla]"
pip install -e ".[libero]"

python - <<'PY'
import os
print("MUJOCO_GL =", os.environ.get("MUJOCO_GL", "<not set>"))
import lerobot
print("lerobot import: OK")
PY
