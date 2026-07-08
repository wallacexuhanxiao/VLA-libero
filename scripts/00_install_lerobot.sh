#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source configs/env.sh
mkdir -p third_party
if [ ! -e third_party/lerobot ]; then
  if [ -d /root/vla_project/lerobot/.git ]; then
    ln -s /root/vla_project/lerobot third_party/lerobot
  else
    git clone https://github.com/huggingface/lerobot.git third_party/lerobot
  fi
fi
python - <<'PYEOF'
import os
import torch
import lerobot
import libero
import mujoco
print('python: OK')
print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'available:', torch.cuda.is_available())
print('lerobot:', lerobot.__file__)
print('libero:', libero.__file__)
print('mujoco:', mujoco.__version__)
print('MUJOCO_GL =', os.environ.get('MUJOCO_GL'))
PYEOF
