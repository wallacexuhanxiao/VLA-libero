# VLA-libero

Official LeRobot + LIBERO + SmolVLA baseline project, with a later path toward Flow Matching action heads.

## Goal

1. Reproduce official LIBERO evaluation with a known checkpoint.
2. Fine-tune official SmolVLA on LIBERO-Spatial using LeRobot.
3. Establish a working closed-loop baseline.
4. Only after the baseline works, modify or replace the action head with a Flow Matching variant.

## Why this repo exists

The previous custom VLA pipeline could reduce offline imitation losses, but closed-loop LIBERO success was low. This project avoids reimplementing data loading, environment wrappers, camera keys, state/action conventions, and rollout logic. It uses LeRobot as the source of truth.

## Structure

```text
configs/
  env.sh
scripts/
  00_install_lerobot.sh
  01_eval_pi05_libero_spatial.sh
  02_train_smolvla_spatial.sh
  03_eval_smolvla_spatial.sh
  04_find_checkpoints.sh
docs/
  SOP.md
  experiment_log_template.md
experiments/
  .gitkeep
```

## Quick start

```bash
conda create -n lerobot python=3.10 -y
conda activate lerobot

bash scripts/00_install_lerobot.sh
source configs/env.sh

bash scripts/01_eval_pi05_libero_spatial.sh
bash scripts/02_train_smolvla_spatial.sh
bash scripts/03_eval_smolvla_spatial.sh
```

## Research route

```text
Official LeRobot LIBERO eval
        ↓
Official SmolVLA fine-tuning
        ↓
Stable closed-loop LIBERO success-rate baseline
        ↓
Replace only the action head / action expert with Flow Matching
        ↓
Compare success rate, smoothness, and robustness under the same official eval
```
