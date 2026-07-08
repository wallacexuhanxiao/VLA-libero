# VLA-libero

LeRobot + LIBERO + SmolVLA baseline project, with action-token and action-expert MoE experiments.

## Goal

1. Reproduce official LIBERO evaluation with a known checkpoint.
2. Fine-tune official SmolVLA on LIBERO-Spatial using LeRobot.
3. Establish a working closed-loop baseline.
4. Compare a vanilla SmolVLA action expert against multiple MoE action-modeling variants.

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
  06_train_moe_spatial.sh
  07_train_ffn_moe_spatial.sh
  08_train_phase_moe_spatial.sh
docs/
  SOP.md
  experiment_log_template.md
  EXPERIMENT_REPORT.md
  MANUAL_MOE_PATCH.md
  MOE_VER2_VER3_PATCH.md
src/vla_libero_moe/
  action_moe_adapter.py
  action_ffn_moe.py
  phase_aware_moe_adapter.py
experiments/
  .gitkeep
results/
  libero_spatial_horizon_sweep/
```

## Quick start

```bash
bash scripts/01_eval_pi05_libero_spatial.sh
bash scripts/02_train_smolvla_spatial.sh
bash scripts/03_eval_smolvla_spatial.sh
```

## Current result

The main closed-loop LIBERO-Spatial report is in:

```text
docs/EXPERIMENT_REPORT.md
```

Best MoE setting measured so far:

```text
MoE 20k, n_action_steps=10: 48/100 = 48.0%
```

Strongest vanilla baseline among tested horizons:

```text
Baseline 20k, n_action_steps=8: 44/100 = 44.0%
```

Largest same-horizon gain:

```text
n_action_steps=10: MoE 48.0% vs Baseline 35.0%, +13.0 points
```

## MoE variants

```text
Ver1: Residual pre-expert MoE adapter
      action_time_mlp -> MoE adapter -> original action expert Transformer
      script: scripts/06_train_moe_spatial.sh

Ver2: Action expert FFN-MoE
      replace action expert Transformer layer.mlp with dense MoE FFN
      script: scripts/07_train_ffn_moe_spatial.sh

Ver3: Phase/time-aware MoE adapter
      action_time_mlp -> phase-aware MoE adapter -> original action expert Transformer
      script: scripts/08_train_phase_moe_spatial.sh
```

Patch guide:

```text
docs/MOE_VER2_VER3_PATCH.md
```

## Research route

```text
Official LeRobot LIBERO eval
        ↓
Official SmolVLA fine-tuning
        ↓
Stable closed-loop LIBERO success-rate baseline
        ↓
Ver1 residual action-token MoE adapter
        ↓
Ver3 phase/time-aware routing or Ver2 action-expert FFN-MoE
        ↓
Compare closed-loop success rate under the same official eval
```
