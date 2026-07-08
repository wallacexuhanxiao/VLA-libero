# Codex Handoff: SmolVLA MoE Action Expert

## Project goal

This repo is for a structure-innovation VLA experiment on LIBERO using official LeRobot.

Main comparison:

```text
Baseline: pretrained VLM + scratch vanilla action expert
Ours:     pretrained VLM + scratch MoE-modulated action expert
```

Do not use `lerobot/smolvla_base` as the main comparison target for this experiment, because that loads a pretrained action expert and makes architecture changes harder to compare fairly.

## Current repository state

Important files:

```text
scripts/00_install_lerobot.sh
scripts/01_eval_pi05_libero_spatial.sh
scripts/02_train_smolvla_spatial.sh
scripts/03_eval_smolvla_spatial.sh
scripts/04_find_checkpoints.sh
scripts/06_train_moe_spatial.sh
src/vla_libero_moe/action_moe_adapter.py
docs/MANUAL_MOE_PATCH.md
```

`02_train_smolvla_spatial.sh` is the vanilla scratch-expert baseline.

`06_train_moe_spatial.sh` is the MoE experiment entrypoint.

`action_moe_adapter.py` contains the residual dense-MoE adapter. It expects action-token embeddings shaped `[B, T, H]` and returns the same shape.

## Required Codex task

Codex should integrate `ActionTokenMoEAdapter` into local LeRobot SmolVLA.

Target LeRobot files after installation:

```text
third_party/lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py
third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py
```

Follow `docs/MANUAL_MOE_PATCH.md`.

The intended insertion point is:

```text
after action_time_mlp_out
before suffix action tokens enter the action expert transformer
```

Do not replace the whole action expert transformer in the first version.

## Expected architecture

```text
image + language + state
        -> VLM prefix condition

noisy action + flow time
        -> action_time_mlp
        -> ActionTokenMoEAdapter
        -> original SmolVLA action expert transformer
        -> action_out_proj
        -> predicted velocity v_t
```

## Training commands

First run vanilla baseline:

```bash
bash scripts/02_train_smolvla_spatial.sh
```

Then run MoE version:

```bash
bash scripts/06_train_moe_spatial.sh
```

If 5090 OOM, reduce `--batch_size=4` to `--batch_size=2` in both scripts.

## Evaluation commands

Find checkpoints:

```bash
bash scripts/04_find_checkpoints.sh
```

Evaluate one task:

```bash
CHECKPOINT=/path/to/checkpoint TASK_IDS='[0]' EPISODES=10 bash scripts/03_eval_smolvla_spatial.sh
```

Evaluate all LIBERO-Spatial tasks:

```bash
CHECKPOINT=/path/to/checkpoint TASK_IDS='[0,1,2,3,4,5,6,7,8,9]' EPISODES=10 bash scripts/03_eval_smolvla_spatial.sh
```

## Success criteria

At minimum, report:

```text
vanilla scratch expert success_rate
MoE scratch expert success_rate
training loss curve
rollout failure type
inference latency if easy to collect
```

The first valid research comparison is not pretrained SmolVLA vs MoE. It is:

```text
scratch vanilla expert vs scratch MoE expert
```

## Safety notes

Keep all official LeRobot dataset, observation, action, normalization, and LIBERO eval code unchanged unless there is a clear bug.

Do not change the VLM backbone for this stage.

Do not change the flow-matching loss except for adding the small MoE load-balancing auxiliary loss.

Do not vendor the entire LeRobot source into this repo unless necessary. Prefer keeping LeRobot under `third_party/lerobot` and documenting small local modifications.
