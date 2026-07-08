# Manual MoE Patch Guide

Goal: compare a vanilla scratch action expert with a MoE-modulated scratch action expert.

Current repo already contains:

- `scripts/02_train_smolvla_spatial.sh`: vanilla scratch action expert baseline.
- `src/vla_libero_moe/action_moe_adapter.py`: residual dense-MoE adapter.
- `scripts/06_train_moe_spatial.sh`: MoE experiment entrypoint.

## Step 1: install LeRobot

Run:

```bash
bash scripts/00_install_lerobot.sh
source configs/env.sh
```

## Step 2: copy adapter into local LeRobot

Run:

```bash
cp src/vla_libero_moe/action_moe_adapter.py third_party/lerobot/src/lerobot/policies/smolvla/action_moe_adapter.py
```

## Step 3: edit local LeRobot config

Open:

```text
third_party/lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py
```

Add these config fields near the other SmolVLA action-expert settings:

```python
use_action_moe_adapter: bool = False
action_moe_num_experts: int = 4
action_moe_hidden_multiplier: float = 2.0
action_moe_residual_scale: float = 1.0
action_moe_aux_loss_coef: float = 0.01
```

## Step 4: edit local LeRobot model

Open:

```text
third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py
```

Add this import near the other SmolVLA imports:

```python
from .action_moe_adapter import ActionTokenMoEAdapter
```

Inside `VLAFlowMatching.__init__`, after `action_time_mlp_out`, create `self.action_moe_adapter`. It should be `ActionTokenMoEAdapter(...)` when `config.use_action_moe_adapter` is true, otherwise `None`.

Inside `embed_suffix`, after `action_time_mlp_out`, apply:

```python
if self.action_moe_adapter is not None:
    action_time_emb = self.action_moe_adapter(action_time_emb)
```

Inside `forward`, after the normal flow-matching MSE loss, add the adapter load-balancing loss when it exists.

## Step 5: sanity check

Run:

```bash
python - <<'PY'
from lerobot.policies.smolvla.action_moe_adapter import ActionTokenMoEAdapter
print('MoE adapter import OK:', ActionTokenMoEAdapter.__name__)
PY
```

## Step 6: run experiments

Vanilla baseline:

```bash
bash scripts/02_train_smolvla_spatial.sh
```

MoE version:

```bash
bash scripts/06_train_moe_spatial.sh
```

Compare success rate, loss curve, failure phase, and rollout videos.
