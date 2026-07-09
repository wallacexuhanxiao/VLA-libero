# MoE Ver2 / Ver3 Patch Guide

This document adds two follow-up action-modeling variants after the current Ver1 residual action-token adapter.

## Variant summary

| Version | Name | Insertion point | Main idea |
|---|---|---|---|
| Ver1 | Pre-expert residual MoE adapter | After `action_time_mlp`, before action expert Transformer | Dense routed multi-MLP action-token modulation |
| Ver2 | Action expert FFN-MoE | Replaces `layer.mlp` inside the action expert Transformer | Closer to standard LLM MoE |
| Ver3 | CTRA: Chunk-Time Routed Adapter | After `action_time_mlp`, before action expert Transformer | Router uses action chunk position + Flow Matching timestep |

## Files added

```text
src/vla_libero_moe/action_ffn_moe.py
src/vla_libero_moe/phase_aware_moe_adapter.py
scripts/07_train_ffn_moe_spatial.sh
scripts/08_train_phase_moe_spatial.sh
```

Note: the Ver3 implementation file is still named `phase_aware_moe_adapter.py` for backward compatibility with earlier experiments, but the method name used in documentation is now **CTRA: Chunk-Time Routed Adapter**. This is more precise because the module is not a standard sparse MoE and does not use ground-truth manipulation phase labels. It is a routed residual adapter whose router uses action chunk position and Flow Matching timestep as context.

## Ver2: action expert FFN-MoE

### Architecture

```text
Image + language + state
        ↓
Frozen VLM prefix condition
        ↓
Noisy action tokens + flow timestep
        ↓
action_time_mlp
        ↓
Action Expert Transformer
        ├─ attention
        └─ FFN / MLP replaced by Dense-MoE FFN  ← Ver2
        ↓
action_out_proj
        ↓
predicted velocity v_t
```

This is the version closest to standard LLM MoE, where the Transformer FFN is replaced by a routed set of expert FFNs.

### Patch local LeRobot

Copy the module:

```bash
cp src/vla_libero_moe/action_ffn_moe.py third_party/lerobot/src/lerobot/policies/smolvla/action_ffn_moe.py
```

Edit:

```text
third_party/lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py
```

Add fields:

```python
use_action_ffn_moe: bool = False
action_ffn_moe_num_experts: int = 4
action_ffn_moe_aux_loss_coef: float = 0.01
```

Edit:

```text
third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py
```

Add imports:

```python
from .action_ffn_moe import patch_action_expert_ffn_moe, collect_moe_aux_loss
```

Inside `VLAFlowMatching.__init__`, after `self.vlm_with_expert = SmolVLMWithExpertModel(...)`, add:

```python
if getattr(self.config, "use_action_ffn_moe", False):
    patched_layers = patch_action_expert_ffn_moe(
        self.vlm_with_expert,
        hidden_size=self.vlm_with_expert.expert_hidden_size,
        num_experts=self.config.action_ffn_moe_num_experts,
        aux_loss_coef=self.config.action_ffn_moe_aux_loss_coef,
    )
    print(f"Patched action expert FFN-MoE layers: {patched_layers}")
```

Inside `VLAFlowMatching.forward`, after:

```python
losses = F.mse_loss(u_t, v_t, reduction="none")
```

add:

```python
ffn_moe_aux_loss = collect_moe_aux_loss(self.vlm_with_expert.lm_expert)
if ffn_moe_aux_loss is not None:
    losses = losses + ffn_moe_aux_loss
```

### Run

```bash
bash scripts/07_train_ffn_moe_spatial.sh
```

## Ver3: CTRA, Chunk-Time Routed Adapter

### Architecture

```text
Image + language + state
        ↓
Frozen VLM prefix condition
        ↓
Noisy action tokens + flow timestep
        ↓
action_time_mlp
        ↓
CTRA  ← Ver3
        ├─ action-token hidden state
        ├─ token position embedding inside the action chunk
        └─ Flow Matching timestep embedding
        ↓
Original Action Expert Transformer
        ↓
action_out_proj
        ↓
predicted velocity v_t
```

Ver3 keeps the stable Ver1 insertion point, but makes routing aware of chunk position and Flow Matching time. The router sees both the action token index inside the chunk and the current denoising timestep. It is better described as a routed adapter rather than a standard MoE because it does not replace Transformer FFNs and does not use sparse top-k expert routing.

### Patch local LeRobot

Copy the module:

```bash
cp src/vla_libero_moe/phase_aware_moe_adapter.py third_party/lerobot/src/lerobot/policies/smolvla/phase_aware_moe_adapter.py
```

Edit:

```text
third_party/lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py
```

Add fields:

```python
use_phase_moe_adapter: bool = False
phase_moe_num_experts: int = 4
phase_moe_hidden_multiplier: float = 2.0
phase_moe_residual_scale: float = 1.0
phase_moe_aux_loss_coef: float = 0.01
phase_moe_max_chunk_size: int = 50
```

Edit:

```text
third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py
```

Add import:

```python
from .phase_aware_moe_adapter import PhaseAwareActionTokenMoEAdapter
```

Inside `VLAFlowMatching.__init__`, after `action_time_mlp_out`, add:

```python
self.phase_moe_adapter = None
if getattr(self.config, "use_phase_moe_adapter", False):
    self.phase_moe_adapter = PhaseAwareActionTokenMoEAdapter(
        hidden_size=self.vlm_with_expert.expert_hidden_size,
        num_experts=self.config.phase_moe_num_experts,
        hidden_multiplier=self.config.phase_moe_hidden_multiplier,
        max_chunk_size=self.config.phase_moe_max_chunk_size,
        residual_scale=self.config.phase_moe_residual_scale,
        aux_loss_coef=self.config.phase_moe_aux_loss_coef,
    )
```

Inside `embed_suffix`, after:

```python
action_time_emb = self.action_time_mlp_out(action_time_emb)
```

add:

```python
if self.phase_moe_adapter is not None:
    action_time_emb = self.phase_moe_adapter(action_time_emb, timestep=timestep)
```

Inside `VLAFlowMatching.forward`, after:

```python
losses = F.mse_loss(u_t, v_t, reduction="none")
```

add:

```python
if self.phase_moe_adapter is not None and self.phase_moe_adapter.last_aux_loss is not None:
    losses = losses + self.phase_moe_adapter.last_aux_loss
```

### Run

```bash
bash scripts/08_train_phase_moe_spatial.sh
```

## Recommended experiment order

Run these in this order:

```text
1. Ver1 residual pre-expert MoE adapter     scripts/06_train_moe_spatial.sh
2. Ver3 CTRA adapter                        scripts/08_train_phase_moe_spatial.sh
3. Ver2 action expert FFN-MoE               scripts/07_train_ffn_moe_spatial.sh
```

Reason:

- Ver3 is closest to the already working Ver1 and should be easier to debug.
- Ver2 is more canonical MoE, but it changes every action expert FFN and is more likely to need LR / aux loss tuning.

## Fair comparison settings

Use the same training budget and evaluation protocol:

```text
steps = 20k initially, then 50k / 100k if promising
batch_size = 16
suite = libero_spatial
eval = 10 tasks × 10 episodes = 100 rollouts
horizon sweep = n_action_steps in {6, 8, 10}
```

Initial success criteria:

```text
Ver3 CTRA is promising if it keeps or improves Ver1's h=10 result while reducing the task 3/4/5 failure.
Ver2 is promising if it beats vanilla at the same horizon without severe instability.
```
