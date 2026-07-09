# Ver4 PAMAE-style Patch Guide

This document adds a PAMAE-style Ver4 route after the existing Ver1/Ver2/Ver3 MoE experiments.

Important: this is a **PAMAE-style research prototype**, not an official PAMAE implementation. It follows the high-level idea of replacing a shared flow-matching action expert with a phase-aware sparse expert mixture, while adapting the implementation to the current LeRobot/SmolVLA code path.

## Method position

| Version | Name | Insertion / replacement point | Comment |
|---|---|---|---|
| Ver1 | Residual adapter MoE | Before action expert Transformer | Lightweight adapter |
| Ver2 | FFN-MoE | Inside action expert Transformer FFN | Closest to LLM MoE |
| Ver3 | Routed Multi-MLP Adapter | Before action expert Transformer | Best current lightweight routed-adapter route |
| Ver4 | PAMAE-style action expert MoE | Replaces shared action expert with sparse expert mixture | Closest to PAMAE idea |

## Architecture

```text
Image + language + state
        ↓
Frozen VLM prefix condition
        ↓
Noisy action tokens + flow timestep
        ↓
action_time_mlp
        ↓
========================================================
Ver4 PAMAE-style sparse phase-aware action expert mixture
        ├─ Action Expert Transformer 1
        ├─ Action Expert Transformer 2
        ├─ Action Expert Transformer 3
        └─ Action Expert Transformer 4
        │
        ├─ Router context
        │     = LN(action-token hidden)
        │       + token-position embedding
        │       + Flow-Matching timestep embedding
        │
        ├─ Sparse top-k routing
        ├─ Phase prediction head
        └─ Routing alignment / load-balance losses
========================================================
        ↓
Mixed action expert hidden states
        ↓
action_out_proj
        ↓
predicted velocity v_t
```

Compared with Ver3, Ver4 does **not** only modulate the input to the original action expert. Instead, it runs multiple action expert Transformer copies and routes/combines their outputs.

## Files added

```text
src/vla_libero_moe/pamae_action_expert.py
scripts/09_train_pamae_spatial.sh
docs/PAMAE_VER4_PATCH.md
```

## Patch local LeRobot

Copy the module:

```bash
cp src/vla_libero_moe/pamae_action_expert.py third_party/lerobot/src/lerobot/policies/smolvla/pamae_action_expert.py
```

Edit:

```text
third_party/lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py
```

Add fields:

```python
use_pamae_action_expert: bool = False
pamae_num_experts: int = 4
pamae_top_k: int = 2
pamae_num_phases: int = 4
pamae_max_chunk_size: int = 50
pamae_phase_loss_coef: float = 0.05
pamae_routing_alignment_coef: float = 0.05
pamae_balance_loss_coef: float = 0.01
```

Edit:

```text
third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py
```

Add import:

```python
from .pamae_action_expert import PAMAEActionExpertMixture
```

Inside `VLAFlowMatching.__init__`, after `self.vlm_with_expert = SmolVLMWithExpertModel(...)`, add:

```python
self.pamae_action_expert = None
if getattr(self.config, "use_pamae_action_expert", False):
    self.pamae_action_expert = PAMAEActionExpertMixture(
        base_lm_expert=self.vlm_with_expert.lm_expert,
        hidden_size=self.vlm_with_expert.expert_hidden_size,
        num_experts=self.config.pamae_num_experts,
        top_k=self.config.pamae_top_k,
        num_phases=self.config.pamae_num_phases,
        max_chunk_size=self.config.pamae_max_chunk_size,
        phase_loss_coef=self.config.pamae_phase_loss_coef,
        routing_alignment_coef=self.config.pamae_routing_alignment_coef,
        balance_loss_coef=self.config.pamae_balance_loss_coef,
    )
```

## Patch training forward

Inside `VLAFlowMatching.forward`, replace the normal action expert call:

```python
(_, suffix_out), _ = self.vlm_with_expert.forward(
    attention_mask=att_2d_masks,
    position_ids=position_ids,
    past_key_values=None,
    inputs_embeds=[prefix_embs, suffix_embs],
    use_cache=False,
    fill_kv_cache=False,
)
```

with:

```python
if self.pamae_action_expert is not None:
    (_, suffix_out), _ = self.pamae_action_expert.forward_with_vlm(
        self.vlm_with_expert,
        attention_mask=att_2d_masks,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, suffix_embs],
        timestep=time,
        use_cache=False,
        fill_kv_cache=False,
    )
else:
    (_, suffix_out), _ = self.vlm_with_expert.forward(
        attention_mask=att_2d_masks,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, suffix_embs],
        use_cache=False,
        fill_kv_cache=False,
    )
```

After the normal flow-matching MSE loss:

```python
losses = F.mse_loss(u_t, v_t, reduction="none")
```

add:

```python
if self.pamae_action_expert is not None and self.pamae_action_expert.last_aux_loss is not None:
    losses = losses + self.pamae_action_expert.last_aux_loss
```

## Patch inference denoise step

Inside `VLAFlowMatching.denoise_step`, replace:

```python
outputs_embeds, _ = self.vlm_with_expert.forward(
    attention_mask=full_att_2d_masks,
    position_ids=position_ids,
    past_key_values=past_key_values,
    inputs_embeds=[None, suffix_embs],
    use_cache=self.config.use_cache,
    fill_kv_cache=False,
)
```

with:

```python
if self.pamae_action_expert is not None:
    outputs_embeds, _ = self.pamae_action_expert.forward_with_vlm(
        self.vlm_with_expert,
        attention_mask=full_att_2d_masks,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=[None, suffix_embs],
        timestep=timestep,
        use_cache=self.config.use_cache,
        fill_kv_cache=False,
    )
else:
    outputs_embeds, _ = self.vlm_with_expert.forward(
        attention_mask=full_att_2d_masks,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=[None, suffix_embs],
        use_cache=self.config.use_cache,
        fill_kv_cache=False,
    )
```

## Run

Start with smaller batch size because Ver4 keeps multiple action expert Transformer copies:

```bash
BATCH_SIZE=8 bash scripts/09_train_pamae_spatial.sh
```

If memory is safe, try:

```bash
BATCH_SIZE=12 bash scripts/09_train_pamae_spatial.sh
BATCH_SIZE=16 bash scripts/09_train_pamae_spatial.sh
```

## Two-stage training suggestion

The PAMAE paper emphasizes staged optimization. A practical version for this repo:

### Stage 1: warm up expert mixture

Use weak auxiliary losses or turn them off:

```bash
STEPS=10000 \
BATCH_SIZE=8 \
OUTPUT_DIR=$OUTPUT_ROOT/smolvla_libero_spatial_pamae_stage1 \
bash scripts/09_train_pamae_spatial.sh
```

Recommended config for Stage 1:

```text
pamae_phase_loss_coef = 0.0
pamae_routing_alignment_coef = 0.0
pamae_balance_loss_coef = 0.01
```

### Stage 2: phase-consistent routing

Resume from Stage 1 checkpoint and enable phase/routing losses:

```text
pamae_phase_loss_coef = 0.05
pamae_routing_alignment_coef = 0.05
pamae_balance_loss_coef = 0.01
```

Note: the current script exposes these as policy config flags. If resume-from-checkpoint is needed, add the corresponding LeRobot resume flag used by your local version.

## Expected behavior

Ver4 is more faithful to the PAMAE idea than the Routed Multi-MLP Adapter, but it is also more expensive and riskier:

```text
Routed Multi-MLP Adapter: lightweight and already strong; best first practical route.
Ver4: closer to PAMAE; may need smaller batch size, more steps, and staged training.
```

Initial success criteria:

```text
1. It should beat vanilla baseline at the same horizon.
2. It should reduce task 3/4/5 collapse.
3. It should ideally beat or match Ver3 after staged training.
```
