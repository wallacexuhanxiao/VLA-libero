from __future__ import annotations

import copy
from typing import Iterable

import torch
from torch import nn


class DenseMoEFeedForwardWrapper(nn.Module):
    """Dense-MoE replacement for an action expert Transformer FFN/MLP.

    This is Ver2 of the MoE route. Unlike `ActionTokenMoEAdapter`, which is
    inserted before the action expert Transformer, this wrapper replaces the
    per-layer `layer.mlp` module inside the SmolVLA action expert Transformer.

    Shape:
        x: [batch, seq_len, hidden_size]

    The wrapper evaluates all experts densely and combines them with a soft
    router. This is slower than sparse top-k MoE but much easier to debug and
    deterministic for small action-token sequences.
    """

    def __init__(
        self,
        base_mlp: nn.Module,
        hidden_size: int,
        num_experts: int = 4,
        aux_loss_coef: float = 0.01,
        init_from_base: bool = True,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}")

        self.num_experts = num_experts
        self.aux_loss_coef = aux_loss_coef
        self.router_norm = nn.LayerNorm(hidden_size)
        self.router = nn.Linear(hidden_size, num_experts, bias=False)

        if init_from_base:
            self.experts = nn.ModuleList([copy.deepcopy(base_mlp) for _ in range(num_experts)])
        else:
            # The base MLP is still used as the architecture template; callers can
            # reinitialize after construction if they want a fully fresh start.
            self.experts = nn.ModuleList([copy.deepcopy(base_mlp) for _ in range(num_experts)])
            for expert in self.experts:
                expert.reset_parameters() if hasattr(expert, "reset_parameters") else None

        self.last_aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        router_logits = self.router(self.router_norm(x))
        router_probs = torch.softmax(router_logits, dim=-1)  # [B, T, K]

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        mixed = torch.sum(router_probs.unsqueeze(-1) * expert_outputs, dim=-2)

        mean_probs = router_probs.mean(dim=(0, 1))
        balance_loss = self.num_experts * torch.sum(mean_probs * mean_probs) - 1.0
        self.last_aux_loss = self.aux_loss_coef * balance_loss
        return mixed


def patch_action_expert_ffn_moe(
    vlm_with_expert: nn.Module,
    hidden_size: int,
    num_experts: int = 4,
    aux_loss_coef: float = 0.01,
    layer_indices: Iterable[int] | None = None,
) -> int:
    """Replace SmolVLA action expert layer MLPs with dense-MoE MLPs.

    Expected target:
        `VLAFlowMatching.vlm_with_expert.lm_expert.layers[*].mlp`

    Returns:
        Number of patched action expert layers.
    """

    layers = getattr(getattr(vlm_with_expert, "lm_expert"), "layers")
    selected = set(layer_indices) if layer_indices is not None else None
    patched = 0

    for idx, layer in enumerate(layers):
        if selected is not None and idx not in selected:
            continue
        if isinstance(getattr(layer, "mlp", None), DenseMoEFeedForwardWrapper):
            continue
        layer.mlp = DenseMoEFeedForwardWrapper(
            base_mlp=layer.mlp,
            hidden_size=hidden_size,
            num_experts=num_experts,
            aux_loss_coef=aux_loss_coef,
        )
        patched += 1

    return patched


def collect_moe_aux_loss(module: nn.Module) -> torch.Tensor | None:
    """Collect load-balancing losses from all MoE modules under `module`."""

    losses = []
    for submodule in module.modules():
        aux = getattr(submodule, "last_aux_loss", None)
        if aux is not None:
            losses.append(aux)
    if not losses:
        return None
    total = losses[0]
    for loss in losses[1:]:
        total = total + loss
    return total
