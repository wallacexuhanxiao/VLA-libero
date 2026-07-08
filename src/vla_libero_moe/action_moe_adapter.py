from __future__ import annotations

import torch
from torch import nn


class ActionTokenMoEAdapter(nn.Module):
    """Residual dense-MoE adapter for SmolVLA action tokens.

    Shape:
        x: [batch, chunk_size, expert_hidden_size]

    It is inserted after SmolVLA's action_time_mlp and before the action expert
    transformer. It does not replace the official action expert.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 4,
        hidden_multiplier: float = 2.0,
        residual_scale: float = 1.0,
        aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}")

        self.num_experts = num_experts
        self.residual_scale = residual_scale
        self.aux_loss_coef = aux_loss_coef

        inner_dim = max(1, int(hidden_size * hidden_multiplier))
        self.router_norm = nn.LayerNorm(hidden_size)
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, inner_dim),
                    nn.SiLU(),
                    nn.Linear(inner_dim, hidden_size),
                )
                for _ in range(num_experts)
            ]
        )

        # Identity-like start: output ~= input at initialization.
        for expert in self.experts:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)

        self.last_aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        router_logits = self.router(self.router_norm(x))
        router_probs = torch.softmax(router_logits, dim=-1)  # [B, T, K]

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        mixed = torch.sum(router_probs.unsqueeze(-1) * expert_outputs, dim=-2)

        mean_probs = router_probs.mean(dim=(0, 1))
        balance_loss = self.num_experts * torch.sum(mean_probs * mean_probs) - 1.0
        self.last_aux_loss = self.aux_loss_coef * balance_loss

        return x + self.residual_scale * mixed
