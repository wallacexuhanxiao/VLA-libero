from __future__ import annotations

import torch
from torch import nn


class PhaseAwareActionTokenMoEAdapter(nn.Module):
    """Phase/time-aware residual dense-MoE adapter for SmolVLA action tokens.

    This is Ver3 of the MoE route. It keeps the lightweight pre-action-expert
    adapter placement from Ver1, but makes the router aware of:

    - action-token position inside the chunk, which is a proxy for manipulation phase;
    - Flow Matching timestep, which tells the router which denoising stage it is in.

    Shape:
        x: [batch, chunk_size, expert_hidden_size]
        timestep: [batch] or scalar in [0, 1]
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 4,
        hidden_multiplier: float = 2.0,
        max_chunk_size: int = 50,
        residual_scale: float = 1.0,
        aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}")

        self.num_experts = num_experts
        self.max_chunk_size = max_chunk_size
        self.residual_scale = residual_scale
        self.aux_loss_coef = aux_loss_coef

        inner_dim = max(1, int(hidden_size * hidden_multiplier))
        self.router_norm = nn.LayerNorm(hidden_size)
        self.token_pos_emb = nn.Embedding(max_chunk_size, hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
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

    def _format_timestep(self, timestep: torch.Tensor | float | None, x: torch.Tensor) -> torch.Tensor:
        bsize = x.shape[0]
        if timestep is None:
            timestep = torch.zeros(bsize, dtype=x.dtype, device=x.device)
        elif not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, dtype=x.dtype, device=x.device).expand(bsize)
        else:
            timestep = timestep.to(device=x.device, dtype=x.dtype)
            if timestep.ndim == 0:
                timestep = timestep.expand(bsize)
        return timestep.reshape(bsize, 1)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor | float | None = None) -> torch.Tensor:
        bsize, seq_len, _ = x.shape
        if seq_len > self.max_chunk_size:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_chunk_size={self.max_chunk_size}; "
                "increase action_phase_moe_max_chunk_size."
            )

        pos_ids = torch.arange(seq_len, device=x.device)
        pos_emb = self.token_pos_emb(pos_ids)[None, :, :].to(dtype=x.dtype)

        timestep = self._format_timestep(timestep, x)
        time_emb = self.time_mlp(timestep)[:, None, :]

        router_context = self.router_norm(x) + pos_emb + time_emb
        router_logits = self.router(router_context)
        router_probs = torch.softmax(router_logits, dim=-1)  # [B, T, K]

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        mixed = torch.sum(router_probs.unsqueeze(-1) * expert_outputs, dim=-2)

        mean_probs = router_probs.mean(dim=(0, 1))
        balance_loss = self.num_experts * torch.sum(mean_probs * mean_probs) - 1.0
        self.last_aux_loss = self.aux_loss_coef * balance_loss
        return x + self.residual_scale * mixed
