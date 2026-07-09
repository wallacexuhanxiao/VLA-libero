from __future__ import annotations

import copy
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn


class PAMAEActionExpertMixture(nn.Module):
    """PAMAE-style phase-aware sparse MoE action expert for SmolVLA.

    Ver4 attempts to approximate the PAMAE idea more directly than the
    Routed Multi-MLP Adapter:

    - Routed Multi-MLP Adapter: action_time_mlp -> routed residual MLP adapter -> original action expert Transformer
    - Ver4: action_time_mlp -> K action expert Transformer copies -> sparse phase-aware routing

    This module is a research prototype, not an official PAMAE implementation.
    The official PAMAE idea replaces the original flow-matching action expert
    with a phase-aware sparse expert mixture. Here we implement the same high
    level idea in a way that can be patched into the current LeRobot/SmolVLA
    code path.

    Routing context:
        router_context = LN(action_token_hidden) + token_pos_emb + time_emb

    Auxiliary losses:
        - load-balance loss over expert usage;
        - phase prediction loss using phase labels if provided;
        - routing alignment loss encouraging phase-consistent expert assignment.

    If no external phase labels are available, the module uses action-token
    position bins as pseudo phases. This is weaker than real phase labels but is
    useful for LIBERO-Spatial where explicit phase annotations are unavailable.
    """

    def __init__(
        self,
        base_lm_expert: nn.Module,
        hidden_size: int,
        num_experts: int = 4,
        top_k: int = 2,
        num_phases: int = 4,
        max_chunk_size: int = 50,
        phase_loss_coef: float = 0.05,
        routing_alignment_coef: float = 0.05,
        balance_loss_coef: float = 0.01,
        init_from_base: bool = True,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}")
        if top_k < 1 or top_k > num_experts:
            raise ValueError(f"top_k must be in [1, num_experts], got top_k={top_k}, num_experts={num_experts}")
        if num_phases < 1:
            raise ValueError(f"num_phases must be >= 1, got {num_phases}")

        self.num_experts = num_experts
        self.top_k = top_k
        self.num_phases = num_phases
        self.max_chunk_size = max_chunk_size
        self.phase_loss_coef = phase_loss_coef
        self.routing_alignment_coef = routing_alignment_coef
        self.balance_loss_coef = balance_loss_coef

        # K action expert Transformer copies. This is memory-heavy but keeps the
        # first implementation easy to understand and debug.
        self.experts = nn.ModuleList([copy.deepcopy(base_lm_expert) for _ in range(num_experts)])
        if not init_from_base:
            for expert in self.experts:
                expert.reset_parameters() if hasattr(expert, "reset_parameters") else None

        self.router_norm = nn.LayerNorm(hidden_size)
        self.token_pos_emb = nn.Embedding(max_chunk_size, hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.phase_head = nn.Linear(hidden_size, num_phases)

        self.last_aux_loss: torch.Tensor | None = None
        self.last_router_probs: torch.Tensor | None = None
        self.last_phase_logits: torch.Tensor | None = None

    @contextmanager
    def _temporarily_use_expert(self, vlm_with_expert: nn.Module, expert: nn.Module):
        """Temporarily swap `vlm_with_expert.lm_expert` for a single expert.

        This makes Ver4 easy to patch without rewriting the whole
        SmolVLMWithExpert forward pass. It is slower than a fused
        implementation but acceptable for a first experimental variant.
        """

        old_expert = vlm_with_expert.lm_expert
        vlm_with_expert.lm_expert = expert
        try:
            yield
        finally:
            vlm_with_expert.lm_expert = old_expert

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

    def _pseudo_phase_labels(self, bsize: int, seq_len: int, device: torch.device) -> torch.Tensor:
        # Split the action chunk into `num_phases` position bins.
        pos = torch.arange(seq_len, device=device)
        labels = torch.div(pos * self.num_phases, seq_len, rounding_mode="floor")
        labels = labels.clamp(max=self.num_phases - 1)
        return labels[None, :].expand(bsize, seq_len)

    def _routing_context(self, suffix_embs: torch.Tensor, timestep: torch.Tensor | float | None):
        bsize, seq_len, _ = suffix_embs.shape
        if seq_len > self.max_chunk_size:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_chunk_size={self.max_chunk_size}; "
                "increase pamae_max_chunk_size."
            )

        pos_ids = torch.arange(seq_len, device=suffix_embs.device)
        pos_emb = self.token_pos_emb(pos_ids)[None, :, :].to(dtype=suffix_embs.dtype)
        time_emb = self.time_mlp(self._format_timestep(timestep, suffix_embs))[:, None, :]
        return self.router_norm(suffix_embs) + pos_emb + time_emb

    def _sparse_router_probs(self, router_logits: torch.Tensor) -> torch.Tensor:
        router_probs = torch.softmax(router_logits, dim=-1)
        if self.top_k == self.num_experts:
            return router_probs

        top_values, top_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        sparse_probs = torch.zeros_like(router_probs).scatter_(-1, top_indices, top_values)
        sparse_probs = sparse_probs / sparse_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return sparse_probs

    def _compute_aux_loss(
        self,
        router_probs: torch.Tensor,
        phase_logits: torch.Tensor,
        phase_labels: torch.Tensor,
    ) -> torch.Tensor:
        # 1) load balancing: avoid routing collapse.
        mean_probs = router_probs.mean(dim=(0, 1))
        balance_loss = self.num_experts * torch.sum(mean_probs * mean_probs) - 1.0

        # 2) phase prediction: make router context explicitly phase-aware.
        phase_loss = F.cross_entropy(
            phase_logits.reshape(-1, self.num_phases),
            phase_labels.reshape(-1),
        )

        # 3) routing alignment: encourage tokens from the same phase to choose
        # consistent experts. If num_phases != num_experts, phases are mapped
        # cyclically to experts as a simple pseudo-supervised prior.
        expert_targets = (phase_labels % self.num_experts).reshape(-1)
        routing_alignment_loss = F.nll_loss(
            torch.log(router_probs.reshape(-1, self.num_experts).clamp_min(1e-8)),
            expert_targets,
        )

        return (
            self.balance_loss_coef * balance_loss
            + self.phase_loss_coef * phase_loss
            + self.routing_alignment_coef * routing_alignment_loss
        )

    def combine_expert_outputs(
        self,
        expert_suffix_outs: list[torch.Tensor],
        suffix_embs: torch.Tensor,
        timestep: torch.Tensor | float | None = None,
        phase_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Combine expert Transformer outputs with sparse phase-aware routing."""

        bsize, seq_len, _ = suffix_embs.shape
        if phase_labels is None:
            phase_labels = self._pseudo_phase_labels(bsize, seq_len, suffix_embs.device)
        else:
            phase_labels = phase_labels.to(device=suffix_embs.device, dtype=torch.long)

        router_context = self._routing_context(suffix_embs, timestep)
        router_logits = self.router(router_context)
        router_probs = self._sparse_router_probs(router_logits)
        phase_logits = self.phase_head(router_context)

        expert_stack = torch.stack(expert_suffix_outs, dim=-2)  # [B, T, K, H]
        mixed = torch.sum(router_probs.unsqueeze(-1) * expert_stack, dim=-2)

        self.last_aux_loss = self._compute_aux_loss(router_probs, phase_logits, phase_labels)
        self.last_router_probs = router_probs.detach()
        self.last_phase_logits = phase_logits.detach()
        return mixed

    def forward_with_vlm(
        self,
        vlm_with_expert: nn.Module,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: list[torch.Tensor | None],
        timestep: torch.Tensor | float | None = None,
        past_key_values=None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
        phase_labels: torch.Tensor | None = None,
    ):
        """Run all action experts and combine their suffix outputs.

        Expected use inside `VLAFlowMatching.forward` or `denoise_step` when
        `inputs_embeds=[prefix_embs, suffix_embs]` or `[None, suffix_embs]`.
        Prefix-only KV cache construction should keep using the original
        `vlm_with_expert.forward` because no action expert output is needed.
        """

        suffix_embs = inputs_embeds[1]
        if suffix_embs is None:
            return vlm_with_expert.forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                fill_kv_cache=fill_kv_cache,
            )

        prefix_out = None
        returned_past_key_values = None
        expert_suffix_outs = []

        for expert in self.experts:
            with self._temporarily_use_expert(vlm_with_expert, expert):
                outputs_embeds, returned_past_key_values = vlm_with_expert.forward(
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    fill_kv_cache=fill_kv_cache,
                )
            if outputs_embeds[0] is not None and prefix_out is None:
                prefix_out = outputs_embeds[0]
            expert_suffix_outs.append(outputs_embeds[1])

        mixed_suffix = self.combine_expert_outputs(
            expert_suffix_outs=expert_suffix_outs,
            suffix_embs=suffix_embs,
            timestep=timestep,
            phase_labels=phase_labels,
        )
        return [prefix_out, mixed_suffix], returned_past_key_values
