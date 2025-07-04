from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class BaseMoELayer(nn.Module, ABC):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_experts: int,
        top_k: int,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_features, hidden_features),
                    nn.GELU(),
                    nn.Linear(hidden_features, out_features),
                )
                for _ in range(num_experts)
            ]
        )

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        pass


class MoELayer(BaseMoELayer):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_experts: int,
        top_k: int = 1,
    ):
        super().__init__(in_features, hidden_features, out_features, num_experts, top_k)
        self.gating = nn.Linear(in_features, num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch_size, num_tokens, in_features = x.shape
        x_flat = x.reshape(-1, in_features)

        gating_logits = self.gating(x_flat)

        weights_top_k, top_indices = torch.topk(gating_logits, self.top_k, dim=-1)
        weights_top_k = torch.softmax(weights_top_k, dim=-1).to(x.dtype)

        flat_top_indices = top_indices.flatten()
        flat_x = x_flat.repeat_interleave(self.top_k, dim=0)

        expert_outputs = torch.empty_like(flat_x)
        for i in range(self.num_experts):
            mask = flat_top_indices == i
            if mask.any():
                expert_outputs[mask] = self.experts[i](flat_x[mask])

        expert_outputs = expert_outputs.view(x_flat.size(0), self.top_k, -1)

        weighted_outputs = weights_top_k.unsqueeze(-1) * expert_outputs
        combined_output = weighted_outputs.sum(dim=1)

        final_output = combined_output.reshape(batch_size, num_tokens, -1)

        routing_info = {
            "gating_logits": gating_logits.reshape(batch_size, num_tokens, -1),
            "top_indices": top_indices.reshape(batch_size, num_tokens, -1),
        }

        return final_output, routing_info


class GaussianMoELayer(BaseMoELayer):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_experts: int,
        top_k: int = 2,
    ):
        super().__init__(in_features, hidden_features, out_features, num_experts, top_k)
        self.expert_mus = nn.Parameter(torch.randn(num_experts, in_features))
        self.expert_log_sigmas = nn.Parameter(torch.zeros(num_experts, in_features))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch_size, num_tokens, in_features = x.shape
        x_flat = x.reshape(-1, in_features)

        sigmas = torch.exp(self.expert_log_sigmas)
        
        log_probs = torch.empty(x_flat.size(0), self.num_experts, device=x.device)
        for i in range(self.num_experts):
            dist_sq = ((x_flat - self.expert_mus[i]) / sigmas[i]).pow(2).sum(dim=-1)
            log_probs[:, i] = -0.5 * dist_sq - self.expert_log_sigmas[i].sum()
        
        weights = torch.softmax(log_probs, dim=-1)
        
        weights_top_k, top_indices = torch.topk(weights, self.top_k, dim=-1)
        weights_top_k = weights_top_k / (weights_top_k.sum(dim=-1, keepdim=True) + 1e-9)

        flat_top_indices = top_indices.flatten()
        flat_x = x_flat.repeat_interleave(self.top_k, dim=0)

        expert_outputs = torch.empty(flat_x.size(0), self.out_features, device=x.device, dtype=x.dtype)

        for i in range(self.num_experts):
            mask = (flat_top_indices == i)
            if mask.any():
                expert_outputs[mask] = self.experts[i](flat_x[mask])
        
        expert_outputs = expert_outputs.view(x_flat.size(0), self.top_k, -1)
        
        weighted_outputs = weights_top_k.unsqueeze(-1) * expert_outputs
        combined_output = weighted_outputs.sum(dim=1)
        
        final_output = combined_output.reshape(batch_size, num_tokens, -1)
        
        routing_info = {
            "log_probs": log_probs.reshape(batch_size, num_tokens, -1),
            "weights": weights.reshape(batch_size, num_tokens, -1),
            "top_indices": top_indices.reshape(batch_size, num_tokens, -1)
        }
        
        return final_output, routing_info


class MemoryGaussianMoELayer(GaussianMoELayer):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_experts: int,
        top_k: int = 2,
        num_heads: int = 4,
        history_len: int = 256,
        rehearsal_rate: float = 0.1,
        priority_decay: float = 0.99,
    ):
        super().__init__(
            in_features, hidden_features, out_features, num_experts, top_k
        )

        self.history_len = history_len
        self.rehearsal_rate = rehearsal_rate
        self.priority_decay = priority_decay

        self.gating_transformer = GatingTransformer(
            in_features, num_heads, in_features, num_experts
        )
        self.raw_gating_proj = nn.Linear(in_features, num_experts)

        self.is_full: torch.Tensor
        self.history_ptr: torch.Tensor
        self.register_buffer("history_q", torch.zeros(self.history_len, in_features))
        self.register_buffer(
            "history_mink", torch.zeros(self.history_len, in_features)
        )
        self.register_buffer(
            "history_priority", torch.full((self.history_len,), -float("inf"))
        )
        self.register_buffer("is_full", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("history_ptr", torch.tensor(0, dtype=torch.long))

    def decay_memory(self):
        current_history_len = self.history_len if self.is_full else self.history_ptr
        if current_history_len > 0:
            self.history_priority[:current_history_len] *= self.priority_decay

    def update_memory(
        self,
        priority: torch.Tensor,
        q_embedding: torch.Tensor,
        min_k_indices: torch.Tensor,
    ):
        if not (self.is_full.item() and priority < self.history_priority.min()):
            with torch.no_grad():
                min_k_embedding = self.expert_mus[min_k_indices].mean(dim=0)

            if self.is_full.item():
                idx_to_replace = torch.argmin(self.history_priority)
            else:
                idx_to_replace = self.history_ptr
                self.history_ptr += 1
                if self.history_ptr.item() >= self.history_len:
                    self.is_full = torch.tensor(True, dtype=torch.bool)

            self.history_q[idx_to_replace] = q_embedding.detach()
            self.history_mink[idx_to_replace] = min_k_embedding.detach()
            self.history_priority[idx_to_replace] = priority.detach()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch_size, num_tokens, in_features = x.shape
        x_flat = x.reshape(-1, in_features)

        raw_logits = self.raw_gating_proj(x_flat)

        current_history_len = (
            self.history_len if self.is_full.item() else int(self.history_ptr.item())
        )
        history_context = self.gating_transformer(
            x_flat,
            self.history_q[:current_history_len].clone(),
            self.history_mink[:current_history_len].clone(),
        )

        gating_logits = raw_logits + history_context

        weights = torch.softmax(gating_logits, dim=-1)

        weights_top_k, top_indices = torch.topk(weights, self.top_k, dim=-1)
        weights_top_k = weights_top_k / (
            weights_top_k.sum(dim=-1, keepdim=True) + 1e-9
        )

        flat_top_indices = top_indices.flatten()
        flat_x = x_flat.repeat_interleave(self.top_k, dim=0)

        expert_outputs = torch.empty(
            flat_x.size(0), self.out_features, device=x.device, dtype=x.dtype
        )

        for i in range(self.num_experts):
            mask = flat_top_indices == i
            if mask.any():
                expert_outputs[mask] = self.experts[i](flat_x[mask])

        expert_outputs = expert_outputs.view(x_flat.size(0), self.top_k, -1)

        weighted_outputs = weights_top_k.unsqueeze(-1) * expert_outputs
        combined_output = weighted_outputs.sum(dim=1)

        final_output = combined_output.reshape(batch_size, num_tokens, -1)

        routing_info = {
            "log_probs": torch.log_softmax(gating_logits, dim=-1).reshape(
                batch_size, num_tokens, -1
            ),
            "weights": weights.reshape(batch_size, num_tokens, -1),
            "top_indices": top_indices.reshape(batch_size, num_tokens, -1),
            "gating_logits": gating_logits.detach(),
            "history_context": history_context.detach(),
            "q_embedding": x_flat.detach(),
        }

        return final_output, routing_info


class GatingTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        context_dim: int,
        num_experts: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(context_dim, embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, num_experts)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        if key.size(0) == 0:
            return torch.zeros(
                query.size(0), self.out_proj.out_features, device=query.device
            )

        q = self.q_proj(query).unsqueeze(1)
        with torch.no_grad():
            k = self.k_proj(key).unsqueeze(0).expand(q.size(0), -1, -1)
            v = self.v_proj(value).unsqueeze(0).expand(q.size(0), -1, -1)

        attn_output, _ = self.attn(q, k, v)
        attn_output = self.norm(attn_output)
        return self.out_proj(attn_output).squeeze(1)
