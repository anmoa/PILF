from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gating_transformer import GatingTransformer


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
        x_flat = x.view(-1, in_features)

        gating_logits = self.gating(x_flat)
        weights, top_indices = torch.topk(gating_logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1).to(x.dtype)

        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            token_indices, expert_indices = (top_indices == i).nonzero(as_tuple=True)
            if token_indices.numel() > 0:
                tokens_for_expert = x_flat[token_indices]
                expert_output = expert(tokens_for_expert)
                y.index_add_(0, token_indices, (weights[token_indices, expert_indices, None] * expert_output))

        final_output = y.view(batch_size, num_tokens, -1)

        routing_info = {
            "gating_logits": gating_logits.view(batch_size, num_tokens, -1),
            "top_indices": top_indices.view(batch_size, num_tokens, -1),
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
        x_flat = x.view(-1, in_features)

        dist = torch.distributions.Normal(self.expert_mus, torch.exp(self.expert_log_sigmas))
        log_probs = dist.log_prob(x_flat.unsqueeze(1)).sum(dim=-1)
        
        weights, top_indices = torch.topk(log_probs, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            token_indices, expert_indices = (top_indices == i).nonzero(as_tuple=True)
            if token_indices.numel() > 0:
                tokens_for_expert = x_flat[token_indices]
                expert_output = expert(tokens_for_expert)
                y.index_add_(0, token_indices, (weights[token_indices, expert_indices, None] * expert_output))

        final_output = y.view(batch_size, num_tokens, -1)

        routing_info = {
            "log_probs": log_probs.view(batch_size, num_tokens, -1),
            "weights": weights.view(batch_size, num_tokens, -1),
            "top_indices": top_indices.view(batch_size, num_tokens, -1),
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
        **kwargs,
    ):
        super().__init__(
            in_features, hidden_features, out_features, num_experts, top_k
        )
        self.gating_transformer = GatingTransformer(
            in_features, num_heads, in_features, num_experts
        )
        self.raw_gating_proj = nn.Linear(in_features, num_experts)

    def forward(self, x: torch.Tensor, experience_buffer=None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch_size, num_tokens, in_features = x.shape
        x_flat = x.view(-1, in_features)

        raw_logits = self.raw_gating_proj(x_flat)
        
        history_context = torch.zeros_like(raw_logits)
        if experience_buffer and experience_buffer.current_size > 0:
            q_hist, _, _ = experience_buffer.sample(experience_buffer.current_size)
            
            if q_hist.numel() > 0:
                x_agg = x.mean(dim=1)
                
                full_sequence = torch.cat([q_hist, x_agg], dim=0)
                
                processed_sequence = self.gating_transformer(full_sequence.unsqueeze(0), full_sequence.unsqueeze(0), full_sequence.unsqueeze(0)).squeeze(0)
                
                history_context_agg = processed_sequence[q_hist.size(0):]
                
                history_context = history_context_agg.repeat_interleave(num_tokens, dim=0)

        gating_logits = raw_logits + history_context
        weights = F.softmax(gating_logits, dim=-1)
        weights_top_k, top_indices = torch.topk(weights, self.top_k, dim=-1)
        weights_top_k = weights_top_k / (weights_top_k.sum(dim=-1, keepdim=True) + 1e-9)

        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            token_indices, expert_indices = (top_indices == i).nonzero(as_tuple=True)
            if token_indices.numel() > 0:
                tokens_for_expert = x_flat[token_indices]
                expert_output = expert(tokens_for_expert)
                y.index_add_(0, token_indices, (weights_top_k[token_indices, expert_indices, None] * expert_output))

        final_output = y.view(batch_size, num_tokens, -1)

        routing_info = {
            "log_probs": F.log_softmax(gating_logits, dim=-1).view(batch_size, num_tokens, -1),
            "weights": weights.view(batch_size, num_tokens, -1),
            "top_indices": top_indices.view(batch_size, num_tokens, -1),
            "gating_logits": gating_logits.detach(),
            "q_embedding": x_flat.detach(),
        }

        return final_output, routing_info

    def meta_train(self, q_rehearsal: torch.Tensor, a_rehearsal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q_hist_processed = self.gating_transformer(q_rehearsal.unsqueeze(0), q_rehearsal.unsqueeze(0), q_rehearsal.unsqueeze(0)).squeeze(0)
        
        loss = F.cross_entropy(q_hist_processed, a_rehearsal, reduction='mean')
        
        td_error = F.cross_entropy(q_hist_processed.detach(), a_rehearsal, reduction='none')
        
        return loss, td_error
