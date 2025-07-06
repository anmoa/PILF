from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.experience_buffer import PrototypingExperienceBuffer

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
    def forward(self, x: torch.Tensor, experience_buffer: PrototypingExperienceBuffer | None = None, **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
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

    def forward(self, x: torch.Tensor, experience_buffer: PrototypingExperienceBuffer | None = None, **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
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

    def forward(self, x: torch.Tensor, experience_buffer: PrototypingExperienceBuffer | None = None, **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
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


class MemoryGaussianMoELayer(BaseMoELayer):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_experts: int,
        top_k: int = 2,
        gating_transformer_layers: int = 2,
        gating_transformer_heads: int = 4,
        history_size: int = 10,
        **kwargs,
    ):
        super().__init__(in_features, hidden_features, out_features, num_experts, top_k)
        self.history_size = history_size
        self.gating_transformer = GatingTransformer(
            embed_dim=in_features,
            num_heads=gating_transformer_heads,
            num_layers=gating_transformer_layers,
            num_experts=num_experts,
        )

    def forward(self, x: torch.Tensor, experience_buffer: PrototypingExperienceBuffer | None = None, **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if experience_buffer is None:
            raise ValueError("MemoryGaussianMoELayer requires an experience_buffer.")

        batch_size, num_tokens, in_features = x.shape
        x_token = x

        q_subject = x.mean(dim=1)
        retrieved_history = experience_buffer.retrieve_similar(q_subject, k=self.history_size)

        subject_dynamic_mus, subject_dynamic_log_sigmas = self.gating_transformer(
            tgt=q_subject.unsqueeze(1), memory=retrieved_history
        )

        dynamic_mus = subject_dynamic_mus.expand(-1, num_tokens, -1, -1)
        dynamic_log_sigmas = subject_dynamic_log_sigmas.expand(-1, num_tokens, -1, -1)

        x_flat = x_token.reshape(-1, in_features)
        dynamic_mus_flat = dynamic_mus.reshape(batch_size * num_tokens, self.num_experts, in_features)
        dynamic_log_sigmas_flat = dynamic_log_sigmas.reshape(batch_size * num_tokens, self.num_experts, in_features)

        dist = torch.distributions.Normal(dynamic_mus_flat, torch.exp(dynamic_log_sigmas_flat))
        log_probs = dist.log_prob(x_flat.unsqueeze(1)).sum(dim=-1)

        weights, top_indices = torch.topk(log_probs, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1,).to(x.dtype)

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
            "top_indices": top_indices.view(batch_size, num_tokens, -1),
            "q_embedding": q_subject.detach(),
            "gating_mus": dynamic_mus.detach(),
            "gating_log_sigmas": dynamic_log_sigmas.detach(),
        }

        return final_output, routing_info

    def meta_train(self, q_rehearsal: torch.Tensor, a_rehearsal: torch.Tensor, experience_buffer: PrototypingExperienceBuffer) -> torch.Tensor:
        retrieved_history = experience_buffer.retrieve_similar(q_rehearsal, k=self.history_size)
        
        q_rehearsal_unsqueezed = q_rehearsal.unsqueeze(1)
        
        dynamic_mus, dynamic_log_sigmas = self.gating_transformer(tgt=q_rehearsal_unsqueezed, memory=retrieved_history)
        
        final_mus = dynamic_mus.squeeze(1)
        final_log_sigmas = dynamic_log_sigmas.squeeze(1)

        q_rehearsal_expanded = q_rehearsal.unsqueeze(1).expand(-1, self.num_experts, -1)
        
        dist = torch.distributions.Normal(final_mus, torch.exp(final_log_sigmas))
        log_probs = dist.log_prob(q_rehearsal_expanded).sum(dim=-1)

        loss = F.binary_cross_entropy_with_logits(log_probs, a_rehearsal)
        
        return loss
