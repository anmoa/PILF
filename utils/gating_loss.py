from typing import Dict, List

import torch
import torch.nn as nn


class ConfidenceLoss(nn.Module):
    def forward(self, log_probs, top_indices):
        batch_size, num_tokens, num_experts = log_probs.shape
        
        flat_log_probs = log_probs.reshape(-1, num_experts)
        flat_top_indices = top_indices.reshape(-1, top_indices.shape[-1])
        
        gathered_log_probs = torch.gather(flat_log_probs, 1, flat_top_indices)
        
        loss = -gathered_log_probs.mean()
        
        return loss

class LoadBalancingLoss(nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, weights):
        batch_size, num_tokens, num_experts = weights.shape
        
        flat_weights = weights.reshape(-1, num_experts)
        
        tokens_per_expert = flat_weights.mean(dim=0)
        
        router_probs = flat_weights.sum(dim=0)
        router_probs = router_probs / router_probs.sum()
        
        loss = (self.num_experts * (tokens_per_expert * router_probs).sum())
        
        return loss

class GatingSelectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, log_probs: torch.Tensor, min_k_expert_indices: Dict[int, List[int]], layer_idx: int) -> torch.Tensor:
        batch_size, num_tokens, num_experts = log_probs.shape
        target_tensor = torch.zeros_like(log_probs, device=log_probs.device).reshape(-1, num_experts)

        if layer_idx in min_k_expert_indices:
            selected_expert_indices = torch.tensor(min_k_expert_indices[layer_idx], device=log_probs.device, dtype=torch.long)
            
            expert_mask = torch.zeros(num_experts, device=log_probs.device, dtype=torch.float32)
            expert_mask[selected_expert_indices] = 1.0
            
            target_tensor = expert_mask.unsqueeze(0).expand(batch_size * num_tokens, -1)

        loss = self.bce_loss(log_probs.reshape(-1, num_experts), target_tensor)
        return loss