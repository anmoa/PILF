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

class TopKMinKLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, log_probs: torch.Tensor, top_k_indices: torch.Tensor, min_k_expert_indices: Dict[int, List[int]], layer_idx: int) -> torch.Tensor:
        batch_size, num_tokens, num_experts = log_probs.shape
        flat_log_probs = log_probs.reshape(-1, num_experts)
        
        # Create target distribution for min_k experts
        min_k_target_dist = torch.full_like(flat_log_probs, -float('inf')) # Initialize with negative infinity for log_target=True
        if layer_idx in min_k_expert_indices:
            selected_expert_indices = torch.tensor(min_k_expert_indices[layer_idx], device=log_probs.device, dtype=torch.long)
            
            # Distribute probability mass equally among min_k experts
            num_min_k = len(selected_expert_indices)
            if num_min_k > 0:
                min_k_target_dist.scatter_(1, selected_expert_indices.unsqueeze(0).expand(flat_log_probs.shape[0], -1), torch.log(torch.tensor(1.0 / num_min_k, device=log_probs.device)))
            
        # Calculate KL divergence between current log_probs and min_k target distribution
        loss = self.kl_div_loss(flat_log_probs, min_k_target_dist)
        return loss
