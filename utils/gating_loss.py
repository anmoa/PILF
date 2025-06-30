import torch
import torch.nn as nn
from typing import List, Dict

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

class GatingSelectionAccuracyLoss(nn.Module):
    def __init__(self, top_k: int):
        super().__init__()
        self.top_k = top_k

    def forward(self, log_probs: torch.Tensor, min_k_expert_indices: Dict[int, List[int]]) -> torch.Tensor:
        total_matched_experts = 0
        total_top_k_experts = 0

        # Assuming log_probs is for a single layer here, or needs to be handled layer by layer
        # The structure of min_k_expert_indices is Dict[int, List[int]]
        # We need to align log_probs with the correct layer index
        
        # This implementation assumes a single layer for simplicity.
        # A more robust implementation would require layer_idx to be passed.
        if 0 in min_k_expert_indices:
            _, top_indices_layer = torch.topk(log_probs, self.top_k, dim=-1)
            
            min_k_indices = set(min_k_expert_indices[0])
            current_top_indices = set(top_indices_layer.cpu().flatten().tolist())
            
            matched_experts = len(min_k_indices.intersection(current_top_indices))
            
            total_matched_experts += matched_experts
            total_top_k_experts += self.top_k * log_probs.shape[0] * log_probs.shape[1]

        if total_top_k_experts > 0:
            accuracy = (total_matched_experts / total_top_k_experts) * 100.0
        else:
            accuracy = 0.0
            
        loss = 1.0 - (accuracy / 100.0)
        return torch.tensor(loss, dtype=torch.float32, device=log_probs.device, requires_grad=True)