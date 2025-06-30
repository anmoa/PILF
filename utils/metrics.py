from typing import Dict, List, Optional

import torch

def calculate_gating_selection_accuracy(
    all_routing_info: List[Dict[str, torch.Tensor]],
    min_k_expert_indices: Dict[int, List[int]],
    top_k: int
) -> float:
    total_matched_experts = 0
    total_top_k_experts = 0
    
    all_top_indices = [info['top_indices'] for info in all_routing_info]
    
    for layer_idx, top_indices_layer in enumerate(all_top_indices):
        if layer_idx in min_k_expert_indices:
            min_k_indices_set = set(min_k_expert_indices[layer_idx])
            current_top_indices = set(top_indices_layer.cpu().flatten().tolist())
            
            matched_experts = len(min_k_indices_set.intersection(current_top_indices))
            total_matched_experts += matched_experts
            total_top_k_experts += len(current_top_indices)
            
    if total_top_k_experts > 0:
        return (total_matched_experts / total_top_k_experts) * 100.0
    return 0.0
