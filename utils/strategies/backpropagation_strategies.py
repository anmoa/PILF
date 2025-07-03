from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from utils.pi_calculator import get_surprise_from_grads_torch
from utils.strategies.base_strategy import StrategyComponent
from utils.types import StepResult


def _get_active_indices(all_top_indices: List[torch.Tensor]) -> Dict[int, List[int]]:
    active_indices: Dict[int, List[int]] = {}
    for layer_idx, indices_tensor in enumerate(all_top_indices):
        active_indices[layer_idx] = torch.unique(indices_tensor.cpu()).tolist()
    return active_indices

class StandardStrategy(StrategyComponent):
    def apply(self, model: nn.Module, optimizer: optim.Optimizer, pi_metrics: Dict[str, Any], 
              all_gating_logits: Optional[Any] = None, all_top_indices: Optional[List[torch.Tensor]] = None) -> StepResult:
        metrics: StepResult = {}
        if all_top_indices is not None:
            metrics['active_expert_indices'] = _get_active_indices(all_top_indices)
        return metrics

class SelectiveUpdateStrategy(StrategyComponent):
    def apply(self, model: nn.Module, optimizer: optim.Optimizer, pi_metrics: Dict[str, Any], 
              all_gating_logits: Optional[Any] = None, all_top_indices: Optional[List[torch.Tensor]] = None) -> StepResult:
        if all_top_indices is None:
            raise ValueError("Selective update requires a model that returns top_indices.")
        
        model_module = model.module if isinstance(model, nn.DataParallel) else model
        if hasattr(model_module, 'zero_inactive_expert_grads'):
             model_module.zero_inactive_expert_grads(all_gating_logits)
        else:
            raise AttributeError("Model does not have 'zero_inactive_expert_grads' method required for SelectiveUpdate.")
        
        return StepResult(active_expert_indices=_get_active_indices(all_top_indices))

class SurpriseMinKStrategy(StrategyComponent):
    def __init__(self, min_k: int, **kwargs):
        super().__init__(**kwargs)
        self.min_k = min_k

    def apply(self, model: nn.Module, optimizer: optim.Optimizer, pi_metrics: Dict[str, Any], 
              all_gating_logits: Optional[Any] = None, all_top_indices: Optional[List[torch.Tensor]] = None) -> StepResult:
        if all_top_indices is None:
            raise ValueError("SurpriseMinKStrategy requires the model to return `all_top_indices`.")
        
        model_module = model.module if isinstance(model, nn.DataParallel) else model
        
        if not hasattr(model_module, 'blocks'):
            raise AttributeError("Model structure not supported. Expected `model.blocks`.")

        active_expert_indices: Dict[int, List[int]] = {}
        updated_expert_indices: Dict[int, List[int]] = {}

        for layer_idx, top_indices_layer in enumerate(all_top_indices):
            experts_in_layer = model_module.blocks[layer_idx].mlp.experts
            unique_expert_indices = torch.unique(top_indices_layer.cpu()).tolist()
            active_expert_indices[layer_idx] = unique_expert_indices

            if len(unique_expert_indices) <= self.min_k:
                updated_expert_indices[layer_idx] = unique_expert_indices
                continue

            expert_surprises = []
            for expert_idx in unique_expert_indices:
                expert = experts_in_layer[expert_idx]
                expert_grads = [p.grad for p in expert.parameters() if p.grad is not None]
                if not expert_grads:
                    continue
                surprise = get_surprise_from_grads_torch(expert_grads)
                expert_surprises.append((surprise.item(), expert_idx))
            
            expert_surprises.sort(key=lambda x: x[0])
            
            winning_expert_indices = [idx for _, idx in expert_surprises[:self.min_k]]
            losing_expert_indices = [idx for _, idx in expert_surprises[self.min_k:]]
            updated_expert_indices[layer_idx] = winning_expert_indices

            for expert_idx in losing_expert_indices:
                expert = experts_in_layer[expert_idx]
                for param in expert.parameters():
                    if param.grad is not None:
                        param.grad.zero_()
        
        if hasattr(model_module, 'zero_inactive_expert_grads'):
            model_module.zero_inactive_expert_grads(all_gating_logits)
        
        return StepResult(
            active_expert_indices=active_expert_indices,
            updated_expert_indices=updated_expert_indices,
            min_k_expert_indices=updated_expert_indices, 
        )
