from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

if TYPE_CHECKING:
    pass
from utils.logging.types import StepResult

from .base_strategy import StrategyComponent


class SurpriseMinKStrategy(StrategyComponent):
    def __init__(self, min_k: int = 1, **kwargs):
        super().__init__(**kwargs)
        if not min_k >= 1:
            raise ValueError("min_k must be at least 1")
        self.min_k = min_k

    def apply(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        pi_metrics: Dict[str, Any],
        all_gating_logits: Optional[Any] = None,
        all_top_indices: Optional[List[torch.Tensor]] = None,
        activated_experts: Optional[Dict[int, List[int]]] = None,
    ) -> StepResult:
        
        min_k_expert_indices: Dict[int, List[int]] = {}
        
        moe_blocks = [
            block for block in model.blocks
            if hasattr(block, "mlp") and "BaseMoELayer" in [c.__name__ for c in block.mlp.__class__.__mro__]
        ]

        for i, block in enumerate(moe_blocks):
            moe_layer = block.mlp
            if not hasattr(moe_layer, 'experts'):
                continue

            min_k = self.min_k
            
            expert_surprises = []
            for expert_idx, expert in enumerate(moe_layer.experts):
                grads = [p.grad for p in expert.parameters() if p.grad is not None]
                if not grads:
                    surprise = float('inf')
                else:
                    total_norm_sq = torch.stack([torch.sum(g.pow(2)) for g in grads]).sum()
                    surprise = torch.sqrt(total_norm_sq).item()
                expert_surprises.append((surprise, expert_idx))

            expert_surprises.sort(key=lambda x: x[0])
            
            min_k_indices = {idx for _, idx in expert_surprises[:min_k]}
            min_k_expert_indices[i] = sorted(list(min_k_indices))

            for expert_idx, expert in enumerate(moe_layer.experts):
                if expert_idx not in min_k_indices:
                    for param in expert.parameters():
                        if param.grad is not None:
                            param.grad.zero_()
        
        return {"surprise_min_k_expert_indices": min_k_expert_indices}


class ActivatedSurpriseMinKStrategy(SurpriseMinKStrategy):
    def apply(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        pi_metrics: Dict[str, Any],
        all_gating_logits: Optional[Any] = None,
        all_top_indices: Optional[List[torch.Tensor]] = None,
        activated_experts: Optional[Dict[int, List[int]]] = None,
    ) -> StepResult:
        if activated_experts is None:
            return {}

        min_k_expert_indices: Dict[int, List[int]] = {}
        
        moe_blocks = [
            block for block in model.blocks
            if hasattr(block, "mlp") and "BaseMoELayer" in [c.__name__ for c in block.mlp.__class__.__mro__]
        ]

        for i, block in enumerate(moe_blocks):
            if i not in activated_experts:
                continue
            
            moe_layer = block.mlp
            active_expert_indices_for_layer = set(activated_experts[i])
            
            expert_surprises = []
            for expert_idx in active_expert_indices_for_layer:
                expert = moe_layer.experts[expert_idx]
                grads = [p.grad for p in expert.parameters() if p.grad is not None]
                if not grads:
                    surprise = float('inf')
                else:
                    total_norm_sq = torch.stack([torch.sum(g.pow(2)) for g in grads]).sum()
                    surprise = torch.sqrt(total_norm_sq).item()
                expert_surprises.append((surprise, expert_idx))

            expert_surprises.sort(key=lambda x: x[0])
            
            min_k_indices = {idx for _, idx in expert_surprises[:self.min_k]}
            min_k_expert_indices[i] = sorted(list(min_k_indices))

            for expert_idx in active_expert_indices_for_layer:
                if expert_idx not in min_k_indices:
                    for param in moe_layer.experts[expert_idx].parameters():
                        if param.grad is not None:
                            param.grad.zero_()
        
        return {"surprise_min_k_expert_indices": min_k_expert_indices}