from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from utils.logging.types import StepResult


class StrategyComponent:
    def __init__(self, **kwargs):
        pass

    def apply(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        pi_metrics: Dict[str, Any],
        all_gating_logits: Optional[Any] = None,
        all_top_indices: Optional[List[torch.Tensor]] = None,
        activated_experts: Optional[Dict[int, List[int]]] = None,
    ) -> StepResult:
        raise NotImplementedError
