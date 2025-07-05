from typing import Dict, List, Optional, TypedDict

import torch


class StepResult(TypedDict, total=False):
    loss: float
    accuracy: float
    routing_accuracy: float
    task_name: str
    global_step: int
    pi_score: float
    surprise: float
    tau: float
    gating_tau: float
    router_surprise: float
    gating_loss: float
    gating_selection_accuracy: float
    historical_routing_loss: float
    lr_mod: float
    gating_lr_mod: float
    expert_lr_mod: float
    sigma: float
    gating_sigma: float
    expert_sigma: float
    decision: str
    active_expert_indices: Optional[Dict[int, List[int]]]
    updated_expert_indices: Optional[Dict[int, List[int]]]
    surprise_min_k_expert_indices: Optional[Dict[int, List[int]]]
    top_k_expert_indices: Optional[Dict[int, List[int]]]
    least_active_expert_indices: Optional[Dict[int, List[int]]]
    all_log_probs: List[torch.Tensor]
    all_top_indices: List[torch.Tensor]


class ValidationResult(TypedDict, total=False):
    global_step: int
    epoch: int
    loss: float
    accuracy: float
    pi_score: float
    surprise: float
    tau: float
