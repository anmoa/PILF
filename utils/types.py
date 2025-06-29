from typing import Dict, List, Optional, Tuple, TypedDict

import torch


class StepResult(TypedDict, total=False):
    """
    A unified data container for all possible metrics collected during a training step.
    This serves as the single source of truth for data passed to logging and plotting.
    All fields are Optional because not all metrics are available in every step or model.
    """
    # --- Core Metrics (always present) ---
    loss: float
    accuracy: float
    task_name: str
    global_step: int

    # --- PI-related Metrics (from SigmaPI) ---
    pi_score: float
    surprise: float
    tau: float
    gating_tau: float

    # --- PISA/PILR-S Strategy Metrics ---
    lr_mod: float
    gating_lr_mod: float
    expert_lr_mod: float
    sigma: float
    gating_sigma: float
    expert_sigma: float
    decision: str

    # --- MoE-related Metrics ---
    active_expert_indices: Dict[int, List[int]]
    updated_expert_indices: Dict[int, List[int]]
    all_log_probs: List[torch.Tensor]
    all_top_indices: List[torch.Tensor]


# Type alias for the return value of the validate function
ValidationResult = Tuple[float, float, float, float, float, Optional[float]]
