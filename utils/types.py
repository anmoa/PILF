from typing import TypedDict, List, Optional, Dict, Tuple

import torch

# PISA/PILR-S 相关
class PisaMetrics(TypedDict, total=False):
    lr_mod: float
    gating_lr_mod: float
    expert_lr_mod: float
    sigma: float
    gating_sigma: float
    expert_sigma: float
    decision: str

# MoE 路由与更新相关
class MoEMetrics(TypedDict, total=False):
    # 每层激活的专家索引 (来自 Top-K)
    active_expert_indices: Dict[int, List[int]]
    # 经过 Surprise 筛选后，实际更新的专家索引
    updated_expert_indices: Dict[int, List[int]]
    # 路由置信度
    all_log_probs: List[torch.Tensor]
    # Top-K 索引
    all_top_indices: List[torch.Tensor]


# 核心指标
class CoreMetrics(TypedDict):
    loss: float
    accuracy: float
    pi_score: float
    surprise: float
    tau: float
    gating_tau: Optional[float]

class StepResult(CoreMetrics, PisaMetrics, MoEMetrics):
    """
    统一的数据容器，用于在训练步骤中收集和传递所有相关指标。
    """
    pass

# 用于 validate 函数返回值的类型
ValidationResult = Tuple[float, float, float, float, float, float]
