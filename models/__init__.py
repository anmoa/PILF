from typing import Any, Dict

import torch.nn as nn

from .base_vit import VisionTransformer

__all__ = ["VisionTransformer", "create_model"]


def create_model(model_config: Dict[str, Any]) -> nn.Module:
    config = model_config.copy()
    router_type = config.pop("router_type", "dense")
    return VisionTransformer(router_type=router_type, **config)
