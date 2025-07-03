from typing import Any, Dict

import torch.nn as nn

from .base_vit import VisionTransformer
from .g2_moe import GenGaussianMoEVisionTransformer
from .gaussian_moe import GaussianMoEVisionTransformer
from .moe_vit import MoEVisionTransformer

__all__ = ["VisionTransformer", "MoEVisionTransformer", "GaussianMoEVisionTransformer", "GenGaussianMoEVisionTransformer", "create_model"]

def create_model(model_config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create a model from a configuration dictionary.
    """
    model_config = model_config.copy() # Avoid modifying the original config dict
    model_type = model_config.pop('model_type')
    
    model_map = {
        'dense': VisionTransformer,
        'moe': MoEVisionTransformer,
        'gaussian_moe': GaussianMoEVisionTransformer,
        'gen_gaussian_moe': GenGaussianMoEVisionTransformer,
    }
    
    model_cls = model_map.get(model_type)
    if not model_cls:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return model_cls(**model_config)
