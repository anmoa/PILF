from typing import Any, Dict
import torch.nn as nn
import torch.optim as optim

def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    learning_rate = config.get('learning_rate', 1e-3)
    weight_decay = config.get('weight_decay', 1e-4)

    param_groups = model.get_param_groups() if hasattr(model, 'get_param_groups') else [
        {'params': model.parameters()}
    ]
    
    for pg in param_groups:
        pg['lr'] = learning_rate
        pg['initial_lr'] = learning_rate
        
    return optim.AdamW(param_groups, lr=learning_rate, weight_decay=weight_decay)