from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from sigma_pi import get_surprise_from_grads_torch

from utils.types import StepResult


def _get_active_indices(all_top_indices: List[torch.Tensor]) -> Dict[int, List[int]]:
    """Converts a list of expert index tensors to a dictionary of unique active indices per layer."""
    active_indices: Dict[int, List[int]] = {}
    for layer_idx, indices_tensor in enumerate(all_top_indices):
        active_indices[layer_idx] = torch.unique(indices_tensor.cpu()).tolist()
    return active_indices

def pilr_modulation(surprise: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, power: float, exponent: float) -> torch.Tensor:
    """
    Calculates the learning rate modulation factor based on surprise.
    This can be a Gaussian, exponential, or other curve based on power and exponent.
    """
    return torch.exp(-power * (torch.abs(surprise - mu) / (sigma + 1e-9)) ** exponent)

class PILRAdaptor:
    """
    Predictive Integrity-driven Learning Rate Adaptor.
    Tracks the second-order statistics of surprise to dynamically adapt sigma.
    """
    def __init__(self, device: torch.device, initial_var: float = 1.0, beta: float = 0.1, **kwargs):
        self.device = device
        self.alpha = 0.1
        self.beta = beta
        self.initial_var = initial_var
        self.reset_state()

    def reset_state(self, mu: Optional[float] = None, var: Optional[float] = None):
        self.ema_mu = torch.tensor(mu if mu is not None else 0.0, device=self.device)
        self.ema_var = torch.tensor(var if var is not None else self.initial_var, device=self.device)

    def step(self, surprise_value: float):
        surprise = torch.tensor(surprise_value, device=self.device)
        self.ema_mu = (1 - self.alpha) * self.ema_mu + self.alpha * surprise
        self.ema_var = (1 - self.beta) * self.ema_var + self.beta * (surprise - self.ema_mu) ** 2
    
    def get_sigma(self) -> torch.Tensor:
        return torch.sqrt(self.ema_var + 1e-9)

class StrategyComponent:
    """
    Base class for strategy components.
    Each component should implement a `apply` method that takes relevant inputs
    and returns a dictionary of metrics/updates.
    """
    def __init__(self, **kwargs):
        pass

    def apply(self, model: nn.Module, optimizer: optim.Optimizer, pi_metrics: Dict[str, Any], 
              all_gating_logits: Optional[Any] = None, all_top_indices: Optional[List[torch.Tensor]] = None) -> StepResult:
        raise NotImplementedError

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
             model_module.zero_inactive_expert_grads(all_top_indices)
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
            model_module.zero_inactive_expert_grads(all_top_indices)
        
        return StepResult(
            active_expert_indices=active_expert_indices,
            updated_expert_indices=updated_expert_indices,
        )

class PILRStrategy(StrategyComponent):
    def __init__(self, device: torch.device, crisis_threshold: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.crisis_threshold = crisis_threshold
        self.kwargs = kwargs

        self.warmup_steps = kwargs.get('warmup_steps', 0)
        self.step_count = 0

        is_dual_mode = kwargs.get('gating_initial_var') is not None and kwargs.get('expert_initial_var') is not None
        self.dual_mode = is_dual_mode

        if self.dual_mode:
            expert_kwargs = {k.replace('expert_', ''): v for k, v in kwargs.items() if k.startswith('expert_')}
            gating_kwargs = {k.replace('gating_', ''): v for k, v in kwargs.items() if k.startswith('gating_')}
            
            self.gating_beta_normal = gating_kwargs.get('beta', 0.1)
            self.gating_beta_warmup = gating_kwargs.get('beta_warmup', self.gating_beta_normal)

            self.pilr_expert = PILRAdaptor(device, **expert_kwargs)
            self.pilr_gating = PILRAdaptor(device, **gating_kwargs)
        else:
            self.pilr_adaptor = PILRAdaptor(device, **kwargs)

    def reset_state(self):
        self.step_count = 0
        if self.dual_mode:
            self.pilr_expert.reset_state()
            self.pilr_gating.reset_state()
        else:
            self.pilr_adaptor.reset_state()

    def apply(self, model: nn.Module, optimizer: optim.Optimizer, pi_metrics: Dict[str, Any], 
              all_gating_logits: Optional[Any] = None, all_top_indices: Optional[List[torch.Tensor]] = None) -> StepResult:
        if all_gating_logits is None:
            raise ValueError("PILRStrategy requires a model that returns gating logits.")
        
        result: StepResult
        if self.dual_mode:
            result = self._apply_dual(model, optimizer, pi_metrics)
        else:
            result = self._apply_single(model, optimizer, pi_metrics)

        return result

    def _apply_single(self, model: nn.Module, optimizer: optim.Optimizer, pi_metrics: Dict[str, Any]) -> StepResult:
        surprise = torch.tensor(pi_metrics['surprise'], device=self.device)
        
        if self.crisis_threshold is not None and surprise.item() >= self.crisis_threshold:
            return StepResult(lr_mod=0.0, sigma=self.pilr_adaptor.get_sigma().item(), decision='REJECT')

        self.pilr_adaptor.step(surprise.item())
        
        sigma = self.pilr_adaptor.get_sigma()
        mu = self.pilr_adaptor.ema_mu
        
        modulation_kwargs = {
            'power': float(self.kwargs.get('modulation_power', 0.5)),
            'exponent': float(self.kwargs.get('modulation_exponent', 2.0))
        }
        lr_modulation = pilr_modulation(surprise, mu, sigma, **modulation_kwargs)
        
        for pg in optimizer.param_groups:
            if pg.get('name') == 'base': # Only modulate the base LR group
                pg['lr'] = pg['initial_lr'] * lr_modulation
            
        return StepResult(lr_mod=lr_modulation.item(), sigma=sigma.item(), decision='CONSOLIDATE')

    def _apply_dual(self, model: nn.Module, optimizer: optim.Optimizer, pi_metrics: Dict[str, Any]) -> StepResult:
        # Access param groups by name, assuming they are set up in _get_optimizer
        gating_param_group = next((pg for pg in optimizer.param_groups if pg.get('name') == 'gating'), None)
        expert_param_group = next((pg for pg in optimizer.param_groups if pg.get('name') == 'experts'), None)

        if gating_param_group is None or expert_param_group is None:
            raise ValueError("Optimizer must have 'gating' and 'experts' parameter groups for dual PILR strategy.")

        gating_grads = [p.grad for p in gating_param_group['params'] if p.grad is not None]
        expert_grads = [p.grad for p in expert_param_group['params'] if p.grad is not None]

        gating_surprise = get_surprise_from_grads_torch(gating_grads)
        expert_surprise = get_surprise_from_grads_torch(expert_grads)

        total_surprise = torch.tensor(pi_metrics['surprise'], device=self.device)
        if self.crisis_threshold is not None and total_surprise.item() >= self.crisis_threshold:
            pi_score = torch.tensor(pi_metrics.get('pi_score', 0.0), device=self.device)
            return StepResult(
                gating_lr_mod=0.0,
                gating_sigma=self.pilr_gating.get_sigma().item(),
                expert_lr_mod=0.0,
                expert_sigma=self.pilr_expert.get_sigma().item(),
                pi_score=pi_score.item(),
                surprise=total_surprise.item(),
                decision='REJECT'
            )

        if self.step_count < self.warmup_steps:
            self.pilr_gating.beta = self.gating_beta_warmup
        else:
            self.pilr_gating.beta = self.gating_beta_normal
        
        self.pilr_gating.step(gating_surprise.item())
        self.pilr_expert.step(expert_surprise.item())

        self.step_count += 1

        sigma_gating = self.pilr_gating.get_sigma()
        sigma_expert = self.pilr_expert.get_sigma()

        mu_gating = self.pilr_gating.ema_mu
        mu_expert = self.pilr_expert.ema_mu

        gating_modulation_kwargs = {
            'power': float(self.kwargs.get('gating_modulation_power', 0.5)),
            'exponent': float(self.kwargs.get('gating_modulation_exponent', 2.0))
        }
        expert_modulation_kwargs = {
            'power': float(self.kwargs.get('expert_modulation_power', 0.0)),
            'exponent': float(self.kwargs.get('expert_modulation_exponent', 1.0))
        }

        lr_mod_gating = pilr_modulation(gating_surprise, mu_gating, sigma_gating, **gating_modulation_kwargs)
        lr_mod_expert = pilr_modulation(expert_surprise, mu_expert, sigma_expert, **expert_modulation_kwargs)

        gating_param_group['lr'] = gating_param_group['initial_lr'] * lr_mod_gating
        expert_param_group['lr'] = expert_param_group['initial_lr'] * lr_mod_expert
        
        pi_score = torch.tensor(pi_metrics.get('pi_score', 0.0), device=self.device)

        return StepResult(
            gating_lr_mod=lr_mod_gating.item(),
            gating_sigma=sigma_gating.item(),
            expert_lr_mod=lr_mod_expert.item(),
            expert_sigma=sigma_expert.item(),
            pi_score=pi_score.item(),
            surprise=total_surprise.item(),
            decision='CONSOLIDATE'
        )
