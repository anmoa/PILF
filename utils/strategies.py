import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from sigma_pi import get_surprise_from_grads_torch

def pisa_modulation(surprise: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, power: float, exponent: float) -> torch.Tensor:
    """
    Calculates the learning rate modulation factor based on surprise.
    This can be a Gaussian, exponential, or other curve based on power and exponent.
    - power: Scaling factor for the exponentiation (e.g., 0.5 for Gaussian).
    - exponent: The power to which the normalized surprise is raised (e.g., 2.0 for Gaussian).
    """
    # The 'abs' ensures it works correctly for any positive exponent.
    return torch.exp(-power * (torch.abs(surprise - mu) / (sigma + 1e-9)) ** exponent)

class PisaAdaptor:
    """
    Predictive Integrity-driven Sigma Adaptor (PISA).
    Tracks the second-order statistics of surprise to dynamically adapt sigma.
    """
    def __init__(self, device: torch.device, initial_var: float = 1.0, beta: float = 0.1, **kwargs):
        self.device = device
        self.alpha = 0.1  # Default value for mean EMA
        self.beta = beta    # EMA factor for variance
        self.initial_var = initial_var
        self.reset_state()

    def reset_state(self):
        self.ema_mu = torch.tensor(0.0, device=self.device)
        self.ema_var = torch.tensor(self.initial_var, device=self.device)

    def step(self, surprise_value: float):
        surprise = torch.tensor(surprise_value, device=self.device)
        self.ema_mu = (1 - self.alpha) * self.ema_mu + self.alpha * surprise
        self.ema_var = (1 - self.beta) * self.ema_var + self.beta * (surprise - self.ema_mu) ** 2
    
    def get_sigma(self) -> torch.Tensor:
        return torch.sqrt(self.ema_var + 1e-9)

class UpdateStrategy(ABC):
    def __init__(self, optimizer: optim.Optimizer, **kwargs):
        self.optimizer = optimizer

    @abstractmethod
    def step(self, model: nn.Module, loss: nn.Module, pi_metrics: Dict[str, Any], all_gating_logits: Optional[Any] = None, all_top_indices: Optional[List[torch.Tensor]] = None) -> Dict[str, Any]:
        pass

class StandardUpdate(UpdateStrategy):
    def step(self, model: nn.Module, loss: nn.Module, pi_metrics: Dict[str, Any], all_gating_logits: Optional[Any] = None, all_top_indices: Optional[List[torch.Tensor]] = None) -> Dict[str, Any]:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        self.optimizer.step()
        return {}

class SelectiveUpdate(UpdateStrategy):
    def step(self, model: nn.Module, loss: nn.Module, pi_metrics: Dict[str, Any], all_gating_logits: Optional[Any] = None, all_top_indices: Optional[List[torch.Tensor]] = None) -> Dict[str, Any]:
        if all_top_indices is None:
            raise ValueError("Selective update requires a model that returns top_indices.")
        
        model_module = model.module if isinstance(model, nn.DataParallel) else model
        if hasattr(model_module, 'zero_inactive_expert_grads'):
             model_module.zero_inactive_expert_grads(all_top_indices)
        else:
            raise AttributeError("Model does not have 'zero_inactive_expert_grads' method required for SelectiveUpdate.")

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        self.optimizer.step()
        return {}

class SurpriseMinKUpdate(UpdateStrategy):
    def __init__(self, optimizer: optim.Optimizer, min_k: int, **kwargs):
        super().__init__(optimizer, **kwargs)
        self.min_k = min_k

    def step(self, model: nn.Module, loss: nn.Module, pi_metrics: Dict[str, Any], all_gating_logits: Optional[Any] = None, all_top_indices: Optional[List[torch.Tensor]] = None) -> Dict[str, Any]:
        if all_top_indices is None:
            raise ValueError("SurpriseMinKUpdate requires the model to return `all_top_indices`.")

        model_module = model.module if isinstance(model, nn.DataParallel) else model
        
        if not hasattr(model_module, 'blocks'):
            raise AttributeError("Model structure not supported. Expected `model.blocks`.")

        for layer_idx, top_indices_layer in enumerate(all_top_indices):
            # This logic assumes MoE layers are within a list/sequential named 'blocks'
            experts_in_layer = model_module.blocks[layer_idx].mlp.experts
            unique_expert_indices = torch.unique(top_indices_layer).tolist()

            if len(unique_expert_indices) <= self.min_k:
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
            
            losing_experts_indices = [idx for _, idx in expert_surprises[self.min_k:]]

            for expert_idx in losing_experts_indices:
                expert = experts_in_layer[expert_idx]
                for param in expert.parameters():
                    if param.grad is not None:
                        param.grad.zero_()
        
        model_module.zero_inactive_expert_grads(all_top_indices)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        self.optimizer.step()
        return {}

class PisaUpdate(UpdateStrategy):
    def __init__(self, optimizer: optim.Optimizer, device: torch.device, crisis_threshold: Optional[float] = None, **kwargs):
        super().__init__(optimizer, **kwargs)
        self.device = device
        self.param_groups = {pg.get('name', 'base'): pg for pg in self.optimizer.param_groups}
        self.base_lrs = {name: pg['lr'] for name, pg in self.param_groups.items()}
        
        self.alpha = torch.tensor(kwargs.get('alpha', 1.0), device=self.device)
        self.gamma = torch.tensor(kwargs.get('gamma', 0.5), device=self.device)
        self.crisis_threshold = crisis_threshold
        self.kwargs = kwargs

        # Check for dual PISA mode
        is_dual_mode = 'gating' in self.param_groups and 'experts' in self.param_groups
        self.dual_mode = is_dual_mode and ('gating_initial_var' in kwargs or 'expert_initial_var' in kwargs)

        if self.dual_mode:
            expert_kwargs = {k.replace('expert_', ''): v for k, v in kwargs.items() if k.startswith('expert_')}
            gating_kwargs = {k.replace('gating_', ''): v for k, v in kwargs.items() if k.startswith('gating_')}
            self.pisa_expert = PisaAdaptor(device, **expert_kwargs)
            self.pisa_gating = PisaAdaptor(device, **gating_kwargs)
        else:
            self.pisa_adaptor = PisaAdaptor(device, **kwargs)

    def reset_state(self):
        if self.dual_mode:
            self.pisa_expert.reset_state()
            self.pisa_gating.reset_state()
        else:
            self.pisa_adaptor.reset_state()

    def step(self, model: nn.Module, loss: nn.Module, pi_metrics: Dict[str, Any], all_gating_logits: Optional[Any] = None, all_top_indices: Optional[List[torch.Tensor]] = None) -> Dict[str, Any]:
        if all_gating_logits is None:
            raise ValueError("PISA update requires a model that returns gating logits.")
        
        if self.dual_mode:
            return self._step_dual(model, pi_metrics)
        else:
            return self._step_single(model, pi_metrics)

    def _step_single(self, model: nn.Module, pi_metrics: Dict[str, Any]) -> Dict[str, Any]:
        surprise = torch.tensor(pi_metrics['surprise'], device=self.device)
        
        if self.crisis_threshold is not None and surprise.item() >= self.crisis_threshold:
            return {'lr_mod': 0.0, 'sigma': self.pisa_adaptor.get_sigma().item(), 'decision': 'REJECT'}

        self.pisa_adaptor.step(surprise.item())
        
        sigma = self.pisa_adaptor.get_sigma()
        mu = self.pisa_adaptor.ema_mu
        
        modulation_kwargs = {
            'power': float(self.kwargs.get('modulation_power', 0.5)),
            'exponent': float(self.kwargs.get('modulation_exponent', 2.0))
        }
        lr_modulation = pisa_modulation(surprise, mu, sigma, **modulation_kwargs)
        
        for name, pg in self.param_groups.items():
            pg['lr'] = self.base_lrs[name] * lr_modulation
        
        self.optimizer.step()
        
        for name, pg in self.param_groups.items():
            pg['lr'] = self.base_lrs[name]
            
        return {'lr_mod': lr_modulation.item(), 'sigma': sigma.item(), 'decision': 'CONSOLIDATE'}

    def _step_dual(self, model: nn.Module, pi_metrics: Dict[str, Any]) -> Dict[str, Any]:
        gating_grads = [p.grad for p in self.param_groups['gating']['params'] if p.grad is not None]
        expert_grads = [p.grad for p in self.param_groups['experts']['params'] if p.grad is not None]

        gating_surprise = get_surprise_from_grads_torch(gating_grads)
        expert_surprise = get_surprise_from_grads_torch(expert_grads)

        total_surprise = torch.tensor(pi_metrics['surprise'], device=self.device)
        if self.crisis_threshold is not None and total_surprise.item() >= self.crisis_threshold:
            # Still calculate PI metrics for logging even if not updating
            tau = torch.tensor(pi_metrics.get('tau', 0.0), device=self.device)
            epsilon = torch.tensor(pi_metrics.get('epsilon', 0.0), device=self.device)
            normalized_error = epsilon / (tau + 1e-9)
            cognitive_cost = (1 - self.gamma) * normalized_error + self.gamma * total_surprise
            pi_score = torch.exp(-self.alpha * cognitive_cost)
            return {
                'gating_lr_mod': 0.0,
                'gating_sigma': self.pisa_gating.get_sigma().item(),
                'expert_lr_mod': 0.0,
                'expert_sigma': self.pisa_expert.get_sigma().item(),
                'pi_score': pi_score.item(),
                'surprise': total_surprise.item(),
                'decision': 'REJECT'
            }

        self.pisa_gating.step(gating_surprise.item())
        self.pisa_expert.step(expert_surprise.item())

        sigma_gating = self.pisa_gating.get_sigma()
        sigma_expert = self.pisa_expert.get_sigma()

        mu_gating = self.pisa_gating.ema_mu
        mu_expert = self.pisa_expert.ema_mu

        gating_modulation_kwargs = {
            'power': float(self.kwargs.get('gating_modulation_power', self.kwargs.get('modulation_power', 0.5))),
            'exponent': float(self.kwargs.get('gating_modulation_exponent', self.kwargs.get('modulation_exponent', 2.0)))
        }
        expert_modulation_kwargs = {
            'power': float(self.kwargs.get('expert_modulation_power', 0.0)),
            'exponent': float(self.kwargs.get('expert_modulation_exponent', 1.0))
        }

        lr_mod_gating = pisa_modulation(gating_surprise, mu_gating, sigma_gating, **gating_modulation_kwargs)
        lr_mod_expert = pisa_modulation(expert_surprise, mu_expert, sigma_expert, **expert_modulation_kwargs)

        self.param_groups['gating']['lr'] = self.base_lrs['gating'] * lr_mod_gating
        self.param_groups['experts']['lr'] = self.base_lrs['experts'] * lr_mod_expert
        
        self.optimizer.step()

        for name, pg in self.param_groups.items():
            if name in self.base_lrs:
                pg['lr'] = self.base_lrs[name]

        # Calculate global metrics for monitoring
        tau = torch.tensor(pi_metrics.get('tau', 0.0), device=self.device)
        epsilon = torch.tensor(pi_metrics.get('epsilon', 0.0), device=self.device)
        
        normalized_error = epsilon / (tau + 1e-9)
        cognitive_cost = (1 - self.gamma) * normalized_error + self.gamma * total_surprise
        pi_score = torch.exp(-self.alpha * cognitive_cost)

        return {
            'gating_lr_mod': lr_mod_gating.item(),
            'gating_sigma': sigma_gating.item(),
            'expert_lr_mod': lr_mod_expert.item(),
            'expert_sigma': sigma_expert.item(),
            'pi_score': pi_score.item(),
            'surprise': total_surprise.item(),
            'decision': 'CONSOLIDATE'
        }
