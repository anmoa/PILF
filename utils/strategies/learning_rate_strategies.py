from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from utils.strategies.base_strategy import StrategyComponent
from utils.types import StepResult


def pilr_modulation(surprise: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, power: float, exponent: float) -> torch.Tensor:
    return torch.exp(-power * (torch.abs(surprise - mu) / (sigma + 1e-9)) ** exponent)

class PILRAdaptor:
    def __init__(self, device: torch.device, initial_var: float = 1.0, beta: float = 0.1, inverse_ema: bool = False, inverse_ema_k: float = 0.1, **kwargs):
        self.device = device
        self.beta = beta
        self.initial_var = initial_var
        self.inverse_ema = inverse_ema
        self.inverse_ema_k = inverse_ema_k
        self.reset_state()

    def reset_state(self, mu: Optional[float] = None, var: Optional[float] = None):
        self.ema_mu = torch.tensor(mu if mu is not None else 0.0, device=self.device)
        self.ema_var = torch.tensor(var if var is not None else self.initial_var, device=self.device)

    def step(self, surprise_value: float):
        surprise = torch.tensor(surprise_value, device=self.device)
        self.ema_mu = (1 - self.beta) * self.ema_mu + self.beta * surprise # Use beta for EMA of mu as well
        
        if self.inverse_ema:
            deviation = torch.abs(surprise - self.ema_mu)
            target_var = torch.exp(-self.inverse_ema_k * deviation)
            self.ema_var = (1 - self.beta) * self.ema_var + self.beta * target_var
        else:
            self.ema_var = (1 - self.beta) * self.ema_var + self.beta * (surprise - self.ema_mu) ** 2
    
    def get_sigma(self) -> torch.Tensor:
        return torch.sqrt(self.ema_var + 1e-9)

class PILRStrategy(StrategyComponent):
    def __init__(self, device: torch.device, crisis_threshold: Optional[float] = None, **kwargs):
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
            if pg.get('name') == 'base':
                pg['lr'] = pg['initial_lr'] * lr_modulation
            
        return StepResult(lr_mod=lr_modulation.item(), sigma=sigma.item(), decision='CONSOLIDATE')

    def _apply_dual(self, model: nn.Module, optimizer: optim.Optimizer, pi_metrics: Dict[str, Any]) -> StepResult:
        gating_param_group = next((pg for pg in optimizer.param_groups if pg.get('name') == 'gating'), None)
        expert_param_group = next((pg for pg in optimizer.param_groups if pg.get('name') == 'experts'), None)

        if gating_param_group is None or expert_param_group is None:
            raise ValueError("Optimizer must have 'gating' and 'experts' parameter groups for dual PILR strategy.")

        gating_surprise = torch.tensor(pi_metrics.get('gating_surprise', 0.0), device=self.device)
        expert_surprise = torch.tensor(pi_metrics.get('expert_surprise', 0.0), device=self.device)

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
