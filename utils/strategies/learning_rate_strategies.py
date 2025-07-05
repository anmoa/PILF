from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from utils.strategies.base_strategy import StrategyComponent

from ..logging.types import StepResult


def pilr_modulation(
    surprise: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    power: float,
    exponent: float,
) -> torch.Tensor:
    return torch.exp(-power * (torch.abs(surprise - mu) / (sigma + 1e-9)) ** exponent)


class PILRAdaptor:
    def __init__(
        self,
        device: torch.device,
        initial_var: float = 1.0,
        beta: float = 0.1,
        inverse_ema: bool = False,
        inverse_ema_k: float = 0.1,
        **kwargs,
    ):
        self.device = device
        self.beta = beta
        self.initial_var = initial_var
        self.inverse_ema = inverse_ema
        self.inverse_ema_k = inverse_ema_k
        self.reset_state()

    def reset_state(self, mu: Optional[float] = None, var: Optional[float] = None):
        self.ema_mu = torch.tensor(mu if mu is not None else 0.0, device=self.device)
        self.ema_var = torch.tensor(
            var if var is not None else self.initial_var, device=self.device
        )

    def step(self, surprise_value: torch.Tensor):
        surprise = surprise_value.to(self.device)
        self.ema_mu = (1 - self.beta) * self.ema_mu + self.beta * surprise
        if self.inverse_ema:
            deviation = torch.abs(surprise - self.ema_mu)
            target_var = torch.exp(-self.inverse_ema_k * deviation)
            self.ema_var = (1 - self.beta) * self.ema_var + self.beta * target_var
        else:
            self.ema_var = (1 - self.beta) * self.ema_var + self.beta * (
                surprise - self.ema_mu
            ) ** 2

    def get_sigma(self) -> torch.Tensor:
        return torch.sqrt(self.ema_var + 1e-9)


class PILRStrategy(StrategyComponent):
    def __init__(
        self, device: torch.device, crisis_threshold: Optional[float] = None, **kwargs
    ):
        self.device = device
        self.crisis_threshold = crisis_threshold
        self.kwargs = kwargs
        self.step_count = 0
        self.warmup_steps = kwargs.get("warmup_steps", 0)
        self.dual_mode = (
            "gating_initial_var" in kwargs and "expert_initial_var" in kwargs
        )

        if self.dual_mode:
            expert_kwargs = {
                k.replace("expert_", ""): v
                for k, v in kwargs.items()
                if k.startswith("expert_")
            }
            gating_kwargs = {
                k.replace("gating_", ""): v
                for k, v in kwargs.items()
                if k.startswith("gating_")
            }
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

    def apply(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        pi_metrics: Dict[str, Any],
        all_gating_logits: Optional[Any] = None,
        all_top_indices: Optional[List[torch.Tensor]] = None,
        activated_experts: Optional[Dict[int, List[int]]] = None,
    ) -> StepResult:
        if self.dual_mode:
            return self._apply_dual(optimizer, pi_metrics)
        else:
            return self._apply_single(optimizer, pi_metrics)

    def _apply_single(
        self, optimizer: optim.Optimizer, pi_metrics: Dict[str, Any]
    ) -> StepResult:
        surprise = pi_metrics["surprise"]
        self.pilr_adaptor.step(surprise)
        sigma = self.pilr_adaptor.get_sigma()
        mu = self.pilr_adaptor.ema_mu
        lr_mod = pilr_modulation(surprise, mu, sigma, **self._get_mod_kwargs())
        self._update_lr(optimizer, lr_mod)
        return StepResult(lr_mod=lr_mod.item(), sigma=sigma.item())

    def _apply_dual(
        self, optimizer: optim.Optimizer, pi_metrics: Dict[str, Any]
    ) -> StepResult:
        gating_surprise = pi_metrics["gating_surprise"]
        expert_surprise = pi_metrics["expert_surprise"]

        self.pilr_gating.step(gating_surprise)
        self.pilr_expert.step(expert_surprise)

        sigma_gating = self.pilr_gating.get_sigma()
        mu_gating = self.pilr_gating.ema_mu
        lr_mod_gating = pilr_modulation(
            gating_surprise, mu_gating, sigma_gating, **self._get_mod_kwargs("gating_")
        )

        sigma_expert = self.pilr_expert.get_sigma()
        mu_expert = self.pilr_expert.ema_mu
        lr_mod_expert = pilr_modulation(
            expert_surprise, mu_expert, sigma_expert, **self._get_mod_kwargs("expert_")
        )

        self._update_lr(optimizer, lr_mod_gating, "gating")
        self._update_lr(optimizer, lr_mod_expert, "experts")

        return StepResult(
            gating_lr_mod=lr_mod_gating.item(),
            gating_sigma=sigma_gating.item(),
            expert_lr_mod=lr_mod_expert.item(),
            expert_sigma=sigma_expert.item(),
        )

    def _get_mod_kwargs(self, prefix: str = "") -> Dict[str, float]:
        return {
            "power": float(self.kwargs.get(f"{prefix}modulation_power", 0.5)),
            "exponent": float(self.kwargs.get(f"{prefix}modulation_exponent", 2.0)),
        }

    def _update_lr(
        self,
        optimizer: optim.Optimizer,
        lr_mod: torch.Tensor,
        param_group_name: Optional[str] = None,
    ):
        for pg in optimizer.param_groups:
            if param_group_name is None or pg.get("name") == param_group_name:
                if "initial_lr" not in pg:
                    pg["initial_lr"] = pg["lr"]
                pg["lr"] = pg["initial_lr"] * lr_mod


def create_lr_strategy(config: Dict[str, Any]) -> StrategyComponent:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    strategy_name = config.get("name")
    if strategy_name in ["PILR_S", "PILR_D"]:
        return PILRStrategy(device=device, **config)
    else:
        raise ValueError(f"Unknown learning rate strategy: {strategy_name}")
