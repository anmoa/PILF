from typing import Dict, List, Union

import torch
import torch.nn.functional as F


class PICalculator:
    def __init__(
        self,
        alpha: float,
        gamma: float,
        device: torch.device,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.device = device

    def _calculate_surprise(
        self, gradients: List[torch.Tensor]
    ) -> torch.Tensor:
        if not gradients:
            return torch.tensor(0.0, device=self.device)

        flat_grads = torch.cat([g.flatten() for g in gradients if g is not None])
        return torch.norm(flat_grads, p=2)

    def _calculate_tau(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        return entropy.mean()

    def calculate(
        self,
        loss: torch.Tensor,
        logits: torch.Tensor,
        gradients: Union[List[torch.Tensor], Dict[str, List[torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        epsilon = loss
        tau = self._calculate_tau(logits)

        metrics: Dict[str, torch.Tensor] = {
            "epsilon": epsilon,
            "tau": tau,
        }

        if isinstance(gradients, dict):
            gating_grads = gradients.get("gating", [])
            expert_grads = gradients.get("experts", [])
            
            gating_surprise = self._calculate_surprise(gating_grads)
            expert_surprise = self._calculate_surprise(expert_grads)
            
            all_grads = [g for g_list in gradients.values() for g in g_list]
            total_surprise = self._calculate_surprise(all_grads)

            metrics.update({
                "gating_surprise": gating_surprise,
                "expert_surprise": expert_surprise,
                "surprise": total_surprise,
            })
            surprise_term = (1 - self.gamma) * expert_surprise + self.gamma * gating_surprise
        else:
            surprise = self._calculate_surprise(gradients)
            metrics["surprise"] = surprise
            surprise_term = surprise

        normalized_epsilon = epsilon / (tau + 1e-9)
        
        pi_score = torch.exp(
            -self.alpha * (normalized_epsilon + surprise_term)
        )
        metrics["pi_score"] = pi_score

        return metrics
