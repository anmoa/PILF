from typing import Dict, List

import torch


def get_surprise_from_grads_torch(gradients: List[torch.Tensor]) -> torch.Tensor:
    valid_gradients = [g for g in gradients if g is not None]
    if not valid_gradients:
        return torch.tensor(0.0)
    all_grads = torch.cat([p.flatten() for p in valid_gradients])
    return torch.norm(all_grads, p=2)

def get_entropy_from_logits_torch(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    return entropy.mean()

def calculate_pi_torch(
    epsilon: torch.Tensor,
    tau: torch.Tensor,
    surprise: torch.Tensor,
    alpha: torch.Tensor,
    gamma: torch.Tensor
) -> Dict[str, torch.Tensor]:
    normalized_error = epsilon / (tau + 1e-9)
    cognitive_cost = (1 - gamma) * normalized_error + gamma * surprise
    pi_score = torch.exp(-alpha * cognitive_cost)

    return {
        "pi_score": pi_score,
        "normalized_error": normalized_error,
        "cognitive_cost": cognitive_cost,
        "epsilon": epsilon,
        "tau": tau,
        "surprise": surprise,
    }

class LocalPiCalculator:
    def calculate(
        self,
        gradients: List[torch.Tensor],
        loss_epsilon: torch.Tensor,
        logits: torch.Tensor
    ) -> Dict[str, float]:
        if not gradients:
            return {
                "epsilon": 0.0,
                "tau": 0.0,
                "surprise": 0.0,
            }

        tau_tensor = get_entropy_from_logits_torch(logits)
        surprise_tensor = get_surprise_from_grads_torch(gradients)
        
        return {
            "epsilon": loss_epsilon.item(),
            "tau": tau_tensor.item(),
            "surprise": surprise_tensor.item(),
        }

class GlobalPiCalculator:
    def __init__(self, alpha: float = 1.0, gamma: float = 0.5, device: str = 'cpu'):
        self.alpha = torch.tensor(alpha, device=device)
        self.gamma = torch.tensor(gamma, device=device)
        self.device = device

    def calculate(
        self,
        local_pi_components: List[Dict[str, float]]
    ) -> Dict[str, float]:
        if not local_pi_components:
            return {
                "pi_score": 0.0,
                "normalized_error": 0.0,
                "cognitive_cost": 0.0,
                "epsilon": 0.0,
                "tau": 0.0,
                "surprise": 0.0,
            }

        total_epsilon = torch.tensor(sum(c['epsilon'] for c in local_pi_components), device=self.device)
        total_tau = torch.tensor(sum(c['tau'] for c in local_pi_components), device=self.device)
        total_surprise = torch.tensor(sum(c['surprise'] for c in local_pi_components), device=self.device)

        pi_metrics_tensors = calculate_pi_torch(
            epsilon=total_epsilon,
            tau=total_tau,
            surprise=total_surprise,
            alpha=self.alpha,
            gamma=self.gamma
        )
        
        pi_metrics_float = {k: v.item() for k, v in pi_metrics_tensors.items()}
        
        return pi_metrics_float
