import torch
from typing import Dict, List

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

class PiCalculator:
    def __init__(self, alpha: float = 1.0, gamma: float = 0.5, device: str = 'cpu'):
        self.alpha = torch.tensor(alpha, device=device)
        self.gamma = torch.tensor(gamma, device=device)
        self.device = device

    def calculate(
        self,
        gradients: List[torch.Tensor],
        loss_epsilon: torch.Tensor,
        logits: torch.Tensor
    ) -> Dict[str, float]:
        if not gradients:
            return {
                "pi_score": 0.0,
                "normalized_error": 0.0,
                "cognitive_cost": 0.0,
                "epsilon": 0.0,
                "tau": 0.0,
                "surprise": 0.0,
            }

        tau_tensor = get_entropy_from_logits_torch(logits)
        surprise_tensor = get_surprise_from_grads_torch(gradients)

        pi_metrics_tensors = calculate_pi_torch(
            epsilon=loss_epsilon,
            tau=tau_tensor,
            surprise=surprise_tensor,
            alpha=self.alpha,
            gamma=self.gamma
        )
        
        pi_metrics_float = {k: v.item() for k, v in pi_metrics_tensors.items()}
        
        return pi_metrics_float