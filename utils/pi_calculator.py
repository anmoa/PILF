from typing import Dict, List

import torch


def get_surprise_from_grads_torch(gradients: List[torch.Tensor]) -> torch.Tensor:
    valid_gradients = [g for g in gradients if g is not None]
    
    # Determine the device for the initial zero tensor if needed
    # If gradients is empty, default to 'cpu'. Otherwise, use the device of the first gradient.
    device = gradients[0].device if gradients else 'cpu'

    # Calculate sum of squares for each gradient and then sum them up
    # Use a torch.tensor(0.0) as the starting point for sum to ensure the result is always a tensor.
    total_norm_sq = sum((torch.sum(g**2) for g in valid_gradients), start=torch.tensor(0.0, device=device))
    return torch.sqrt(total_norm_sq)

def get_entropy_from_logits_torch(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    # Add a small epsilon to log(probs) to prevent log(0)
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
    ) -> Dict[str, torch.Tensor]: # Changed return type to torch.Tensor
        if not gradients:
            # Ensure returned tensors are on the same device as loss_epsilon/logits
            device = loss_epsilon.device if isinstance(loss_epsilon, torch.Tensor) else logits.device
            return {
                "epsilon": torch.tensor(0.0, device=device),
                "tau": torch.tensor(0.0, device=device),
                "surprise": torch.tensor(0.0, device=device),
            }

        tau_tensor = get_entropy_from_logits_torch(logits)
        surprise_tensor = get_surprise_from_grads_torch(gradients)
        
        return {
            "epsilon": loss_epsilon, # Keep as tensor
            "tau": tau_tensor, # Keep as tensor
            "surprise": surprise_tensor, # Keep as tensor
        }

class GlobalPiCalculator:
    def __init__(self, alpha: float = 1.0, gamma: float = 0.5, device: str = 'cpu'):
        self.alpha = torch.tensor(alpha, device=device)
        self.gamma = torch.tensor(gamma, device=device)
        self.device = device

    def calculate(
        self,
        local_pi_components: List[Dict[str, torch.Tensor]] # Changed to accept torch.Tensor
    ) -> Dict[str, torch.Tensor]: # Changed return type to torch.Tensor
        if not local_pi_components:
            return {
                "pi_score": torch.tensor(0.0, device=self.device),
                "normalized_error": torch.tensor(0.0, device=self.device),
                "cognitive_cost": torch.tensor(0.0, device=self.device),
                "epsilon": torch.tensor(0.0, device=self.device),
                "tau": torch.tensor(0.0, device=self.device),
                "surprise": torch.tensor(0.0, device=self.device),
            }

        # Stack tensors and sum on GPU
        total_epsilon = torch.stack([c['epsilon'] for c in local_pi_components]).sum()
        total_tau = torch.stack([c['tau'] for c in local_pi_components]).sum()
        total_surprise = torch.stack([c['surprise'] for c in local_pi_components]).sum()

        pi_metrics_tensors = calculate_pi_torch(
            epsilon=total_epsilon,
            tau=total_tau,
            surprise=total_surprise,
            alpha=self.alpha,
            gamma=self.gamma
        )
        
        # Return tensors directly, no .item() here
        return pi_metrics_tensors
