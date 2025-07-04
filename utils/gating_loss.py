from typing import Dict, List, Optional

import torch
import torch.nn as nn


class CompositeGatingLoss(nn.Module):
    def __init__(self, rehearsal_weight: float = 0.5):
        super().__init__()
        self.rehearsal_weight = rehearsal_weight
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

    def _get_target_distribution(
        self,
        ref_log_probs: torch.Tensor,
        indices: Optional[List[int]],
        log_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if indices is not None and len(indices) > 0:
            target_dist = torch.full_like(ref_log_probs, -float("inf"))
            selected_indices = torch.tensor(
                indices, device=ref_log_probs.device, dtype=torch.long
            )
            
            # Use uniform log-probabilities for the selected indices
            uniform_log_prob = torch.log(torch.tensor(1.0 / len(indices), device=ref_log_probs.device))
            log_weights_expanded = uniform_log_prob.expand(ref_log_probs.shape[0], len(indices))

            target_dist.scatter_(
                1,
                selected_indices.unsqueeze(0).expand(ref_log_probs.shape[0], -1),
                log_weights_expanded,
            )
            return target_dist
        
        if log_weights is not None:
            return log_weights.expand_as(ref_log_probs)

        # Fallback to a uniform distribution if no indices or weights are provided
        num_experts = ref_log_probs.size(-1)
        return torch.full_like(ref_log_probs, 1.0 / num_experts).log()

    def forward(
        self,
        all_routing_info: List[Dict[str, torch.Tensor]],
        min_k_expert_indices: Dict[int, List[int]],
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=all_routing_info[0]["log_probs"].device)
        num_layers_processed = 0

        for i, info in enumerate(all_routing_info):
            smk_indices = min_k_expert_indices.get(i)
            if not smk_indices:
                continue

            log_probs = info["log_probs"]
            history_context = info.get("history_context")

            batch_size, num_tokens, num_experts = log_probs.shape
            flat_log_probs = log_probs.reshape(-1, num_experts)

            smk_target_dist = self._get_target_distribution(
                flat_log_probs, smk_indices
            )
            smk_loss = self.kl_div_loss(flat_log_probs, smk_target_dist)

            layer_loss = smk_loss

            if history_context is not None and history_context.abs().sum() > 0:
                clamped_history_context = torch.clamp(history_context, min=-10, max=10)
                rehearsal_log_probs = clamped_history_context.log_softmax(dim=-1)
                
                if torch.isfinite(rehearsal_log_probs).all():
                    rehearsal_target_dist = self._get_target_distribution(
                        flat_log_probs, None, log_weights=rehearsal_log_probs
                    )
                    rehearsal_loss = self.kl_div_loss(flat_log_probs, rehearsal_target_dist)
                    
                    if torch.isfinite(rehearsal_loss):
                        layer_loss = (1 - self.rehearsal_weight) * smk_loss + self.rehearsal_weight * rehearsal_loss

            if torch.isfinite(layer_loss):
                total_loss += layer_loss
                num_layers_processed += 1

        if num_layers_processed > 0:
            return total_loss / num_layers_processed
        return total_loss
