import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_vit import VisionTransformer
from utils.strategies import PisaAdaptor

class GaussianMoELayer(nn.Module):
    """
    A Mixture of Experts layer using Gaussian-based routing with PISA-driven adaptive inhibition.
    """
    def __init__(self, in_features, hidden_features, out_features, num_experts, top_k=2, 
                 ood_inhibition_c=2.0, routing_pisa_initial_var=1.0, routing_pisa_beta=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.in_features = in_features
        self.ood_inhibition_c = ood_inhibition_c
        
        self.expert_mus = nn.Parameter(torch.randn(num_experts, in_features))
        self.expert_log_sigmas = nn.Parameter(torch.zeros(num_experts, in_features))
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, hidden_features),
                nn.GELU(),
                nn.Linear(hidden_features, out_features)
            ) for _ in range(num_experts)
        ])

        # This PISA instance monitors the routing confidence (max_log_probs)
        self.routing_pisa = None
        if routing_pisa_initial_var is not None:
            self.routing_pisa = PisaAdaptor(
                device=self.expert_mus.device, 
                initial_var=routing_pisa_initial_var, 
                beta=routing_pisa_beta
            )

    def forward(self, x):
        batch_size, num_tokens, in_features = x.shape
        x_flat = x.reshape(-1, in_features)

        # Calculate Gaussian probability density
        sigmas = torch.exp(self.expert_log_sigmas)
        
        # Broadcasting x_flat and mus for vectorized computation
        # x_flat: (batch*tokens, in_features) -> (batch*tokens, 1, in_features)
        # mus: (num_experts, in_features) -> (1, num_experts, in_features)
        # sigmas: (num_experts, in_features) -> (1, num_experts, in_features)
        
        x_unsqueezed = x_flat.unsqueeze(1)
        mus_unsqueezed = self.expert_mus.unsqueeze(0)
        sigmas_unsqueezed = sigmas.unsqueeze(0)

        # Mahalanobis distance squared (assuming diagonal covariance)
        dist_sq = ((x_unsqueezed - mus_unsqueezed) / sigmas_unsqueezed).pow(2).sum(dim=-1)
        
        # Log probability density (ignoring constant terms for softmax)
        log_probs = -0.5 * dist_sq - self.expert_log_sigmas.sum(dim=-1)
        
        final_log_probs = log_probs
        if self.routing_pisa is not None:
            # --- PISA-driven Adaptive Routing Inhibition ---

            # 1. Get routing confidence signal
            max_log_probs, top_indices_raw = torch.topk(log_probs, self.top_k, dim=-1)

            # 2. Update routing PISA with the mean confidence of the current batch
            if self.training:
                # Use the confidence of the top-1 expert as the signal
                self.routing_pisa.step(max_log_probs[:, 0].mean().item())

            # 3. Calculate dynamic OOD threshold
            dynamic_threshold = self.routing_pisa.ema_mu - self.ood_inhibition_c * self.routing_pisa.get_sigma()

            # 4. Identify OOD tokens
            # An input is considered OOD if its best-matching expert's confidence is below the dynamic threshold.
            ood_mask = max_log_probs[:, 0] < dynamic_threshold
            
            final_log_probs = log_probs.clone()

            if ood_mask.any():
                # 5. For OOD tokens, apply inhibitory bias to the top-k experts
                # Get the indices of the top-k experts for the OOD tokens
                ood_top_k_indices = top_indices_raw[ood_mask]
                
                # Create an inhibitory bias and apply it
                # We use scatter_ to efficiently apply the bias to the correct locations
                inhibition_bias = torch.full_like(final_log_probs[ood_mask], 0, device=x.device)
                inhibition_bias.scatter_(1, ood_top_k_indices, -1e9) # Large negative bias
                
                final_log_probs[ood_mask] += inhibition_bias

        # 6. Final routing decision
        weights = torch.softmax(final_log_probs, dim=-1)
        
        # The top_indices for sparse gradient update should always be from the original, uninhibited routing
        _, top_indices_for_update = torch.topk(log_probs, self.top_k, dim=-1)
        
        # Pass original input to all experts and get their outputs
        expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)
        
        # Weighted sum of expert outputs using potentially inhibited weights
        combined_output = (weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
        
        final_output = combined_output.reshape(batch_size, num_tokens, -1)
        
        # Return final output, original log_probs for analysis, and top_indices for sparse update
        return final_output, log_probs.reshape(batch_size, num_tokens, -1), top_indices_for_update

class GPILMoEVisionTransformer(VisionTransformer):
    """A Vision Transformer with GaussianMoE layers."""
    def __init__(self, num_experts=8, top_k=2, ood_inhibition_c=2.0, 
                 routing_pisa_initial_var=None, routing_pisa_beta=0.1, **kwargs):
        super().__init__(**kwargs)

        embed_dim = kwargs.get('embed_dim', 128)
        mlp_dim = kwargs.get('mlp_dim', 256)
        depth = kwargs.get('depth', 6)

        for i in range(depth):
            self.blocks[i].mlp = GaussianMoELayer(
                in_features=embed_dim,
                hidden_features=mlp_dim,
                out_features=embed_dim,
                num_experts=num_experts,
                top_k=top_k,
                ood_inhibition_c=ood_inhibition_c,
                routing_pisa_initial_var=routing_pisa_initial_var,
                routing_pisa_beta=routing_pisa_beta
            )

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        all_log_probs = []
        all_top_indices = []
        for block in self.blocks:
            x_norm1 = block.norm1(x)
            attn_output, _ = block.attn(x_norm1, x_norm1, x_norm1)
            x = x + block.dropout(attn_output)
            
            x_norm2 = block.norm2(x)
            mlp_output, log_probs, top_indices = block.mlp(x_norm2)
            x = x + block.dropout(mlp_output)
            all_log_probs.append(log_probs)
            all_top_indices.append(top_indices)
            
        x = self.norm(x)
        
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)
        
        # Return logits, top_indices for sparse update, and log_probs for analysis
        return logits, all_top_indices, all_log_probs

    def zero_inactive_expert_grads(self, all_top_indices):
        with torch.no_grad():
            for i, block in enumerate(self.blocks):
                if isinstance(block.mlp, GaussianMoELayer):
                    top_indices_block = all_top_indices[i]
                    device = top_indices_block.device 
                    
                    active_experts_mask = torch.zeros(block.mlp.num_experts, dtype=torch.bool, device=device)
                    active_experts_mask[top_indices_block.unique()] = True
                    
                    for expert_idx, expert_layer in enumerate(block.mlp.experts):
                        if not active_experts_mask[expert_idx]:
                            for param in expert_layer.parameters():
                                if param.grad is not None:
                                    param.grad.zero_()
                    
                    # Also zero out gradients for inactive mus and sigmas
                    for expert_idx in range(block.mlp.num_experts):
                        if not active_experts_mask[expert_idx]:
                            if block.mlp.expert_mus.grad is not None:
                                block.mlp.expert_mus.grad[expert_idx].zero_()
                            if block.mlp.expert_log_sigmas.grad is not None:
                                block.mlp.expert_log_sigmas.grad[expert_idx].zero_()
