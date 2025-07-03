import torch
import torch.nn as nn

from .base_vit import VisionTransformer


class GaussianMoELayer(nn.Module):
    """
    A Mixture of Experts layer using Gaussian-based routing with adaptive inhibition.
    """
    def __init__(self, in_features, hidden_features, out_features, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.in_features = in_features
        
        self.expert_mus = nn.Parameter(torch.randn(num_experts, in_features))
        self.expert_log_sigmas = nn.Parameter(torch.zeros(num_experts, in_features))
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, hidden_features),
                nn.GELU(),
                nn.Linear(hidden_features, out_features)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        batch_size, num_tokens, in_features = x.shape
        x_flat = x.reshape(-1, in_features)

        sigmas = torch.exp(self.expert_log_sigmas)
        
        x_unsqueezed = x_flat.unsqueeze(1)
        mus_unsqueezed = self.expert_mus.unsqueeze(0)
        sigmas_unsqueezed = sigmas.unsqueeze(0)

        dist_sq = ((x_unsqueezed - mus_unsqueezed) / sigmas_unsqueezed).pow(2).sum(dim=-1)
        
        log_probs = -0.5 * dist_sq - self.expert_log_sigmas.sum(dim=-1)
        
        final_log_probs = log_probs
        
        weights = torch.softmax(final_log_probs, dim=-1)
        
        _, top_indices_for_update = torch.topk(log_probs, self.top_k, dim=-1)
        
        expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)
        
        combined_output = (weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
        
        final_output = combined_output.reshape(batch_size, num_tokens, -1)
        
        routing_info = {
            "log_probs": log_probs.reshape(batch_size, num_tokens, -1),
            "weights": weights.reshape(batch_size, num_tokens, -1),
            "top_indices": top_indices_for_update
        }
        
        return final_output, routing_info

class GaussianMoEVisionTransformer(VisionTransformer):
    """A Vision Transformer with GaussianMoE layers."""
    def __init__(self, num_experts=8, top_k=2, **kwargs):
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
            )

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        all_routing_info = []
        for block in self.blocks:
            x_norm1 = block.norm1(x)
            attn_output, _ = block.attn(x_norm1, x_norm1, x_norm1)
            x = x + block.dropout(attn_output)
            
            x_norm2 = block.norm2(x)
            mlp_output, routing_info = block.mlp(x_norm2)
            x = x + block.dropout(mlp_output)
            all_routing_info.append(routing_info)
            
        x = self.norm(x)
        
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)
        
        return logits, all_routing_info

    def zero_inactive_expert_grads(self, all_routing_info):
        with torch.no_grad():
            for i, block in enumerate(self.blocks):
                if isinstance(block.mlp, GaussianMoELayer):
                    top_indices_block = all_routing_info[i]["top_indices"]
                    device = top_indices_block.device
                    
                    active_experts_mask = torch.zeros(block.mlp.num_experts, dtype=torch.bool, device=device)
                    active_experts_mask[top_indices_block.unique()] = True
                    
                    for expert_idx, expert_layer in enumerate(block.mlp.experts):
                        if not active_experts_mask[expert_idx]:
                            for param in expert_layer.parameters():
                                if param.grad is not None:
                                    param.grad.zero_()
                    
                    for expert_idx in range(block.mlp.num_experts):
                        if not active_experts_mask[expert_idx]:
                            if block.mlp.expert_mus.grad is not None:
                                block.mlp.expert_mus.grad[expert_idx].zero_()
                            if block.mlp.expert_log_sigmas.grad is not None:
                                block.mlp.expert_log_sigmas.grad[expert_idx].zero_()

    def get_param_groups(self):
        gating_param_ids = set()
        expert_param_ids = set()

        for block in self.blocks:
            if isinstance(block.mlp, GaussianMoELayer):
                gating_param_ids.add(id(block.mlp.expert_mus))
                gating_param_ids.add(id(block.mlp.expert_log_sigmas))
                for param in block.mlp.experts.parameters():
                    expert_param_ids.add(id(param))

        gating_params = []
        expert_params = []
        base_params = []

        for _, param in self.named_parameters():
            param_id = id(param)
            if param_id in gating_param_ids:
                gating_params.append(param)
            elif param_id in expert_param_ids:
                expert_params.append(param)
            else:
                base_params.append(param)

        return [
            {'params': base_params, 'name': 'base'},
            {'params': gating_params, 'name': 'gating'},
            {'params': expert_params, 'name': 'experts'}
        ]
