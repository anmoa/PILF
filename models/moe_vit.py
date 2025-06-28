import torch
import torch.nn as nn
from .base_vit import VisionTransformer

class MoELayer(nn.Module):
    """A Mixture of Experts layer."""
    def __init__(self, in_features, hidden_features, out_features, num_experts, top_k=1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gating = nn.Linear(in_features, num_experts)
        
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

        gating_logits = self.gating(x_flat)
        
        weights, indices = torch.topk(gating_logits, self.top_k, dim=-1)
        weights = torch.softmax(weights, dim=-1).to(x.dtype)
        
        dispatch_tensor = torch.zeros(x_flat.shape[0], self.num_experts, device=x.device, dtype=x.dtype)
        dispatch_tensor.scatter_(1, indices, weights)
        
        dispatched_x = dispatch_tensor.unsqueeze(-1) * x_flat.unsqueeze(1)
        
        expert_outputs_list = []
        for i in range(self.num_experts):
            expert_input_i = dispatched_x[:, i, :]
            expert_output_i = self.experts[i](expert_input_i)
            expert_outputs_list.append(expert_output_i)
            
        expert_outputs = torch.stack(expert_outputs_list, dim=1)

        combined_output = (dispatch_tensor.unsqueeze(-1) * expert_outputs).sum(dim=1)
        
        final_output = combined_output.reshape(batch_size, num_tokens, -1)
        
        return final_output, gating_logits

class MoEVisionTransformer(VisionTransformer):
    """A Vision Transformer with MoE layers replacing the MLP layers."""
    def __init__(self, num_experts=4, top_k=1, **kwargs):
        kwargs.pop('model_type', None)
        super().__init__(**kwargs)

        embed_dim = kwargs.get('embed_dim', 128)
        mlp_dim = kwargs.get('mlp_dim', 256)
        depth = kwargs.get('depth', 6)

        for i in range(depth):
            self.blocks[i].mlp = MoELayer(
                in_features=embed_dim,
                hidden_features=mlp_dim,
                out_features=embed_dim,
                num_experts=num_experts,
                top_k=top_k
            )

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        all_gating_logits = []
        for block in self.blocks:
            x_norm1 = block.norm1(x)
            attn_output, _ = block.attn(x_norm1, x_norm1, x_norm1)
            x = x + block.dropout(attn_output)
            
            x_norm2 = block.norm2(x)
            mlp_output, gating_logits = block.mlp(x_norm2)
            x = x + block.dropout(mlp_output)
            all_gating_logits.append(gating_logits)
            
        x = self.norm(x)
        
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)
        
        return logits, all_gating_logits

    def get_param_groups(self):
        """Helper method to get parameter groups for the optimizer."""
        gating_params = []
        expert_params = []
        base_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            # Identify parameters by name
            if 'mlp.gating' in name:
                gating_params.append(param)
            elif 'mlp.experts' in name:
                expert_params.append(param)
            else:
                base_params.append(param)
        
        return [
            {'name': 'base', 'params': base_params},
            {'name': 'gating', 'params': gating_params},
            {'name': 'experts', 'params': expert_params},
        ]

    def zero_inactive_expert_grads(self, all_gating_logits):
        with torch.no_grad():
            for i, block in enumerate(self.blocks):
                if hasattr(block.mlp, 'experts'):
                    gating_logits_block = all_gating_logits[i]
                    device = gating_logits_block.device 
                    
                    _, top_indices = torch.topk(gating_logits_block, block.mlp.top_k, dim=-1)
                    
                    active_experts_mask = torch.zeros(block.mlp.num_experts, dtype=torch.bool, device=device)
                    active_experts_mask[top_indices.unique()] = True
                    
                    for expert_idx, expert_layer in enumerate(block.mlp.experts):
                        if not active_experts_mask[expert_idx]:
                            for param in expert_layer.parameters():
                                if param.grad is not None:
                                    param.grad.zero_()
