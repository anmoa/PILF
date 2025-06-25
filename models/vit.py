import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Splits an image into patches and embeds them.
    
    Args:
        img_size (int): Size of the input image (e.g., 32 for CIFAR-10).
        patch_size (int): Size of each patch (e.g., 4 for CIFAR-10).
        in_channels (int): Number of input channels (e.g., 3 for RGB).
        embed_dim (int): The embedding dimension for each patch.
    """
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x)  # (B, E, P, P)
        x = x.flatten(2) # (B, E, N)
        x = x.transpose(1, 2) # (B, N, E)
        return x

class TransformerBlock(nn.Module):
    """A standard Transformer block."""
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Attention part
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_output)
        
        # MLP part
        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        x = x + self.dropout(mlp_output)
        
        return x

class VisionTransformer(nn.Module):
    """
    A Vision Transformer model customized for smaller images like CIFAR-10.
    """
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=128, depth=6, num_heads=4, mlp_dim=256, dropout=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        # Get CLS token for classification
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)
        
        return logits

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
        
        # Non-einsum implementation to avoid inplace errors with torch.compile
        dispatched_x = dispatch_tensor.unsqueeze(-1) * x_flat.unsqueeze(1)
        
        expert_outputs_list = []
        for i in range(self.num_experts):
            expert_input_i = dispatched_x[:, i, :]
            expert_output_i = self.experts[i](expert_input_i)
            expert_outputs_list.append(expert_output_i)
            
        expert_outputs = torch.stack(expert_outputs_list, dim=1)

        # Non-einsum implementation for combining outputs
        combined_output = (dispatch_tensor.unsqueeze(-1) * expert_outputs).sum(dim=1)
        
        final_output = combined_output.reshape(batch_size, num_tokens, -1)
        
        # For PILR-S or other analysis, we might need to know which experts were activated.
        # We return the raw gating logits for flexibility.
        return final_output, gating_logits

class MoEVisionTransformer(VisionTransformer):
    """A Vision Transformer with MoE layers replacing the MLP layers."""
    def __init__(self, num_experts=4, top_k=1, **kwargs):
        # Pop 'model_type' as it's not a parameter for the parent VisionTransformer
        kwargs.pop('model_type', None)
        super().__init__(**kwargs)

        # Extract necessary dimensions from kwargs
        embed_dim = kwargs.get('embed_dim', 128)
        mlp_dim = kwargs.get('mlp_dim', 256)
        depth = kwargs.get('depth', 6)

        # Replace MLP blocks with MoE layers
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
            # Attention part is the same
            x_norm1 = block.norm1(x)
            attn_output, _ = block.attn(x_norm1, x_norm1, x_norm1)
            x = x + block.dropout(attn_output)
            
            # MoE MLP part
            x_norm2 = block.norm2(x)
            mlp_output, gating_logits = block.mlp(x_norm2)
            x = x + block.dropout(mlp_output)
            all_gating_logits.append(gating_logits)
            
        x = self.norm(x)
        
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)
        
        # Return logits and gating decisions for analysis
        return logits, all_gating_logits

    def zero_inactive_expert_grads(self, all_gating_logits):
        """
        Zeros out the gradients for inactive experts.
        This is the core of the FFB's selective update mechanism.
        """
        with torch.no_grad():
            for i, block in enumerate(self.blocks):
                if hasattr(block.mlp, 'experts'):
                    gating_logits_block = all_gating_logits[i]
                    # Assuming gating_logits_block is on the same device as the model
                    device = gating_logits_block.device 
                    
                    _, top_indices = torch.topk(gating_logits_block, block.mlp.top_k, dim=-1)
                    
                    active_experts_mask = torch.zeros(block.mlp.num_experts, dtype=torch.bool, device=device)
                    active_experts_mask[top_indices.unique()] = True
                    
                    for expert_idx, expert_layer in enumerate(block.mlp.experts):
                        if not active_experts_mask[expert_idx]:
                            for param in expert_layer.parameters():
                                if param.grad is not None:
                                    param.grad.zero_()
