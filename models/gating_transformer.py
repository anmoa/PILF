import torch
import torch.nn as nn


class GatingTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        context_dim: int,
        num_experts: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, num_experts)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        
        attn_output, _ = self.attn(query, key, value)
        attn_output = self.norm(attn_output)
        return self.out_proj(attn_output)