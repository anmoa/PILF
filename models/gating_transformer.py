import torch
import torch.nn as nn


class GatingTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        num_experts: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(embed_dim, num_experts)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        encoded_sequence = self.encoder(src)
        return self.out_proj(encoded_sequence)
