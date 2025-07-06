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
        self.embed_dim = embed_dim
        self.num_experts = num_experts

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(embed_dim, num_experts * 2 * embed_dim)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        decoded_sequence = self.decoder(tgt=tgt, memory=memory)
        
        params = self.out_proj(decoded_sequence)
        
        b, s, _ = tgt.shape
        params = params.view(b, s, self.num_experts, 2, self.embed_dim)
        
        mus = params[..., 0, :]
        log_sigmas = params[..., 1, :]
        
        return mus, log_sigmas
