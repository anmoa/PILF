from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn

from .moe_layers import (
    BaseMoELayer,
    GaussianMoELayer,
    MemoryGaussianMoELayer,
    MoELayer,
)


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, mlp_layer: nn.Module, dropout: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = mlp_layer
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_output)

        x_norm = self.norm2(x)

        if isinstance(self.mlp, BaseMoELayer):
            mlp_output, routing_info = self.mlp(x_norm)
            x = x + self.dropout(mlp_output)
            return x, routing_info
        else:
            mlp_output = self.mlp(x_norm)
            x = x + self.dropout(mlp_output)
            return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 128,
        depth: int = 6,
        num_heads: int = 4,
        mlp_dim: int = 256,
        dropout: float = 0.1,
        router_type: str = "dense",
        num_experts: int = 8,
        top_k: int = 2,
        **kwargs,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList([])
        for _ in range(depth):
            mlp_layer: nn.Module
            if router_type == "dense":
                mlp_layer = nn.Sequential(
                    nn.Linear(embed_dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, embed_dim),
                    nn.Dropout(dropout),
                )
            elif router_type == "moe":
                mlp_layer = MoELayer(embed_dim, mlp_dim, embed_dim, num_experts, top_k)
            elif router_type == "gaussian_moe":
                mlp_layer = GaussianMoELayer(
                    embed_dim, mlp_dim, embed_dim, num_experts, top_k
                )
            elif router_type == "memory_gaussian_moe":
                mlp_layer = MemoryGaussianMoELayer(
                    embed_dim,
                    mlp_dim,
                    embed_dim,
                    num_experts,
                    top_k,
                    num_heads=num_heads,
                )
            else:
                raise ValueError(f"Unknown router type: {router_type}")

            self.blocks.append(
                TransformerBlock(embed_dim, num_heads, mlp_layer, dropout)
            )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Union[List[Dict[str, Any]], None]]:
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        all_routing_info: List[Dict[str, Any]] = []
        for block in self.blocks:
            block_output = block(x)
            if isinstance(block_output, tuple):
                x, routing_info = block_output
                all_routing_info.append(routing_info)
            else:
                x = block_output

        x = self.norm(x)

        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)

        if all_routing_info:
            return logits, all_routing_info
        else:
            return logits, None


    def get_param_groups(self) -> List[Dict[str, Any]]:
        gating_param_ids = set()
        expert_param_ids = set()

        for block in self.blocks:
            if isinstance(block.mlp, BaseMoELayer):
                if isinstance(block.mlp, GaussianMoELayer):
                    gating_param_ids.add(id(block.mlp.expert_mus))
                    gating_param_ids.add(id(block.mlp.expert_log_sigmas))
                    if isinstance(block.mlp, MemoryGaussianMoELayer):
                        for param in block.mlp.gating_transformer.parameters():
                            gating_param_ids.add(id(param))
                elif isinstance(block.mlp, MoELayer):
                    for param in block.mlp.gating.parameters():
                        gating_param_ids.add(id(param))

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
            {"params": base_params, "name": "base"},
            {"params": gating_params, "name": "gating"},
            {"params": expert_params, "name": "experts"},
        ]
