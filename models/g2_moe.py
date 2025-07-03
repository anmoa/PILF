from typing import Optional

import torch
import torch.nn as nn

from .base_vit import VisionTransformer
from .gaussian_moe import GaussianMoELayer


def _count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class GenGaussianMoELayer(GaussianMoELayer):
    def __init__(self, in_features, hidden_features, out_features, num_experts, top_k=2,
                 narrative_generator_dim: Optional[int] = None, vae_latent_dim: Optional[int] = None,
                 nhead: int = 4, use_vae: bool = False):
        super().__init__(in_features, hidden_features, out_features, num_experts, top_k)
        
        self.narrative_generator_dim = narrative_generator_dim if narrative_generator_dim is not None else max(1, in_features // 20)
        self.use_vae = use_vae
        
        if (self.narrative_generator_dim + 1) % nhead != 0:
            self.narrative_generator_dim = self.narrative_generator_dim + (nhead - ((self.narrative_generator_dim + 1) % nhead)) - 1
            if self.narrative_generator_dim <= 0:
                self.narrative_generator_dim = nhead - 1

        self.vae_latent_dim = vae_latent_dim if vae_latent_dim is not None else max(1, in_features // 30)

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.narrative_generator_dim + 1,
            nhead=nhead,
            dim_feedforward=self.narrative_generator_dim * 2,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=1)

        # 简化版的调制参数生成器
        self.simple_modulation_generator = nn.Sequential(
            nn.Linear(self.narrative_generator_dim + 1, in_features * 4),
            nn.GELU(),
            nn.Linear(in_features * 4, in_features * 2)
        )

        # 保留VAE代码以便将来使用
        vae_encoder_input_dim = (self.narrative_generator_dim + 1) + self.narrative_generator_dim
        self.vae_encoder = nn.Sequential(
            nn.Linear(vae_encoder_input_dim, self.vae_latent_dim * 2),
            nn.GELU(),
            nn.Linear(self.vae_latent_dim * 2, self.vae_latent_dim * 2)
        )
        self.vae_decoder = nn.Sequential(
            nn.Linear(self.vae_latent_dim, in_features * 2),
            nn.GELU(),
            nn.Linear(in_features * 2, in_features * 2)
        )
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, pi_score=None, task_features=None, use_narrative_generator: bool = True):
        batch_size, num_tokens, in_features = x.shape
        x_flat = x.reshape(-1, in_features)

        vae_kl_loss = torch.tensor(0.0, device=x.device)

        if use_narrative_generator:
            if task_features is None:
                task_features = torch.zeros(x_flat.shape[0], self.narrative_generator_dim, device=x.device)
            else:
                task_features = task_features.reshape(-1, self.narrative_generator_dim)

            if pi_score is None:
                pi_score_tensor = torch.zeros(x_flat.shape[0], 1, device=x.device)
            else:
                pi_score_tensor = pi_score.unsqueeze(-1).expand(x_flat.shape[0], 1)

            transformer_input = torch.cat([task_features, pi_score_tensor], dim=-1).unsqueeze(1)
            transformer_output = self.transformer_encoder(transformer_input).squeeze(1)
            
            if self.use_vae:
                # 使用VAE生成调制参数（保留以便将来使用）
                vae_encoder_input = torch.cat([transformer_output, task_features], dim=-1)
                mu_log_var = self.vae_encoder(vae_encoder_input)
                mu, log_var = mu_log_var.chunk(2, dim=-1)
                
                z = self.reparameterize(mu, log_var)
                modulation_params = self.vae_decoder(z)
                
                vae_kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()
            else:
                # 使用简化版的调制参数生成器
                modulation_params = self.simple_modulation_generator(transformer_output)
            
            modulation_mus, modulation_log_sigmas = modulation_params.chunk(2, dim=-1)
            
            modulated_expert_mus = self.expert_mus.unsqueeze(0).expand(x_flat.shape[0], -1, -1) + modulation_mus.unsqueeze(1)
            modulated_expert_log_sigmas = self.expert_log_sigmas.unsqueeze(0).expand(x_flat.shape[0], -1, -1) + modulation_log_sigmas.unsqueeze(1)
        else:
            modulated_expert_mus = self.expert_mus.unsqueeze(0).expand(x_flat.shape[0], -1, -1)
            modulated_expert_log_sigmas = self.expert_log_sigmas.unsqueeze(0).expand(x_flat.shape[0], -1, -1)
            
        sigmas = torch.exp(modulated_expert_log_sigmas)
        
        x_unsqueezed = x_flat.unsqueeze(1)
        
        dist_sq = ((x_unsqueezed - modulated_expert_mus) / sigmas).pow(2).sum(dim=-1)
        
        log_probs = -0.5 * dist_sq - modulated_expert_log_sigmas.sum(dim=-1)
        
        final_log_probs = log_probs
        
        weights = torch.softmax(final_log_probs, dim=-1)
        
        # 获取 top_k 专家的索引
        _, top_indices_for_update = torch.topk(weights, self.top_k, dim=-1)
        
        # 计算所有专家的输出
        # 形状为 (num_tokens_flat, num_experts, out_features)
        expert_outputs_all = torch.stack([expert(x_flat) for expert in self.experts], dim=1)
        
        # 使用 gather 收集 top_k 专家的输出
        # top_indices_for_update 的形状是 (num_tokens_flat, top_k)
        # 我们需要将其扩展到 (num_tokens_flat, top_k, out_features)
        top_indices_expanded = top_indices_for_update.unsqueeze(-1).expand(-1, -1, expert_outputs_all.shape[-1])
        
        # 从所有专家的输出中收集 top_k 专家的输出
        expert_outputs_top_k = torch.gather(expert_outputs_all, 1, top_indices_expanded)
        
        # 重新归一化 top_k 专家的权重
        weights_top_k = torch.gather(weights, 1, top_indices_for_update)
        weights_top_k_normalized = weights_top_k / weights_top_k.sum(dim=-1, keepdim=True)
        
        # 使用重新归一化的权重和 top_k 专家的输出进行加权求和
        combined_output = (weights_top_k_normalized.unsqueeze(-1) * expert_outputs_top_k).sum(dim=1)
        
        final_output = combined_output.reshape(batch_size, num_tokens, -1)
        
        routing_info = {
            "log_probs": log_probs.reshape(batch_size, num_tokens, -1),
            "weights": weights.reshape(batch_size, num_tokens, -1),
            "top_indices": top_indices_for_update,
            "vae_kl_loss": vae_kl_loss
        }
        
        return final_output, routing_info

class GenGaussianMoEVisionTransformer(VisionTransformer):
    def __init__(self, num_experts=8, top_k=2, narrative_generator_dim: Optional[int] = None, 
                 vae_latent_dim: Optional[int] = None, max_narrative_params_ratio: float = 0.1, 
                 use_vae: bool = False, **kwargs):
        super().__init__(**kwargs)

        embed_dim = kwargs.get('embed_dim', 128)
        mlp_dim = kwargs.get('mlp_dim', 256)
        depth = kwargs.get('depth', 6)
        num_heads = kwargs.get('num_heads', 4)

        dummy_base_model = VisionTransformer(**kwargs)
        base_model_params = _count_parameters(dummy_base_model)
        
        target_narrative_params = base_model_params * max_narrative_params_ratio

        current_narrative_generator_dim = narrative_generator_dim if narrative_generator_dim is not None else max(1, embed_dim // 20)
        current_vae_latent_dim = vae_latent_dim if vae_latent_dim is not None else max(1, embed_dim // 30)

        # Define task_feature_extractor input dimension
        # Now task_feature_extractor will process the output of patch_embed, which has embed_dim
        task_feature_extractor_input_dim = embed_dim
        
        # Initialize task_feature_extractor with initial dimensions
        self.task_feature_extractor = nn.Sequential(
            nn.Linear(task_feature_extractor_input_dim, current_narrative_generator_dim // 2),
            nn.GELU(),
            nn.Linear(current_narrative_generator_dim // 2, current_narrative_generator_dim)
        )

        for _ in range(10): # Iterate to find suitable dimensions
            if (current_narrative_generator_dim + 1) % num_heads != 0:
                current_narrative_generator_dim = current_narrative_generator_dim + (num_heads - ((current_narrative_generator_dim + 1) % num_heads)) - 1
                if current_narrative_generator_dim <= 0:
                    current_narrative_generator_dim = num_heads - 1
            
            task_feature_extractor_hidden_dim = current_narrative_generator_dim // 2
            task_feature_extractor_output_dim = current_narrative_generator_dim

            task_feature_extractor_params = (task_feature_extractor_input_dim * task_feature_extractor_hidden_dim + task_feature_extractor_hidden_dim) + \
                                            (task_feature_extractor_hidden_dim * task_feature_extractor_output_dim + task_feature_extractor_output_dim)

            dummy_transformer_encoder_layer = nn.TransformerEncoderLayer(
                d_model=current_narrative_generator_dim + 1,
                nhead=num_heads,
                dim_feedforward=current_narrative_generator_dim * 2,
                batch_first=True
            )
            transformer_encoder_params_per_layer = _count_parameters(dummy_transformer_encoder_layer)

            vae_encoder_input_dim = (current_narrative_generator_dim + 1) + current_narrative_generator_dim
            vae_encoder_hidden_dim = current_vae_latent_dim * 2
            vae_encoder_output_dim = current_vae_latent_dim * 2

            vae_encoder_params = (vae_encoder_input_dim * vae_encoder_hidden_dim + vae_encoder_hidden_dim) + \
                                 (vae_encoder_hidden_dim * vae_encoder_output_dim + vae_encoder_output_dim)

            vae_decoder_input_dim = current_vae_latent_dim
            vae_decoder_hidden_dim = embed_dim * 2
            vae_decoder_output_dim = embed_dim * 2

            vae_decoder_params = (vae_decoder_input_dim * vae_decoder_hidden_dim + vae_decoder_hidden_dim) + \
                                 (vae_decoder_hidden_dim * vae_decoder_output_dim + vae_decoder_output_dim)

            total_narrative_params = task_feature_extractor_params + \
                                     depth * (transformer_encoder_params_per_layer + vae_encoder_params + vae_decoder_params)

            if total_narrative_params <= target_narrative_params:
                break
            else:
                scale_factor = (target_narrative_params / total_narrative_params) ** 0.5
                current_narrative_generator_dim = max(num_heads - 1, int(current_narrative_generator_dim * scale_factor))
                current_vae_latent_dim = max(1, int(current_vae_latent_dim * scale_factor))
        
        self.narrative_generator_dim = current_narrative_generator_dim
        self.vae_latent_dim = current_vae_latent_dim

        # Update task_feature_extractor with final dimensions
        self.task_feature_extractor = nn.Sequential(
            nn.Linear(task_feature_extractor_input_dim, self.narrative_generator_dim // 2),
            nn.GELU(),
            nn.Linear(self.narrative_generator_dim // 2, self.narrative_generator_dim)
        )

        for i in range(depth):
            self.blocks[i].mlp = GenGaussianMoELayer(
                in_features=embed_dim,
                hidden_features=mlp_dim,
                out_features=embed_dim,
                num_experts=num_experts,
                top_k=top_k,
                narrative_generator_dim=self.narrative_generator_dim,
                vae_latent_dim=self.vae_latent_dim,
                nhead=num_heads,
                use_vae=use_vae
            )

    def forward(self, x, pi_score=None, use_narrative_generator: bool = True):
        B = x.shape[0]
        
        # Process input through patch_embed first
        x_patch_embedded = self.patch_embed(x)
        
        # task_feature_extractor now processes the patch_embedded output
        # We take the mean across tokens to get a single feature vector per image
        task_features = self.task_feature_extractor(x_patch_embedded.mean(dim=1))
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x_patch_embedded), dim=1) # Use x_patch_embedded here
        
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        all_routing_info = []
        for block in self.blocks:
            x_norm1 = block.norm1(x)
            attn_output, _ = block.attn(x_norm1, x_norm1, x_norm1)
            x = x + block.dropout(attn_output)
            
            x_norm2 = block.norm2(x)
            mlp_output, routing_info = block.mlp(x_norm2, pi_score=pi_score, task_features=task_features.unsqueeze(1).expand(-1, x_norm2.shape[1], -1), use_narrative_generator=use_narrative_generator) 
            x = x + block.dropout(mlp_output)
            all_routing_info.append(routing_info)
            
        x = self.norm(x)
        
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)
        
        return logits, all_routing_info

    def zero_inactive_expert_grads(self, all_routing_info):
        with torch.no_grad():
            for i, block in enumerate(self.blocks):
                if isinstance(block.mlp, GenGaussianMoELayer):
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
        narrative_generator_param_ids = set()

        for block in self.blocks:
            if isinstance(block.mlp, GenGaussianMoELayer):
                gating_param_ids.add(id(block.mlp.expert_mus))
                gating_param_ids.add(id(block.mlp.expert_log_sigmas))
                for param in block.mlp.experts.parameters():
                    expert_param_ids.add(id(param))
        
        for param in self.task_feature_extractor.parameters():
            narrative_generator_param_ids.add(id(param))

        for block in self.blocks:
            if isinstance(block.mlp, GenGaussianMoELayer):
                for param in block.mlp.transformer_encoder.parameters():
                    narrative_generator_param_ids.add(id(param))
                for param in block.mlp.vae_encoder.parameters():
                    narrative_generator_param_ids.add(id(param))
                for param in block.mlp.vae_decoder.parameters():
                    narrative_generator_param_ids.add(id(param))


        gating_params = []
        expert_params = []
        base_params = []
        narrative_generator_params = []

        for _, param in self.named_parameters():
            param_id = id(param)
            if param_id in gating_param_ids:
                gating_params.append(param)
            elif param_id in expert_param_ids:
                expert_params.append(param)
            elif param_id in narrative_generator_param_ids:
                narrative_generator_params.append(param)
            else:
                base_params.append(param)

        return [
            {'params': base_params, 'name': 'base'},
            {'params': gating_params, 'name': 'gating'},
            {'params': expert_params, 'name': 'experts'},
            {'params': narrative_generator_params, 'name': 'narrative_generator'}
        ]
