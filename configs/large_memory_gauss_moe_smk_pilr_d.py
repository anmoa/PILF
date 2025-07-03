# Meta-Config for the new flagship model.
# This file demonstrates the "Strategy Chaining" pattern.

# 1. Defines the core model architecture
model_config = {
    'model_type': 'memory_gaussian_moe', # Changed to memory_gaussian_moe
    'img_size': 28,       # MNIST original size
    'patch_size': 4,      # 28/4 = 7x7 patches
    'in_channels': 3,     # Dataloader repeats channel to 3
    'num_classes': 10,
    'embed_dim': 64,
    'depth': 3,
    'num_heads': 4,
    'mlp_ratio': 1.0, # Derived from embed_dim * mlp_ratio
    'dropout': 0.1,
    'num_experts': 16,
    'top_k': 4,
}

# 2. Defines the chain of update strategies to be applied in order.
#    train.py will execute these as a pipeline.
train_strategy_config = {
    'strategies': [
        {
            'name': 'SurpriseMinK',
            'min_k': 4,
        },
        {
            'name': 'PILR_Dual', # Use the dual-adaptor PILR
        }
    ]
}

# 3. PILR-specific parameters, shared across strategies if needed.
pilr_config = {
    'gating_beta': 0.05, # Normal beta for gating
    'expert_beta': 0.0, # As per our findings, SMK replaces expert-side PILR.
    'gating_beta_warmup': 0.9,
    'warmup_steps': 50,
    'gating_modulation_power': 1.0,
    'gating_modulation_exponent': 1.0,
    'expert_modulation_power': 0.5,
    'expert_modulation_exponent': 2.0,
    'gating_inverse_ema': True,
    'gating_inverse_ema_k': 0.1,
}

# 4. Gating-specific loss configuration (DEPRECATED)
gating_loss_config = None

# New parameters for MemoryGaussianMoE
trainer_config = {
    'routing_history_length': 10,
    'historical_routing_loss_weight': 0.1,
}