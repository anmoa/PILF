# Core Configuration: Large-GaussMoE-SMK-PILR-S
# Static Flagship: Verifies the pure effect of Gaussian routing + SMK, without dynamic LR.

model_config = {
    'model_type': 'gaussian_moe',
    'img_size': 28,
    'patch_size': 4,
    'in_channels': 3,
    'num_classes': 10,
    'embed_dim': 64,
    'depth': 3,
    'num_heads': 4,
    'mlp_dim': 64,
    'dropout': 0.1,
    'num_experts': 16,
    'top_k': 4,
}

train_strategy_config = {
    'strategies': [
        {
            'name': 'SurpriseMinK',
            'min_k': 4,
        },
        {
            'name': 'PILR_Dual',
        }
    ]
}

pilr_config = {
    'gating_initial_var': 5.0,
    'expert_initial_var': 0.8,
    'gating_beta': 0.0, # PILR-S for gating (static sigma)
    'expert_beta': 0.0, # PILR-S for experts (static sigma), effectively disabled by SMK
    'gating_beta_warmup': 0.0, # No warmup for static beta
    'warmup_steps': 0,
    'gating_modulation_power': 1.0,
    'gating_modulation_exponent': 1.0,
    'expert_modulation_power': 0.5,
    'expert_modulation_exponent': 2.0,
}
