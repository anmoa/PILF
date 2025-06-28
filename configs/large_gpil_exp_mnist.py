# Large PILR-S ViT Configuration with Dual PISA and Exponential Modulation for MNIST
# This model uses a PISA adaptor to dynamically adjust sigma for the
# gating network and the expert networks independently.
# It uses an exponential modulation curve instead of a Gaussian one.

model_config = {
    'model_type': 'gpil_moe',
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
    
    # Keyword arguments for the PisaUpdate strategy.
    # The presence of 'gating_' and 'expert_' prefixes triggers dual mode.
    'train_fn_kwargs': {
        'expert_initial_var': 0.8,
        'expert_beta': 0.0,  # Disable PISA for experts (fixed sigma)
        'gating_initial_var': 5.0,
        'gating_beta': 0.05, # Normal beta for gating
        'gating_beta_warmup': 0.9, # High beta for rapid adaptation during warmup
        'warmup_steps': 50, # Number of steps for the warmup period
        # Gating network: Inverse relationship (higher surprise -> lower LR)
        'gating_modulation_power': 1.0,
        'gating_modulation_exponent': 1.0,
        # Expert network: Bell curve (medium surprise -> highest LR)
        'expert_modulation_power': 0.5,
        'expert_modulation_exponent': 2.0,
    }
}

train_config = {
    'update_strategy': 'surprise_min_k',
    'min_k': 4,
}
