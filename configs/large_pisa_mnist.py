# Large PILR-S ViT Configuration with Dual PISA for MNIST
# This model uses a PISA adaptor to dynamically adjust sigma for the
# gating network and the expert networks independently.

model_config = {
    'model_type': 'pilr_moe',
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
        'expert_initial_var': 1.0,
        'expert_beta': 0.1,
        'gating_initial_var': 0.5,
        'gating_beta': 0.1,
    }
}
