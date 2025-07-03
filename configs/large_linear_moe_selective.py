# Core Configuration: Large-LinearMoE-Selective
# Classic MoE baseline: Uses simple linear gating and selective update.

model_config = {
    'model_type': 'moe',
    'img_size': 28,
    'patch_size': 4,
    'in_channels': 3,
    'num_classes': 10,
    'embed_dim': 64,
    'depth': 3,
    'num_heads': 4,
    'mlp_ratio': 1.0, # Derived from embed_dim * mlp_ratio
    'dropout': 0.1,
    'num_experts': 16,
    'top_k': 4,
}

train_strategy_config = {
    'strategies': [
        {
            'name': 'Selective',
        }
    ]
}
