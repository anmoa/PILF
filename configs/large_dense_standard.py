# Core Configuration: Large-Dense-Standard
# Absolute baseline: A standard Vision Transformer without any MoE or PILF mechanisms.

model_config = {
    'model_type': 'dense',
    'img_size': 28,
    'patch_size': 4,
    'in_channels': 3,
    'num_classes': 10,
    'embed_dim': 64,
    'depth': 3,
    'num_heads': 4,
    'mlp_dim': 64,
    'dropout': 0.1,
}

train_strategy_config = {
    'strategies': [
        {
            'name': 'Standard',
        }
    ]
}
