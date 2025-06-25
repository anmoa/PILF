# MoE 16x4 ViT Configuration

# Model parameters
model_config = {
    'model_type': 'moe',
    'img_size': 32,
    'patch_size': 4,
    'in_channels': 3,
    'num_classes': 10,
    'embed_dim': 128,
    'depth': 6,
    'num_heads': 4,
    'mlp_dim': 32,  # mlp_dim for each expert
    'dropout': 0.1,
    'num_experts': 16,
    'top_k': 4,
}

# Training parameters
train_config = {
    'epochs': 20,
    'batch_size': 512,
    'accumulation_steps': 2,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'output_dir': 'output/ViT/',
}

# PI Monitor parameters
pi_config = {
    'alpha': 1.0,
    'gamma': 0.5,
}
