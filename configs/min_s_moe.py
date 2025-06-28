# Large MoE ViT Configuration for MNIST with Surprise-Min-K Update Strategy

# Model parameters
model_config = {
    'model_type': 'moe', 
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

# Training parameters
train_config = {
    'update_strategy': 'surprise_min_k',
    'min_k': 2, # Select top 2 experts with the lowest surprise
}
