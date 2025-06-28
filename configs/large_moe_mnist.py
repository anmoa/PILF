# Large MoE ViT Configuration for MNIST

# Model parameters
model_config = {
    'model_type': 'moe', 
    'img_size': 28,       # MNIST original size
    'patch_size': 4,      # 28/4 = 7x7 patches
    'in_channels': 3,     # Dataloader repeats channel to 3
    'num_classes': 10,
    'embed_dim': 64,      # Back to 64
    'depth': 3,           # Same as tiny
    'num_heads': 4,       # Same as tiny
    'mlp_dim': 64,        # Back to 64
    'dropout': 0.1,
    'num_experts': 16,    # Doubled from 8
    'top_k': 4,           # Doubled from 2
}
