# Tiny PILR-S ViT Configuration for MNIST

# Model parameters
model_config = {
    'model_type': 'pilr_moe', # Use pilr_moe for PILR-S mode
    'img_size': 28,       # MNIST original size
    'patch_size': 4,      # 28/4 = 7x7 patches
    'in_channels': 3,     # Dataloader repeats channel to 3
    'num_classes': 10,
    'embed_dim': 64,      # Halved from 128
    'depth': 3,           # Halved from 6
    'num_heads': 4,
    'mlp_dim': 64,
    'dropout': 0.1,
    'num_experts': 8,    # Number of experts in MoE
    'top_k': 2,           # Top-K routing
}
