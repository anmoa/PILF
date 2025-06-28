# Large Selective MoE ViT Configuration for MNIST
# This model uses selective expert weight updates (inactive experts are frozen)
# but does NOT use the PILR-S dynamic learning rate scheduler.

model_config = {
    'model_type': 'sel_moe',  # Use sel_moe for selective update mode
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
