# Base ViT Configuration for MNIST

# Model parameters
model_config = {
    'model_type': 'base',
    'img_size': 32,       # MNIST is resized to 32x32 by the dataloader
    'patch_size': 4,      # 32/4 = 8x8 patches
    'in_channels': 3,     # Dataloader repeats channel to 3
    'num_classes': 10,
    'embed_dim': 128,
    'depth': 6,
    'num_heads': 4,
    'mlp_dim': 256,
    'dropout': 0.1,
}

# Training parameters
train_config = {
    'epochs': 30, # Longer training for grokking
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
