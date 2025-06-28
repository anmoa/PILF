# Large PILR-S ViT Configuration for MNIST

# Model parameters
model_config = {
    'model_type': 'pilr_moe', # Use pilr_moe for PILR-S mode
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

# Training parameters
train_config = {
    'epochs': 30, # Longer training for grokking
    'batch_size': 1024, # Doubled from 512
    'accumulation_steps': 1, # Adjusted for larger batch size
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'output_dir': 'output/ViT/',
    'train_fn_kwargs': {
        'pilr_mode': 'lr_scheduler', # Enable LR scheduler mode for PILR-S
        'sigma_threshold': 1.0,     # Sigma threshold for Gaussian modulation (reduced for stricter modulation)
        'initial_surprise_ema': 0.0 # Initial EMA for surprise
    }
}

# PI Monitor parameters
pi_config = {
    'alpha': 1.0,
    'gamma': 0.5,
}
