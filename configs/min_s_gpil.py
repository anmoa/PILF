# Large G2PIL-MoE ViT Configuration for MNIST with Surprise-Min-K Update Strategy

model_config = {
    'model_type': 'gpil_moe', # Use the GPIL-MoE model
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
    'top_k': 2,

    # PISA-driven routing inhibition parameters
    'ood_inhibition_c': 2.0,
    'routing_pisa_initial_var': 1.0,
    'routing_pisa_beta': 0.1,
    
    # Keyword arguments for the PisaUpdate strategy.
    # The presence of 'gating_' and 'expert_' prefixes triggers dual mode.
    'train_fn_kwargs': {
        'expert_initial_var': 1.2,
        'expert_beta': 0.01,
        'gating_initial_var': 1.2,
        'gating_beta': 0.01,
    }
}

# Training parameters
train_config = {
    'update_strategy': 'surprise_min_k',
    'min_k': 4, # Select top 2 experts with the lowest surprise
}
