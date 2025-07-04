BASE_CONFIG = {
    "train_config": {
        "batch_size": 64,
        "accumulation_steps": 4,
    },
    "model_config": {
        "name": "large_vit",
        "img_size": 32,
        "patch_size": 4,
        "embed_dim": 32,
        "depth": 3,
        "num_heads": 4,
        "mlp_ratio": 1.0,
        "num_classes": 10,
        "num_experts": 8,
        "top_k": 2,
    },
    "router_configs": {
        "linear": {"router_type": "standard"},
        "gauss": {"router_type": "gaussian_moe"},
        "memory_gauss": {
            "router_type": "memory_gaussian_moe",
            "memory_beta": 0.9,
            "rehearsal_weight": 0.5,
        },
    },
    "train_strategy_configs": {
        "standard": {"strategies": [{"name": "Standard"}]},
        "smk": {
            "strategies": [
                {
                    "name": "SurpriseMinK",
                    "min_k": 1,
                    "ema_factor": 0.99,
                    "ema_update_freq": 10,
                }
            ]
        },
    },
    "pilr_configs": {
        "none": None,
        "pilr_s": {"name": "PILR_S", "base_lr": 1e-3, "sensitivity": 1.0},
        "pilr_d": {
            "name": "PILR_D",
            "base_lr": 1e-3,
            "sensitivity": 0.1,
            "ema_factor": 0.95,
            "expert_initial_var": 1.0,
        },
    },
    "pi_config": {
        "alpha": 1.0,
        "gamma": 0.5,
    },
}
