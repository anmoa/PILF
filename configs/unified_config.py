BASE_CONFIG = {
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
        "top_k": 4,
    },
    "router_configs": {
        "dense": {"router_type": "dense"},
        "moe": {"router_type": "moe"},
        "gaussian": {"router_type": "gaussian_moe"},
        "memory_gaussian": {
            "router_type": "memory_gaussian_moe",
            "gating_config": {
                "total_buffer_size": 128
            }
        },
    },
    "update_strategy_configs": {
        "standard": {"name": "Standard"},
        "selective": {"name": "Selective"},
        "smk": {"name": "SurpriseMinK", "min_k": 2},
    },
}
