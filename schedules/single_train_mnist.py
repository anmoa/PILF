SCHEDULE = {
    "name": "single_mnist_pilf",
    "num_cycles": 1,
    "val_datasets": ["MNIST"],
    "tasks": [
        ("MNIST", 1),
    ],
    "train_config": {
        "batch_size": 512,
        "accumulation_steps": 1,
        "learning_rate": 1e-4,
        "gating_learning_rate": 1e-3,
        "output_dir": "output/pilf_test",
    },
    "model_config": {
        "name": "pilf_vit",
        "img_size": 32,
        "patch_size": 4,
        "embed_dim": 128,
        "depth": 4,
        "num_heads": 4,
        "num_classes": 10,
        "num_experts": 8,
        "top_k": 2,
    },
    "pilf_config": {
        "smk_min_k": 2,
        "buffer_size": 1024,
    },
    "pi_config": {
        "alpha": 1.0,
        "gamma": 0.5,
    },
}
