schedule_config = {
    "train_config": {
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "batch_size": 64,
        "epochs": 1,
        "output_dir": "output/marathon-v4/",
        "accumulation_steps": 2,
    },
    "pi_config": {
        "alpha": 1.0,
        "gamma": 0.5,
    },
    "tasks": [
        ("MNIST", 10),
        ("FashionMNIST", 4),
        ("MNIST", 1),
        ("FashionMNIST", 4),
        ("VALIDATE", 1),
        ("MNIST", 1),
        ("CIFAR10", 3),
        ("MNIST", 1),
        ("FashionMNIST", 1),
    ],
    "val_datasets": ["CIFAR10", "MNIST", "FashionMNIST"],
    "num_cycles": 1,
}
