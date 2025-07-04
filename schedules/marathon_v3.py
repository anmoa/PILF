# Marathon Rehearsal Schedule v3
# Configuration: 4 cycles of (4 epochs CIFAR10, 2 epochs MNIST, 3 epochs FashionMNIST)

schedule_config = {
    "name": "marathon_v3",
    "num_cycles": 4,
    "tasks": [
        ("CIFAR10", 5),
        ("MNIST", 2),
        ("FashionMNIST", 3),
        ("VALIDATE", 1),
    ],
    "val_datasets": ["CIFAR10", "MNIST", "FashionMNIST"],
    "train_config": {
        "batch_size": 1024,
        "accumulation_steps": 1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "output_dir": "output/marathon_v3",
    },
}
