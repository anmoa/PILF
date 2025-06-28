schedule_config = {
    'train_config': {
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'batch_size': 64,
        'epochs': 1,
        'output_dir': 'output/ViT/null-hypothesis-test/',
        'accumulation_steps': 2,
    },
    'pi_config': {
        'alpha': 1.0,
        'gamma': 0.5,
    },
    'tasks': [
        ('MNIST', 10), # 充分学习任务A
        ('CIFAR10', 5), # 学习任务C
        ('MNIST', 1), # 再次学习任务A，观察再学习能力
        ('VALIDATE', 1), # 验证所有任务
        ('CIFAR10', 5), # 学习任务C
        ('MNIST', 1), # 再次学习任务A，观察再学习能力
    ],
    'val_datasets': ['CIFAR10', 'MNIST'],
    'num_cycles': 1,
}
