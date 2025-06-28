# Marathon Rehearsal Schedule v2
# Configuration: 4 cycles of (4 epochs CIFAR10, 2 epochs MNIST, 3 epochs FashionMNIST, 5 epochs SVHN)

schedule_config = {
    # Number of times to repeat the entire task sequence.
    'num_cycles': 4,

    # List of datasets to be used for validation after each training task.
    'val_datasets': ['CIFAR10', 'MNIST', 'FashionMNIST', 'SVHN'],

    # Sequence of training tasks to be executed in each cycle.
    # Each task is a tuple of (dataset_name, num_epochs_to_train).
    'tasks': [
        ('CIFAR10', 4),
        ('MNIST', 2),
        ('FashionMNIST', 3),
        ('SVHN', 5),
    ],

    # Training parameters
    'train_config': {
        'batch_size': 1024,
        'accumulation_steps': 1,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'output_dir': 'output/ViT/',
    },

    # PI Monitor parameters
    'pi_config': {
        'alpha': 1.0,
        'gamma': 0.5,
    }
}
