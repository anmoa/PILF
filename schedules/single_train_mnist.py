# Single Training Schedule for MNIST

schedule_config = {
    # num_cycles is 1 for a single training run.
    'num_cycles': 1,

    # List of datasets for validation.
    'val_datasets': ['MNIST', 'FashionMNIST'],

    # A single task defining the training run.
    'tasks': [
        ('MNIST', 30),
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
