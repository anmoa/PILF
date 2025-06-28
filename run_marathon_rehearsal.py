import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import importlib.util
from sigma_pi import SigmaPI # type: ignore
from utils.training import train, validate
from utils.plotting import plot_metrics
from models.vit import VisionTransformer, MoEVisionTransformer
from torchvision import datasets, transforms # type: ignore
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

class RepeatChannel(object):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat(3, 1, 1)

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

def setup_experiment(config: Any, base_model_name: str, model_name_suffix: str = "", checkpoint_path: Optional[str] = None):
    model_cfg = config.model_config
    train_cfg = config.train_config
    pi_cfg = config.pi_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        raise RuntimeError("CUDA not available, exiting.")

    model_type = model_cfg.get('model_type')
    # Ensure img_size is consistent, e.g., 32 for CIFAR/SVHN compatibility
    model_cfg['img_size'] = 32 

    if model_type == 'base':
        vit_model_cfg = model_cfg.copy()
        vit_model_cfg.pop('model_type', None)
        model = VisionTransformer(**vit_model_cfg).to(device)
    elif model_type in ['moe', 'pilr_moe']:
        model = MoEVisionTransformer(**model_cfg).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if checkpoint_path:
        print(f"Loading checkpoint from: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    model_name = f"{base_model_name}{model_name_suffix}"
    print(f"Model: {model_name}")
    print(f"Total Trainable Parameters: {total_params/1e6:.2f}M")

    optimizer = optim.AdamW(model.parameters(), lr=train_cfg['learning_rate'], weight_decay=train_cfg['weight_decay'])
    loss_fn = nn.CrossEntropyLoss()
    pi_monitor = SigmaPI(**pi_cfg, device=device)

    return model, optimizer, loss_fn, pi_monitor, device, model_name

def get_dataloaders(dataset_name: str, batch_size: int, img_size: int, num_workers: int = 0):
    # Standard 3-channel transforms for CIFAR/SVHN
    transform_3_channel_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_3_channel_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 1-channel to 3-channel transforms for MNIST/FashionMNIST
    transform_1_channel = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        RepeatChannel()
    ])

    dataset_map = {
        'CIFAR10': (datasets.CIFAR10, {'train': True, 'transform': transform_3_channel_train}, {'train': False, 'transform': transform_3_channel_test}),
        'SVHN': (datasets.SVHN, {'split': 'train', 'transform': transform_3_channel_train}, {'split': 'test', 'transform': transform_3_channel_test}),
        'MNIST': (datasets.MNIST, {'train': True, 'transform': transform_1_channel}, {'train': False, 'transform': transform_1_channel}),
        'FASHIONMNIST': (datasets.FashionMNIST, {'train': True, 'transform': transform_1_channel}, {'train': False, 'transform': transform_1_channel}),
    }
    
    dataset_class, train_kwargs, test_kwargs = dataset_map[dataset_name.upper()]
    data_dir = f"temp_data/{dataset_name.upper()}"
    
    train_dataset = dataset_class(data_dir, download=True, **train_kwargs)
    val_dataset = dataset_class(data_dir, download=True, **test_kwargs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader

def validate_and_record(model, device, dataloaders, loss_fn, pi_monitor, epoch_metrics, global_step):
    """Helper function to run validation on all datasets and record metrics."""
    print(f"\n--- Validating at global step: {global_step} ---")
    for val_name, val_loaders in dataloaders.items():
        val_loss, val_acc, val_pi, val_surprise, val_tau = validate(
            model, device, val_loaders['val'], loss_fn, pi_monitor, dataset_name=f"{val_name} Validation"
        )
        for metric_name, metric_val in [('loss', val_loss), ('acc', val_acc), ('pi', val_pi), ('surprise', val_surprise), ('tau', val_tau)]:
            epoch_metrics.setdefault(f'{val_name}_val_{metric_name}', []).append((global_step, metric_val))

def main():
    parser = argparse.ArgumentParser(description="Run a marathon rehearsal experiment across multiple datasets.")
    parser.add_argument('--config', type=str, required=True, help="Path to the model configuration file.")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="Path to a model checkpoint to start from.")
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location(name="config", location=args.config)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for module at: {args.config}")
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    train_cfg = config.train_config
    output_base_dir = train_cfg['output_dir']
    
    current_time_iso = datetime.now().strftime("%Y%m%dT%H%M%S")
    config_name = os.path.basename(args.config).replace('.py', '').replace('_', '-')
    file_prefix = f"{current_time_iso}_{config_name}-marathon-rehearsal"

    log_dir = os.path.join(output_base_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{file_prefix}.log")

    original_stdout = sys.stdout
    with open(log_file_path, 'w') as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        print(f"Logging output to: {log_file_path}")

        try:
            model, optimizer, loss_fn, pi_monitor, device, model_name = setup_experiment(
                config, base_model_name=config_name, model_name_suffix="-marathon-rehearsal", checkpoint_path=args.checkpoint_path
            )
            
            datasets_to_load = ['CIFAR10', 'MNIST', 'FashionMNIST', 'SVHN']
            dataloaders = {
                name: dict(zip(['train', 'val'], get_dataloaders(name, train_cfg['batch_size'], config.model_config['img_size'])))
                for name in datasets_to_load
            }

            metrics = {
                'step': {}, 'epoch': {}, 'pilr_surprise': [], 'pilr_decisions': [], 'lr_mod': []
            }
            global_step = 0
            total_epochs = 0
            
            use_pilr = (config.model_config.get('model_type') == 'pilr_moe')
            train_fn_kwargs = train_cfg.get('train_fn_kwargs', {})
            task_schedule = [('CIFAR10', 5), ('MNIST', 1), ('FashionMNIST', 1), ('SVHN', 1)]
            num_marathon_cycles = 5

            for cycle in range(1, num_marathon_cycles + 1):
                print(f"\n{'='*20} MARATHON CYCLE {cycle}/{num_marathon_cycles} {'='*20}")
                
                for task_name, num_epochs in task_schedule:
                    print(f"\n--- Training on {task_name} for {num_epochs} epoch(s) ---")
                    for _ in range(num_epochs):
                        total_epochs += 1
                        train_result = train(
                            model=model, optimizer=optimizer, loss_fn=loss_fn, pi_monitor=pi_monitor,
                            device=device, train_loader=dataloaders[task_name]['train'], epoch=total_epochs,
                            step_metrics=metrics['step'], epoch_metrics=metrics['epoch'], global_step=global_step,
                            accumulation_steps=train_cfg['accumulation_steps'],
                            use_pilr=use_pilr, **train_fn_kwargs
                        )
                        
                        global_step, new_surprises, new_decisions, new_lr_mods = train_result
                        if new_surprises:
                            metrics['pilr_surprise'].extend(new_surprises)
                        if new_decisions:
                            metrics['pilr_decisions'].extend(new_decisions)
                        if new_lr_mods:
                            metrics['lr_mod'].extend(new_lr_mods)
                    
                    validate_and_record(model, device, dataloaders, loss_fn, pi_monitor, metrics['epoch'], global_step)

                # Save checkpoint at the end of each marathon cycle
                checkpoint_dir = os.path.join(output_base_dir, 'checkpoints')
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"{file_prefix}-cycle_{cycle}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"\nCheckpoint for cycle {cycle} saved to: {checkpoint_path}")

            # Final plot
            plot_metrics(metrics['step'], metrics['epoch'], os.path.join(output_base_dir, 'img'), file_prefix=file_prefix,
                         pilr_surprise_values=metrics['pilr_surprise'], pilr_decisions=metrics['pilr_decisions'], lr_mod_values=metrics['lr_mod'])
            print(f"\nPlots saved to: {os.path.abspath(os.path.join(output_base_dir, 'img'))}")

        finally:
            sys.stdout = original_stdout
            print(f"Training completed. Log saved to: {log_file_path}")

if __name__ == "__main__":
    main()
