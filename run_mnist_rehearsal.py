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
    transform_3_channel_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_3_channel_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    mnist_transforms = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), RepeatChannel()]
    if img_size != 28:
        mnist_transforms.insert(0, transforms.Resize((img_size, img_size)))
    transform_1_channel = transforms.Compose(mnist_transforms)

    if dataset_name.upper() == 'CIFAR10':
        data_dir = "temp_data/CIFAR10"
        os.makedirs(data_dir, exist_ok=True)
        is_downloaded = os.path.exists(os.path.join(data_dir, 'cifar-10-batches-py'))
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=not is_downloaded, transform=transform_3_channel_train)
        val_dataset = datasets.CIFAR10(data_dir, train=False, download=not is_downloaded, transform=transform_3_channel_test)
    elif dataset_name.upper() == 'SVHN':
        data_dir = "temp_data/SVHN"
        os.makedirs(data_dir, exist_ok=True)
        train_file = os.path.join(data_dir, 'train_32x32.mat')
        test_file = os.path.join(data_dir, 'test_32x32.mat')
        train_dataset = datasets.SVHN(data_dir, split='train', download=not os.path.exists(train_file), transform=transform_3_channel_train)
        val_dataset = datasets.SVHN(data_dir, split='test', download=not os.path.exists(test_file), transform=transform_3_channel_test)
    elif dataset_name.upper() == 'MNIST':
        data_dir = "temp_data/MNIST"
        os.makedirs(data_dir, exist_ok=True)
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform_1_channel)
        val_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform_1_channel)
    elif dataset_name.upper() == 'FASHIONMNIST':
        data_dir = "temp_data/FashionMNIST"
        os.makedirs(data_dir, exist_ok=True)
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform_1_channel)
        val_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform_1_channel)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader

def main():
    parser = argparse.ArgumentParser(description="Run a rehearsal experiment on MNIST/FashionMNIST.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file.")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="Path to a model checkpoint to load for rehearsal.")
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location(name="config", location=args.config)
    if spec is None:
        raise ImportError(f"Could not load spec for module at: {args.config}")
    config = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Could not get loader from spec for module at: {args.config}")
    spec.loader.exec_module(config)

    train_cfg = config.train_config
    output_base_dir = train_cfg['output_dir']
    
    current_time_iso = datetime.now().strftime("%Y%m%dT%H%M%S")
    config_name = os.path.basename(args.config).replace('.py', '').replace('_', '-')
    
    sigma_value_from_config = train_cfg.get('train_fn_kwargs', {}).get('sigma_threshold')
    if sigma_value_from_config is not None:
        sigma_value_str = f"{sigma_value_from_config:.1f}".replace('.', '_')
    else:
        sigma_value_str = "control"

    file_prefix = f"{current_time_iso}_{config_name}"
    if sigma_value_str != "control":
        file_prefix += f"_sigma_{sigma_value_str}"
    file_prefix += "-mnist-rehearsal"

    log_dir = os.path.join(output_base_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = f"{file_prefix}.log"
    log_file_path = os.path.join(log_dir, log_file_name)

    original_stdout = sys.stdout
    log_file = open(log_file_path, 'w')
    sys.stdout = Tee(original_stdout, log_file)
    print(f"Logging output to: {log_file_path}")

    try:
        model, optimizer, loss_fn, pi_monitor, device, model_name = setup_experiment(
            config, base_model_name=config_name, model_name_suffix="-mnist-rehearsal", checkpoint_path=args.checkpoint_path
        )
        
        train_dataset_name = 'FashionMNIST'
        val_dataset_name = 'MNIST'

        print(f"Rehearsal Training on: {train_dataset_name}, Validating on: {val_dataset_name}")

        model_img_size = config.model_config['img_size']
        train_loader, val_loader = get_dataloaders(train_dataset_name, train_cfg['batch_size'], model_img_size)

        val_loaders = {
            f"{train_dataset_name}_Val": train_loader,
            f"{val_dataset_name}_OOD_Val": val_loader
        }

        use_pilr = (config.model_config.get('model_type') == 'pilr_moe')
        train_fn_kwargs = train_cfg.get('train_fn_kwargs', {})
        
        step_metrics: Dict[str, List[Tuple[int, float]]] = {}
        epoch_metrics: Dict[str, List[Tuple[int, float]]] = {}
        
        train_result = train(
            model=model, optimizer=optimizer, loss_fn=loss_fn, pi_monitor=pi_monitor,
            device=device, train_loader=train_loader, epoch=1,
            step_metrics=step_metrics, epoch_metrics=epoch_metrics, global_step=0,
            accumulation_steps=train_cfg['accumulation_steps'],
            use_pilr=use_pilr, **train_fn_kwargs
        )
        
        _, surprise_values, decisions, lr_mods = train_result

        checkpoint_dir = os.path.join(output_base_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_name = f"{file_prefix}-final.pth"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"\nFinal checkpoint saved to: {checkpoint_path}")

        plot_metrics(step_metrics, epoch_metrics, os.path.join(output_base_dir, 'img'), file_prefix=file_prefix,
                     pilr_surprise_values=surprise_values, pilr_decisions=decisions, lr_mod_values=lr_mods)
        print(f"\nPlots saved to: {os.path.abspath(os.path.join(output_base_dir, 'img'))}")

    finally:
        sys.stdout = original_stdout
        log_file.close()
        print(f"Training completed. Log saved to: {log_file_path}")

if __name__ == "__main__":
    main()
