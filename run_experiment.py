import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
import importlib.util
from sigma_pi import SigmaPI # ty
from utils.training import train, validate
from utils.plotting import plot_metrics
from models.vit import VisionTransformer, MoEVisionTransformer
from torchvision import datasets, transforms # type: ignore
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

class RepeatChannel(object):
    def __call__(self, x):
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
    def isatty(self):
        return False

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
        print(f"Loading checkppe: ignoreoint from: {checkpoint_path}")
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

def execute_training_loop(
    model: nn.Module, optimizer: optim.Optimizer, loss_fn: nn.Module, pi_monitor: SigmaPI, device: torch.device, model_name: str,
    train_loader: DataLoader, val_loaders: Dict[str, DataLoader],
    epochs: int, accumulation_steps: int, output_dir: str,
    use_pilr: bool,
    start_epoch: int = 1, global_step: int = 0,
    train_fn_kwargs: Dict[str, Any] = {}
) -> Tuple[nn.Module, int, Dict[str, List[Tuple[int, float]]], Dict[str, List[Tuple[int, float]]], List[Dict[str, Any]]]:
    metric_keys = ['loss', 'acc', 'pi', 'surprise', 'tau', 'lr_mod']
    step_metrics: Dict[str, List[Tuple[int, float]]] = {f'train_{key}': [] for key in metric_keys}
    
    # Correctly type epoch_metrics to handle mixed value types
    epoch_metrics: Dict[str, List[Any]] = {f'train_{key}': [] for key in metric_keys}
    for val_set_name in val_loaders.keys():
        for key in ['loss', 'acc', 'pi', 'surprise', 'tau']:
            epoch_metrics[f'{val_set_name}_{key}'] = []
    
    if use_pilr and train_fn_kwargs.get('pilr_mode', 'gate') == 'gate':
        epoch_metrics['pilr_surprise_values'] = []
        epoch_metrics['pilr_decisions'] = []

    all_metrics_data: List[Dict[str, Any]] = []

    best_val_acc = 0.0
    patience_counter = 0
    patience = 3

    for epoch in range(start_epoch, start_epoch + epochs):
        print(f"\n--- Epoch {epoch} ---")
        
        train_output = train(
            model, device, train_loader, optimizer, epoch, loss_fn, 
            pi_monitor, step_metrics, epoch_metrics, global_step, 
            accumulation_steps, use_pilr=use_pilr, sigma_threshold=train_fn_kwargs.get('sigma_threshold', 1.0), **train_fn_kwargs
        )

        if isinstance(train_output, tuple):
            global_step, pilr_surprise_values, pilr_decisions = train_output
            epoch_metrics['pilr_surprise_values'].extend(pilr_surprise_values)
            epoch_metrics['pilr_decisions'].extend(pilr_decisions)
        else:
            global_step = train_output

        current_val_pi = 0.0
        current_val_acc = 0.0
        
        epoch_data: Dict[str, Any] = {
            "epoch": epoch,
            "global_step_end": global_step - 1,
            "train_metrics": {key: epoch_metrics[f'train_{key}'][-1][1] if epoch_metrics[f'train_{key}'] else None for key in metric_keys},
            "val_metrics": {}
        }

        for name, loader in val_loaders.items():
            val_loss, val_acc, val_pi, val_surprise, val_tau = validate(model, device, loader, loss_fn, pi_monitor, dataset_name=name)
            epoch_metrics[f'{name}_loss'].append((global_step - 1, val_loss))
            epoch_metrics[f'{name}_acc'].append((global_step - 1, val_acc))
            epoch_metrics[f'{name}_pi'].append((global_step - 1, val_pi))
            epoch_metrics[f'{name}_surprise'].append((global_step - 1, val_surprise))
            epoch_metrics[f'{name}_tau'].append((global_step - 1, val_tau))
            
            # Ensure correct typing for the val_metrics dictionary
            epoch_data["val_metrics"][name] = {
                "loss": val_loss, "acc": val_acc, "pi": val_pi, 
                "surprise": val_surprise, "tau": val_tau
            }
            
            if name.startswith('MNIST_Val'):
                current_val_pi = val_pi
                current_val_acc = val_acc

        all_metrics_data.append(epoch_data)

        if current_val_pi >= 0.95 and current_val_acc >= 99.0:
            patience_counter += 1
            print(f"Early stopping condition met: Val PI >= 0.95 and Val Acc >= 99.0. Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Sustained high performance for {patience} epochs. Stopping training early.")
                checkpoint_dir = os.path.join(output_dir, 'checkpoints')
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_name = f"{model_name}-final-epoch_{epoch}.pth"
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
                torch.save(model.state_dict(), checkpoint_path)
                print(f"\nFinal checkpoint saved to: {checkpoint_path}")
                break
        else:
            patience_counter = 0

    return model, global_step, step_metrics, epoch_metrics, all_metrics_data

def run_experiment(config: Any, config_path: str, task_name_suffix: str = "", checkpoint_path: Optional[str] = None):
    train_cfg = config.train_config
    output_base_dir = train_cfg['output_dir']
    
    current_time_iso = datetime.now().strftime("%Y%m%dT%H%M%S")
    config_name = os.path.basename(config_path).replace('.py', '').replace('_', '-')
    
    sigma_value_from_config = train_cfg.get('train_fn_kwargs', {}).get('sigma_threshold')
    if sigma_value_from_config is not None:
        sigma_value_str = f"{sigma_value_from_config:.1f}".replace('.', '_')
    else:
        sigma_value_str = "control"

    file_prefix = f"{current_time_iso}_{config_name}"
    if sigma_value_str != "control":
        file_prefix += f"_sigma_{sigma_value_str}"
    file_prefix += task_name_suffix

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
            config, base_model_name=config_name, model_name_suffix=f"-{task_name_suffix}", checkpoint_path=checkpoint_path
        )
        
        train_dataset_name = train_cfg.get('train_dataset', 'CIFAR10')
        val_dataset_name = train_cfg.get('val_dataset', 'CIFAR10')
        ood_dataset_name = train_cfg.get('ood_dataset', 'SVHN')

        print(f"Training on: {train_dataset_name}, Validating on: {val_dataset_name}, OOD on: {ood_dataset_name}")

        model_img_size = config.model_config['img_size']
        train_loader, val_loader = get_dataloaders(train_dataset_name, train_cfg['batch_size'], model_img_size)
        _, ood_val_loader = get_dataloaders(ood_dataset_name, train_cfg['batch_size'], model_img_size)

        val_loaders = {
            f"{val_dataset_name}_Val": val_loader,
            f"{ood_dataset_name}_OOD_Val": ood_val_loader
        }

        use_pilr = (config.model_config.get('model_type') == 'pilr_moe')
        train_fn_kwargs = train_cfg.get('train_fn_kwargs', {})
        
        model, _, step_metrics, epoch_metrics, all_metrics_data = execute_training_loop(
            model=model, optimizer=optimizer, loss_fn=loss_fn, pi_monitor=pi_monitor,
            device=device, model_name=model_name,
            train_loader=train_loader, val_loaders=val_loaders,
            epochs=train_cfg['epochs'], accumulation_steps=train_cfg['accumulation_steps'],
            use_pilr=use_pilr,
            output_dir=output_base_dir,
            train_fn_kwargs=train_fn_kwargs
        )
        
        checkpoint_dir = os.path.join(output_base_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_name = f"{file_prefix}-final.pth"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"\nFinal checkpoint saved to: {checkpoint_path}")

        json_output_dir = os.path.join(output_base_dir, 'json_results')
        os.makedirs(json_output_dir, exist_ok=True)
        json_file_name = f"{file_prefix}.json"
        json_file_path = os.path.join(json_output_dir, json_file_name)
        
        json_data = {
            "metadata": {
                "model_name": model_name,
                "sigma_factor": train_fn_kwargs.get('sigma_threshold', 1.0),
                "train_dataset": train_dataset_name,
                "val_datasets": list(val_loaders.keys())
            },
            "epoch_data": all_metrics_data,
            "step_metrics": {k: v for k, v in step_metrics.items() if v}
        }
        
        with open(json_file_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        print(f"\nAll metrics saved to JSON: {json_file_path}")

        # Prepare kwargs for plot_metrics, ensuring correct types
        plot_metrics_kwargs: Dict[str, Any] = {}
        if use_pilr:
            pilr_mode = train_fn_kwargs.get('pilr_mode', 'gate')
            if pilr_mode == 'gate':
                plot_metrics_kwargs['pilr_surprise_values'] = epoch_metrics.get('pilr_surprise_values')
                plot_metrics_kwargs['pilr_decisions'] = epoch_metrics.get('pilr_decisions')
            elif pilr_mode == 'lr_scheduler':
                plot_metrics_kwargs['lr_mod_values'] = epoch_metrics.get('train_lr_mod')

        # Cast epoch_metrics to the type expected by plot_metrics
        plot_epoch_metrics: Dict[str, List[Tuple[int, float]]] = {
            k: v for k, v in epoch_metrics.items() 
            if isinstance(v, list) and all(isinstance(item, tuple) for item in v)
        }

        plot_metrics(step_metrics, plot_epoch_metrics, os.path.join(output_base_dir, 'img'), file_prefix=file_prefix, **plot_metrics_kwargs)
        print(f"\nPlots saved to: {os.path.abspath(os.path.join(output_base_dir, 'img'))}")

    finally:
        sys.stdout = original_stdout
        log_file.close()
        print(f"Training completed. Log saved to: {log_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Run a ViT experiment from a configuration file.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file.")
    parser.add_argument('--task', type=str, default="single", help="Task name for logging (e.g., 'CIFAR10', 'continual').")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="Path to a model checkpoint to load.")
    parser.add_argument('--train_dataset', type=str, default='CIFAR10', help="Dataset for training.")
    parser.add_argument('--val_dataset', type=str, default='CIFAR10', help="Dataset for validation.")
    parser.add_argument('--ood_dataset', type=str, default='SVHN', help="Dataset for out-of-distribution validation.")
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location(name="config", location=args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    config.train_config['train_dataset'] = args.train_dataset
    config.train_config['val_dataset'] = args.val_dataset
    config.train_config['ood_dataset'] = args.ood_dataset
    
    run_experiment(config, config_path=args.config, task_name_suffix=f"-{args.task}", checkpoint_path=args.checkpoint_path)

if __name__ == "__main__":
    main()
