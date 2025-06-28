import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import importlib.util
from sigma_pi import SigmaPI
from utils.training import train, validate
from utils.plotting import plot_metrics
from utils.strategies import UpdateStrategy, StandardUpdate, SelectiveUpdate, PisaUpdate
from models import VisionTransformer, MoEVisionTransformer, GPILMoEVisionTransformer
from torchvision import datasets, transforms
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

def setup_experiment(model_config: Dict[str, Any], train_config: Dict[str, Any], pi_config: Dict[str, Any], model_name_suffix: str = "", checkpoint_path: Optional[str] = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        raise RuntimeError("CUDA not available, exiting.")

    model_type = model_config.get('model_type')
    if 'img_size' not in model_config:
        model_config['img_size'] = 32

    if model_type == 'base':
        model = VisionTransformer(**{k: v for k, v in model_config.items() if k != 'model_type'}).to(device)
    elif model_type in ['moe', 'pilr_moe', 'sel_moe']:
        model = MoEVisionTransformer(**model_config).to(device)
    elif model_type == 'gpil_moe':
        model = GPILMoEVisionTransformer(**model_config).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if checkpoint_path:
        print(f"Loading checkpoint from: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Suffix: {model_name_suffix}")
    print(f"Total Trainable Parameters: {total_params/1e6:.2f}M")

    params: List[Dict[str, Any]]
    if isinstance(model, GPILMoEVisionTransformer):
        gating_params = [p for n, p in model.named_parameters() if 'expert_mus' in n or 'expert_log_sigmas' in n]
        expert_params = [p for n, p in model.named_parameters() if 'experts' in n]
        base_params = [p for n, p in model.named_parameters() if 'expert_mus' not in n and 'expert_log_sigmas' not in n and 'experts' not in n]
        
        params = [
            {'name': 'gating', 'params': gating_params},
            {'name': 'experts', 'params': expert_params},
            {'name': 'base', 'params': base_params}
        ]
        print("Using parameter groups for GPIL-MoE optimizer.")
    elif hasattr(model, 'get_param_groups'):
        params = model.get_param_groups()
        print("Using parameter groups for optimizer.")
    else:
        params = [{'params': model.parameters()}]

    optimizer = optim.AdamW(params, lr=train_config['learning_rate'], weight_decay=train_config['weight_decay'])
    loss_fn = nn.CrossEntropyLoss()
    pi_monitor = SigmaPI(**pi_config, device=device)

    return model, optimizer, loss_fn, pi_monitor, device

def get_dataloaders(dataset_name: str, batch_size: int, img_size: int, num_workers: int = 0):
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
    print(f"\n--- Validating at global step: {global_step} ---")
    stop_early = False
    for val_name, val_loaders in dataloaders.items():
        val_loss, val_acc, val_pi, val_surprise, val_tau, val_gating_tau = validate(
            model, device, val_loaders['val'], loss_fn, pi_monitor, dataset_name=f"{val_name} Validation"
        )
        metrics_to_log = [('loss', val_loss), ('acc', val_acc), ('pi', val_pi), ('surprise', val_surprise), ('tau', val_tau)]
        if val_gating_tau > 0:
            metrics_to_log.append(('gating_tau', val_gating_tau))
        
        for metric_name, metric_val in metrics_to_log:
            epoch_metrics.setdefault(f'{val_name}_val_{metric_name}', []).append((global_step, metric_val))

        # Early stopping check for this validation dataset
        if val_acc > 95.0 and val_pi > 0.95 and val_surprise < 0.1:
            print(f"Early stopping condition met for {val_name}: Acc > 95%, PI > 0.95, Surprise < 0.1")
            stop_early = True
            
    return stop_early

def run_schedule(schedule_config: Dict[str, Any], model_config: Dict[str, Any], schedule_path: str, model_config_path: str, checkpoint_path: Optional[str] = None):
    model_cfg = model_config
    train_cfg = schedule_config['train_config']
    pi_cfg = schedule_config.get('pi_config', {'alpha': 1.0, 'gamma': 0.5}) # Default pi_config
    output_base_dir = train_cfg['output_dir']
    
    current_time_iso = datetime.now().strftime("%Y%m%dT%H%M%S")
    schedule_name = os.path.basename(schedule_path).replace('.py', '')
    model_config_name = os.path.basename(model_config_path).replace('.py', '')
    file_prefix = f"{current_time_iso}-{schedule_name}-{model_config_name}"

    log_dir = os.path.join(output_base_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{file_prefix}.log")

    original_stdout = sys.stdout
    with open(log_file_path, 'w') as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        print(f"Logging output to: {log_file_path}")
        try:
            model, optimizer, loss_fn, pi_monitor, device = setup_experiment(
                model_cfg, train_cfg, pi_cfg, model_name_suffix=f"-{schedule_name}", checkpoint_path=checkpoint_path
            )

            model_type = model_cfg.get('model_type')
            update_strategy: UpdateStrategy
            if model_type == 'moe':
                update_strategy = StandardUpdate(optimizer)
            elif model_type == 'gpil_moe':
                if 'train_fn_kwargs' in model_cfg:
                    pisa_kwargs = model_cfg.get('train_fn_kwargs', {})
                    pisa_kwargs.update(pi_cfg)
                    update_strategy = PisaUpdate(optimizer, device=device, **pisa_kwargs)
                else:
                    update_strategy = StandardUpdate(optimizer)
            elif model_type in ['sel_moe']:
                update_strategy = SelectiveUpdate(optimizer)
            elif model_type == 'pilr_moe':
                    pisa_kwargs = model_cfg.get('train_fn_kwargs', {})
                    pisa_kwargs.update(pi_cfg)
                    update_strategy = PisaUpdate(optimizer, device=device, **pisa_kwargs)
            else:
                # Default to standard for 'base' model or if unspecified
                update_strategy = StandardUpdate(optimizer)

            all_datasets = list(set(schedule_config['val_datasets'] + [task[0] for task in schedule_config['tasks'] if task[0] != 'VALIDATE']))
            dataloaders = {
                name: dict(zip(['train', 'val'], get_dataloaders(name, train_cfg['batch_size'], model_cfg.get('img_size', 32), num_workers=0)))
                for name in all_datasets
            }
            val_dataloaders = {name: {'val': dataloaders[name]['val']} for name in schedule_config['val_datasets']}

            metrics: Dict[str, Any] = {
                'step': {}, 'epoch': {}, 
                'pilr_surprise': [], 'pilr_decisions': [], 
                'lr_mod': [], 'sigma': [],
                'gating_lr_mod': [], 'gating_sigma': [],
                'expert_lr_mod': [], 'expert_sigma': [],
                'all_top_indices': [], 'all_log_probs': [],
            }
            global_step, total_epochs = 0, 0
            
            num_cycles = schedule_config.get('num_cycles', 1)

            for cycle in range(1, num_cycles + 1):
                print(f"\n{'='*20} CYCLE {cycle}/{num_cycles} {'='*20}")
                stop_training = False
                for task_name, num_epochs_task in schedule_config['tasks']:
                    if isinstance(update_strategy, PisaUpdate):
                        update_strategy.reset_state()
                        print(f"PISA state reset for task: {task_name}")

                    if task_name == 'VALIDATE':
                        if validate_and_record(model, device, val_dataloaders, loss_fn, pi_monitor, metrics['epoch'], global_step):
                            stop_training = True
                            break
                        continue

                    # If num_epochs is None, use the one from train_config, else use the task-specific one
                    num_epochs = num_epochs_task if num_epochs_task is not None else train_cfg.get('epochs', 1)
                    print(f"\n--- Training on {task_name} for {num_epochs} epoch(s) ---")
                    for _ in range(num_epochs):
                        total_epochs += 1
                        global_step, epoch_summary = train(
                            model, device, dataloaders[task_name]['train'], optimizer, total_epochs, loss_fn, pi_monitor,
                            update_strategy, metrics['step'], metrics['epoch'], global_step, train_cfg['accumulation_steps']
                        )
                        # Append metrics from the epoch summary to the main metrics dict
                        metrics['pilr_surprise'].extend(epoch_summary.get('surprise_values', []))
                        metrics['pilr_decisions'].extend(epoch_summary.get('decisions', []))
                        metrics['lr_mod'].extend(epoch_summary.get('lr_mod', []))
                        metrics['sigma'].extend(epoch_summary.get('sigma', []))
                        metrics['gating_lr_mod'].extend(epoch_summary.get('gating_lr_mod', []))
                        metrics['gating_sigma'].extend(epoch_summary.get('gating_sigma', []))
                        metrics['expert_lr_mod'].extend(epoch_summary.get('expert_lr_mod', []))
                        metrics['expert_sigma'].extend(epoch_summary.get('expert_sigma', []))
                        metrics['all_top_indices'].extend(epoch_summary.get('all_top_indices', []))
                        metrics['all_log_probs'].extend(epoch_summary.get('all_log_probs', []))

                if stop_training:
                    print("Early stopping triggered. Moving to final checkpoint and plotting.")
                    break 

                # Final validation at the end of the cycle, unless the last task was already a validation
                if schedule_config['tasks'] and schedule_config['tasks'][-1][0] != 'VALIDATE':
                    if validate_and_record(model, device, val_dataloaders, loss_fn, pi_monitor, metrics['epoch'], global_step):
                        print("Early stopping triggered after final validation.")
                        break
                
                if num_cycles > 1:
                    checkpoint_dir = os.path.join(output_base_dir, 'checkpoints')
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"{file_prefix}-cycle_{cycle}.pth"))
                    print(f"\nCheckpoint for cycle {cycle} saved.")

            # Save final model
            checkpoint_dir = os.path.join(output_base_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"{file_prefix}-final.pth"))
            print(f"\nFinal checkpoint saved.")

            plot_metrics(
                step_metrics=metrics['step'],
                epoch_metrics=metrics['epoch'],
                output_dir=os.path.join(output_base_dir, 'img'),
                file_prefix=file_prefix,
                pilr_surprise_values=metrics.get('pilr_surprise'),
                pilr_decisions=metrics.get('pilr_decisions'),
                lr_mod_values=metrics.get('lr_mod'),
                sigma_values=metrics.get('sigma'),
                gating_lr_mod_values=metrics.get('gating_lr_mod'),
                gating_sigma_values=metrics.get('gating_sigma'),
                expert_lr_mod_values=metrics.get('expert_lr_mod'),
                expert_sigma_values=metrics.get('expert_sigma'),
                all_top_indices=metrics.get('all_top_indices'),
                all_log_probs=metrics.get('all_log_probs'),
            )
            print(f"\nPlots saved to: {os.path.abspath(os.path.join(output_base_dir, 'img'))}")
        finally:
            sys.stdout = original_stdout
            print(f"Training completed. Log saved to: {log_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Unified Training Runner for PILF, driven by schedule files.")
    parser.add_argument('--schedule', type=str, required=True, help="Path to a schedule configuration file.")
    parser.add_argument('--model-config', type=str, required=True, help="Path to a model configuration file.")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to a model checkpoint to start from.")
    args = parser.parse_args()

    # Load schedule config
    schedule_spec = importlib.util.spec_from_file_location("schedule", args.schedule)
    if schedule_spec is None or schedule_spec.loader is None: raise ImportError(f"Could not load schedule config from {args.schedule}")
    schedule_module = importlib.util.module_from_spec(schedule_spec)
    schedule_spec.loader.exec_module(schedule_module)
    
    # Load model config
    model_spec = importlib.util.spec_from_file_location("model_config", args.model_config)
    if model_spec is None or model_spec.loader is None: raise ImportError(f"Could not load model config from {args.model_config}")
    model_config_module = importlib.util.module_from_spec(model_spec)
    model_spec.loader.exec_module(model_config_module)

    run_schedule(
        schedule_module.schedule_config, 
        model_config_module.model_config,
        args.schedule,
        args.model_config,
        args.checkpoint
    )

if __name__ == "__main__":
    main()
