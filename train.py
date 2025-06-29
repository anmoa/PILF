import atexit
import importlib
import os
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sigma_pi import SigmaPI
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import GaussianMoEVisionTransformer, MoEVisionTransformer, VisionTransformer
from utils.datasets import get_dataset
from utils.plotting import plot_expert_heatmap, plot_expert_scatter
from utils.strategies import (
    PILRStrategy,
    SelectiveUpdateStrategy,
    StandardStrategy,
    StrategyComponent,
    SurpriseMinKStrategy,
)
from utils.training import validate
from utils.types import StepResult


def _get_model(model_config: Dict[str, Any], device: torch.device) -> nn.Module:
    model_type = model_config.pop('model_type')
    model_map = {
        'dense': VisionTransformer,
        'moe': MoEVisionTransformer,
        'gaussian_moe': GaussianMoEVisionTransformer,
    }
    model_cls = model_map.get(model_type)
    if not model_cls:
        raise ValueError(f"Unknown model type: {model_type}")
    return model_cls(**model_config).to(device)

def _get_optimizer(model: nn.Module, learning_rate: float, weight_decay: float) -> optim.Optimizer:
    param_groups = model.get_param_groups() if hasattr(model, 'get_param_groups') else [
        {'params': model.parameters()}
    ]
    for pg in param_groups:
        pg['lr'] = learning_rate
        pg['initial_lr'] = learning_rate
    return optim.AdamW(param_groups, lr=learning_rate, weight_decay=weight_decay)

def _get_loss_fn() -> nn.Module:
    return nn.CrossEntropyLoss()

def _get_strategy_components(
    strategy_configs: List[Dict[str, Any]],
    device: torch.device,
    pilr_config: Dict[str, Any]
) -> List[StrategyComponent]:
    strategy_map = {
        'Standard': StandardStrategy,
        'Selective': SelectiveUpdateStrategy,
        'SurpriseMinK': SurpriseMinKStrategy,
        'PILR_Single': PILRStrategy,
        'PILR_Dual': PILRStrategy,
    }
    
    components: List[StrategyComponent] = []
    for config in strategy_configs:
        strategy_name = config['name']
        strategy_cls = strategy_map.get(strategy_name)
        if not strategy_cls:
            raise ValueError(f"Unknown strategy component: {strategy_name}")
        
        component_kwargs = {**pilr_config, **config}
        
        if strategy_cls == PILRStrategy:
            components.append(strategy_cls(device=device, **component_kwargs))
        else:
            components.append(strategy_cls(**component_kwargs))
            
    return components

tensorboard_process: Optional[subprocess.Popen[bytes]] = None

def _cleanup_tensorboard() -> None:
    global tensorboard_process
    if tensorboard_process:
        print("Shutting down TensorBoard...")
        tensorboard_process.kill()
        tensorboard_process = None

def _initialize_experiment(
    model_config_module: Any,
    schedule_config: Dict[str, Any],
    output_dir: str,
    device: torch.device
) -> Tuple[nn.Module, optim.Optimizer, nn.Module, SigmaPI, List[StrategyComponent], SummaryWriter, Dict[str, Any]]:
    
    global tensorboard_process
    
    model_config = getattr(model_config_module, 'model_config')
    train_strategy_config = getattr(model_config_module, 'train_strategy_config', {'strategies': [{'name': 'Standard'}]})
    pilr_config = getattr(model_config_module, 'pilr_config', {})

    model = _get_model(model_config, device)
    optimizer = _get_optimizer(model, schedule_config['train_config']['learning_rate'], schedule_config['train_config']['weight_decay'])
    loss_fn = _get_loss_fn()
    pi_monitor = SigmaPI(device=device, **schedule_config['pi_config'])

    strategy_components = _get_strategy_components(train_strategy_config['strategies'], device, pilr_config)

    runs_dir = os.path.join(output_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    log_dir = os.path.join(runs_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)
    
    tb_command = ["tensorboard", "--logdir", log_dir, "--port", "6006"]
    tensorboard_process = subprocess.Popen(tb_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    atexit.register(_cleanup_tensorboard)
    
    print(f"TensorBoard launched for this experiment. View at: http://127.0.0.1:6006")

    return model, optimizer, loss_fn, pi_monitor, strategy_components, writer, model_config

def run_schedule(
    model_config_path: str,
    schedule_path: str,
    resume_from: Optional[str] = None
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    schedule_module_name = schedule_path.replace('.py', '').replace(os.sep, '.')
    schedule_config = importlib.import_module(schedule_module_name).schedule_config

    model_module_name = model_config_path.replace('.py', '').replace(os.sep, '.')
    model_config_module = importlib.import_module(model_module_name)
    
    model_name = os.path.basename(model_config_path).replace('.py', '')
    schedule_name = os.path.basename(schedule_path).replace('.py', '')
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    output_dir = os.path.join('output', schedule_name, model_name, timestamp)
    
    for subdir in ["checkpoints", "runs", "img"]:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    model, optimizer, loss_fn, pi_monitor, strategy_components, writer, model_config = \
        _initialize_experiment(model_config_module, schedule_config, output_dir, device)

    print(f"Experiment output will be saved to: {output_dir}")

    print("\n--- Pre-checking datasets ---")
    all_datasets_in_schedule = {task_name for task_name, _ in schedule_config['tasks'] if task_name != 'VALIDATE'}
    all_datasets_in_schedule.update(schedule_config['val_datasets'])

    for dataset_name_to_check in all_datasets_in_schedule:
        try:
            print(f"Checking dataset: {dataset_name_to_check}...")
            get_dataset(dataset_name_to_check, model_config['img_size'], model_config['patch_size'], data_root='./temp_data')
            print(f"Dataset '{dataset_name_to_check}' is available.")
        except Exception as e:
            print(f"Error: Dataset '{dataset_name_to_check}' is not available or failed to load: {e}")
            print("Please ensure all required datasets are correctly configured and accessible.")
            return
    print("--- All datasets pre-checked successfully ---")

    global_step = 0
    
    if resume_from:
        if os.path.isfile(resume_from):
            print(f"Resuming from checkpoint: {resume_from}")
            model.load_state_dict(torch.load(resume_from, map_location=device))
        else:
            print(f"Warning: Checkpoint file not found at '{resume_from}'. Starting from scratch.")
    
    all_train_results: List[StepResult] = []
    val_logs: Dict[str, List[Tuple[int, float]]] = {ds: [] for ds in schedule_config['val_datasets']}

    for cycle in range(schedule_config['num_cycles']):
        print(f"\n--- Cycle {cycle + 1}/{schedule_config['num_cycles']} ---")
        for task_name, num_epochs in schedule_config['tasks']:
            if task_name == 'VALIDATE':
                print("Performing validation across all datasets...")
                for val_ds_name in schedule_config['val_datasets']:
                    _, test_dataset = get_dataset(val_ds_name, model_config['img_size'], model_config['patch_size'])
                    val_loader = DataLoader(test_dataset, batch_size=schedule_config['train_config']['batch_size'], shuffle=False)
                    avg_loss, accuracy, avg_pi, avg_surprise, avg_tau, avg_gating_tau = validate(model, device, val_loader, loss_fn, pi_monitor, val_ds_name)
                    
                    writer.add_scalar(f'Validation/Loss/{val_ds_name}', avg_loss, global_step)
                    writer.add_scalar(f'Validation/Accuracy/{val_ds_name}', accuracy, global_step)
                    if avg_pi is not None: writer.add_scalar(f'Validation/PI_Score/{val_ds_name}', avg_pi, global_step)
                    if avg_surprise is not None: writer.add_scalar(f'Validation/Surprise/{val_ds_name}', avg_surprise, global_step)
                    if avg_tau is not None: writer.add_scalar(f'Validation/Tau/{val_ds_name}', avg_tau, global_step)
                    if avg_gating_tau is not None: writer.add_scalar(f'Validation/Gating_Tau/{val_ds_name}', avg_gating_tau, global_step)
                    val_logs[val_ds_name].append((global_step, accuracy))
                continue

            print(f"Training on {task_name} for {num_epochs} epochs...")
            train_dataset, _ = get_dataset(task_name, model_config['img_size'], model_config['patch_size'])
            train_loader = DataLoader(train_dataset, batch_size=schedule_config['train_config']['batch_size'], shuffle=True)

            for epoch in range(1, num_epochs + 1):
                model.train()
                optimizer.zero_grad()

                epoch_results: List[StepResult] = []

                train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch} ({task_name})", leave=False)
                for batch_idx, (data, target) in enumerate(train_loader_tqdm):
                    data, target = data.to(device), target.to(device)

                    output = model(data)
                    
                    all_top_indices, all_log_probs, all_gating_logits = None, None, None
                    if isinstance(output, tuple) and len(output) == 3:
                        logits, all_top_indices, all_log_probs = output
                        all_gating_logits = all_log_probs
                    elif isinstance(output, tuple):
                        logits, all_gating_logits = output
                    else:
                        logits = output

                    loss = loss_fn(logits, target)
                    loss_normalized = loss / schedule_config['train_config']['accumulation_steps']
                    loss_normalized.backward()

                    pred = logits.argmax(dim=1, keepdim=True)
                    correct = pred.eq(target.view_as(pred)).sum().item()
                    accuracy = 100. * correct / len(data)

                    if (batch_idx + 1) % schedule_config['train_config']['accumulation_steps'] == 0:
                        pi_metrics = pi_monitor.calculate(model, loss, logits)
                        
                        step_result: StepResult = {
                            "global_step": global_step,
                            "loss": loss.item(),
                            "accuracy": accuracy,
                            "task_name": task_name,
                            **pi_metrics,
                        }

                        for component in strategy_components:
                            component_metrics = component.apply(model, optimizer, pi_metrics, all_gating_logits, all_top_indices)
                            step_result.update(component_metrics)
                        
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()

                        epoch_results.append(step_result)
                        global_step += 1

                        for key, value in step_result.items():
                            if key in ['cognitive_cost', 'epsilon', 'active_expert_indices', 'updated_expert_indices', 'all_log_probs', 'all_top_indices']:
                                continue
                            if isinstance(value, (int, float)):
                                tag = f'Predictive Integrity/train/{task_name}' if key == 'pi_score' else f'{key.replace("_", " ").title()}/train/{task_name}'
                                writer.add_scalar(tag, value, global_step)
                        writer.flush()

                        train_loader_tqdm.set_postfix(
                            loss=f"{step_result.get('loss', 0.0):.4f}",
                            acc=f"{step_result.get('accuracy', 0.0):.2f}%",
                            pi=f"{step_result.get('pi_score', 0.0):.4f}",
                            surprise=f"{step_result.get('surprise', 0.0):.4f}",
                        )
                
                if epoch_results:
                    def safe_avg(key: str) -> Optional[float]:
                        values = [r.get(key) for r in epoch_results if r.get(key) is not None]
                        numeric_values = [v for v in values if isinstance(v, (int, float))]
                        return sum(numeric_values) / len(numeric_values) if numeric_values else None

                    summary_parts = [
                        f"Train Epoch {epoch} Summary:",
                        f"Avg loss: {safe_avg('loss'):.4f}",
                        f"Avg Accuracy: {safe_avg('accuracy'):.2f}%",
                        f"Avg PI: {safe_avg('pi_score'):.4f}",
                        f"Avg Surprise: {safe_avg('surprise'):.4f}",
                    ]
                    print(", ".join(filter(None, summary_parts)))
                
                all_train_results.extend(epoch_results)

                if epoch_results and any('active_expert_indices' in r for r in epoch_results):
                    num_layers = model_config.get('depth', 1)
                    num_experts = model_config.get('num_experts', 1)

                    fig_scatter_active_epoch = plot_expert_scatter(epoch_results, num_layers, num_experts, 'active')
                    writer.add_figure(f'Expert Dynamics/Epoch {epoch}/{task_name}/Active Experts Scatter', fig_scatter_active_epoch, global_step)
                    plt.close(fig_scatter_active_epoch)

                    fig_heatmap_active_epoch = plot_expert_heatmap(epoch_results, num_layers, num_experts, 'active')
                    writer.add_figure(f'Expert Dynamics/Epoch {epoch}/{task_name}/Active Experts Heatmap', fig_heatmap_active_epoch, global_step)
                    plt.close(fig_heatmap_active_epoch)

                    if any('updated_expert_indices' in r for r in epoch_results):
                        fig_scatter_updated_epoch = plot_expert_scatter(epoch_results, num_layers, num_experts, 'updated')
                        writer.add_figure(f'Expert Dynamics/Epoch {epoch}/{task_name}/Updated Experts Scatter', fig_scatter_updated_epoch, global_step)
                        plt.close(fig_scatter_updated_epoch)

                        fig_heatmap_updated_epoch = plot_expert_heatmap(epoch_results, num_layers, num_experts, 'updated')
                        writer.add_figure(f'Expert Dynamics/Epoch {epoch}/{task_name}/Updated Experts Heatmap', fig_heatmap_updated_epoch, global_step)
                        plt.close(fig_heatmap_updated_epoch)

            checkpoint_path = os.path.join(output_dir, "checkpoints", f"epoch_{global_step}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    print("\n--- Final Validation ---")
    for val_ds_name in schedule_config['val_datasets']:
        _, test_dataset = get_dataset(val_ds_name, model_config['img_size'], model_config['patch_size'])
        val_loader = DataLoader(test_dataset, batch_size=schedule_config['train_config']['batch_size'], shuffle=False)
        avg_loss, accuracy, avg_pi, avg_surprise, avg_tau, avg_gating_tau = validate(model, device, val_loader, loss_fn, pi_monitor, val_ds_name)
        
        writer.add_scalar(f'Validation/Loss/{val_ds_name}', avg_loss, global_step)
        writer.add_scalar(f'Validation/Accuracy/{val_ds_name}', accuracy, global_step)
        if avg_pi is not None: writer.add_scalar(f'Validation/PI_Score/{val_ds_name}', avg_pi, global_step)
        if avg_surprise is not None: writer.add_scalar(f'Validation/Surprise/{val_ds_name}', avg_surprise, global_step)
        if avg_tau is not None: writer.add_scalar(f'Validation/Tau/{val_ds_name}', avg_tau, global_step)
        if avg_gating_tau is not None: writer.add_scalar(f'Validation/Gating_Tau/{val_ds_name}', avg_gating_tau, global_step)
        val_logs[val_ds_name].append((global_step, accuracy))

    final_model_path = os.path.join(output_dir, "checkpoints", f"epoch_{global_step}_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    if all_train_results and any('active_expert_indices' in r for r in all_train_results):
        num_layers = model_config.get('depth', 1)
        num_experts = model_config.get('num_experts', 1)
        img_dir = os.path.join(output_dir, "img")

        fig_scatter_active = plot_expert_scatter(all_train_results, num_layers, num_experts, 'active')
        fig_scatter_active.savefig(os.path.join(img_dir, "active_experts_scatter.png"))
        writer.add_figure('Expert Dynamics/Active Experts Scatter', fig_scatter_active, global_step)
        plt.close(fig_scatter_active)

        fig_heatmap_active = plot_expert_heatmap(all_train_results, num_layers, num_experts, 'active')
        fig_heatmap_active.savefig(os.path.join(img_dir, "active_experts_heatmap.png"))
        writer.add_figure('Expert Dynamics/Active Experts Heatmap', fig_heatmap_active, global_step)
        plt.close(fig_heatmap_active)

        if any('updated_expert_indices' in r for r in all_train_results):
            fig_scatter_updated = plot_expert_scatter(all_train_results, num_layers, num_experts, 'updated')
            fig_scatter_updated.savefig(os.path.join(img_dir, "updated_experts_scatter.png"))
            writer.add_figure('Expert Dynamics/Updated Experts Scatter', fig_scatter_updated, global_step)
            plt.close(fig_scatter_updated)

            fig_heatmap_updated = plot_expert_heatmap(all_train_results, num_layers, num_experts, 'updated')
            fig_heatmap_updated.savefig(os.path.join(img_dir, "updated_experts_heatmap.png"))
            writer.add_figure('Expert Dynamics/Updated Experts Heatmap', fig_heatmap_updated, global_step)
            plt.close(fig_heatmap_updated)

    writer.close()
    print("Experiment finished.")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run PILF training schedule.")
    parser.add_argument('--schedule', type=str, required=True,
                        help='Path to the schedule configuration file (e.g., schedules/marathon_v3.py)')
    parser.add_argument('--model-config', type=str, required=True,
                        help='Path to the model configuration file (e.g., configs/large_gpil_smk.py)')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Path to a checkpoint file to resume training from.')

    args = parser.parse_args()

    run_schedule(args.model_config, args.schedule, args.resume_from)
