import atexit
import os
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models
from utils.config import load_config
from utils.datasets import get_dataset
from utils.optimizers import create_optimizer
from utils.pi_calculator import PiCalculator
from utils.plotting import plot_expert_heatmap, plot_expert_scatter
from utils.strategies.backpropagation_strategies import (
    SelectiveUpdateStrategy,
    StandardStrategy,
    SurpriseMinKStrategy,
)
from utils.strategies.base_strategy import StrategyComponent
from utils.strategies.learning_rate_strategies import PILRStrategy
from utils.training import Trainer
from utils.types import StepResult

tensorboard_process: Optional[subprocess.Popen[bytes]] = None

def _cleanup_tensorboard() -> None:
    global tensorboard_process
    if tensorboard_process:
        print("Shutting down TensorBoard...")
        tensorboard_process.kill()
        tensorboard_process = None

def run_schedule(
    model_config_path: str,
    schedule_path: str,
    resume_from: Optional[str] = None
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Configuration ---
    config = load_config(model_config_path, schedule_path)
    
    # --- 2. Experiment Setup ---
    model_name = os.path.basename(model_config_path).replace('.py', '')
    schedule_name = os.path.basename(schedule_path).replace('.py', '')
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join('output', schedule_name, model_name, timestamp)
    
    for subdir in ["checkpoints", "runs", "img"]:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    # --- 3. Object Construction ---
    model = models.create_model(config.model).to(device)
    optimizer = create_optimizer(model, config.schedule['train_config'])
    loss_fn = nn.CrossEntropyLoss()
    pi_monitor = PiCalculator(device=str(device), **config.schedule['pi_config'])
    strategy_components: List[StrategyComponent] = []
    for strategy_config in config.train_strategy['strategies']:
        name = strategy_config['name']
        if name == 'Standard':
            strategy_components.append(StandardStrategy())
        elif name == 'Selective':
            strategy_components.append(SelectiveUpdateStrategy())
        elif name == 'SMK':
            strategy_components.append(SurpriseMinKStrategy(**strategy_config.get('params', {})))
        elif name == 'PILR':
            strategy_components.append(PILRStrategy(device, **config.pilr))

    # --- 4. Tensorboard and Logging ---
    runs_dir = os.path.join(output_dir, "runs")
    log_dir = os.path.join(runs_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)
    
    global tensorboard_process
    tb_command = ["tensorboard", "--logdir", log_dir, "--port", "6006"]
    tensorboard_process = subprocess.Popen(tb_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    atexit.register(_cleanup_tensorboard)
    print("TensorBoard launched. View at: http://127.0.0.1:6006")
    print(f"Experiment output will be saved to: {output_dir}")

    # --- 5. Trainer Initialization ---
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        pi_monitor=pi_monitor,
        strategy_components=strategy_components,
        device=device,
        writer=writer,
        gating_loss_config=config.gating_loss
    )

    # --- 6. Dataset Pre-check ---
    print("\n--- Pre-checking datasets ---")
    all_datasets_in_schedule = {task[0] for task in config.schedule['tasks'] if task[0] != 'VALIDATE'}
    all_datasets_in_schedule.update(config.schedule['val_datasets'])
    for dataset_name in all_datasets_in_schedule:
        try:
            get_dataset(dataset_name, config.model['img_size'], config.model['patch_size'])
            print(f"Dataset '{dataset_name}' is available.")
        except Exception as e:
            print(f"Error checking dataset '{dataset_name}': {e}")
            return
    print("--- All datasets pre-checked successfully ---")

    # --- 7. Training Orchestration ---
    global_step = 0
    if resume_from:
        if os.path.isfile(resume_from):
            print(f"Resuming from checkpoint: {resume_from}")
            model.load_state_dict(torch.load(resume_from, map_location=device))
        else:
            print(f"Warning: Checkpoint file not found at '{resume_from}'. Starting from scratch.")
    
    all_train_results: List[StepResult] = []
    val_logs: Dict[str, List[Tuple[int, float]]] = {ds: [] for ds in config.schedule['val_datasets']}
    last_checkpoint_path: Optional[str] = None

    for cycle in range(config.schedule['num_cycles']):
        print(f"\n--- Cycle {cycle + 1}/{config.schedule['num_cycles']} ---")
        for task_name, num_epochs in config.schedule['tasks']:
            if task_name == 'VALIDATE':
                print("Performing validation across all datasets...")
                for val_ds_name in config.schedule['val_datasets']:
                    _, test_dataset = get_dataset(val_ds_name, config.model['img_size'], config.model['patch_size'])
                    val_loader = DataLoader(test_dataset, batch_size=config.schedule['train_config']['batch_size'], shuffle=False)
                    avg_loss, accuracy, avg_pi, avg_surprise, avg_tau, avg_gating_tau, avg_gating_loss = trainer.validate(val_loader, global_step, val_ds_name)
                    
                    writer.add_scalar(f'Validation/Loss/{val_ds_name}', avg_loss, global_step)
                    writer.add_scalar(f'Validation/Accuracy/{val_ds_name}', accuracy, global_step)
                    if avg_pi is not None:
                        writer.add_scalar(f'Validation/PI_Score/{val_ds_name}', avg_pi, global_step)
                    if avg_surprise is not None:
                        writer.add_scalar(f'Validation/Surprise/{val_ds_name}', avg_surprise, global_step)
                    if avg_tau is not None:
                        writer.add_scalar(f'Validation/Tau/{val_ds_name}', avg_tau, global_step)
                    if avg_gating_tau is not None:
                        writer.add_scalar(f'Validation/Gating_Tau/{val_ds_name}', avg_gating_tau, global_step)
                    if avg_gating_loss is not None:
                        writer.add_scalar(f'Validation/Gating_Loss/{val_ds_name}', avg_gating_loss, global_step)
                    val_logs[val_ds_name].append((global_step, accuracy))
                continue

            print(f"Training on {task_name} for {num_epochs} epochs...")
            train_dataset, _ = get_dataset(task_name, config.model['img_size'], config.model['patch_size'])
            train_loader = DataLoader(train_dataset, batch_size=config.schedule['train_config']['batch_size'], shuffle=True)

            for epoch in range(1, num_epochs + 1):
                global_step, epoch_results = trainer.train_one_epoch(
                    train_loader=train_loader,
                    epoch=epoch,
                    global_step=global_step,
                    accumulation_steps=config.schedule['train_config']['accumulation_steps'],
                    task_name=task_name,
                )
                
                if epoch_results: # Accumulate results from the current epoch
                    all_train_results.extend(epoch_results)
            
            # Save checkpoint after each task
            current_checkpoint_path = os.path.join(output_dir, "checkpoints", f"epoch_{global_step}.pth")
            torch.save(model.state_dict(), current_checkpoint_path)
            print(f"Checkpoint saved to {current_checkpoint_path}")

            # Plotting for Active Experts after each task, if data is available
            if all_train_results and any('active_expert_indices' in r for r in all_train_results):
                num_layers = config.model.get('depth', 1)
                num_experts = config.model.get('num_experts', 1)
                
                fig_heatmap_active_task = plot_expert_heatmap(all_train_results, num_layers, num_experts, 'active')
                writer.add_figure(f'Expert Dynamics/Active Experts Heatmap (Task: {task_name})', fig_heatmap_active_task, global_step)
                plt.close(fig_heatmap_active_task)

                if any('updated_expert_indices' in r for r in all_train_results):
                    fig_heatmap_updated_task = plot_expert_heatmap(all_train_results, num_layers, num_experts, 'updated')
                    writer.add_figure(f'Expert Dynamics/Updated Experts Heatmap (Task: {task_name})', fig_heatmap_updated_task, global_step)
                    plt.close(fig_heatmap_updated_task)

            # Delete previous checkpoint to save disk space
            if last_checkpoint_path is not None and os.path.exists(last_checkpoint_path):
                os.remove(last_checkpoint_path)
                print(f"Deleted previous checkpoint: {last_checkpoint_path}")
            last_checkpoint_path = current_checkpoint_path

    # --- 8. Finalization ---
    print("\n--- Final Validation ---")
    for val_ds_name in config.schedule['val_datasets']:
        _, test_dataset = get_dataset(val_ds_name, config.model['img_size'], config.model['patch_size'])
        val_loader = DataLoader(test_dataset, batch_size=config.schedule['train_config']['batch_size'], shuffle=False)
        avg_loss, accuracy, avg_pi, avg_surprise, avg_tau, avg_gating_tau, avg_gating_loss = trainer.validate(val_loader, global_step, val_ds_name)
        writer.add_scalar(f'Validation/Loss/{val_ds_name}', avg_loss, global_step)
        writer.add_scalar(f'Validation/Accuracy/{val_ds_name}', accuracy, global_step)
        if avg_pi is not None:
            writer.add_scalar(f'Validation/PI_Score/{val_ds_name}', avg_pi, global_step)
        if avg_surprise is not None:
            writer.add_scalar(f'Validation/Surprise/{val_ds_name}', avg_surprise, global_step)
        if avg_tau is not None:
            writer.add_scalar(f'Validation/Tau/{val_ds_name}', avg_tau, global_step)
        if avg_gating_tau is not None:
            writer.add_scalar(f'Validation/Gating_Tau/{val_ds_name}', avg_gating_tau, global_step)
        if avg_gating_loss is not None:
            writer.add_scalar(f'Validation/Gating_Loss/{val_ds_name}', avg_gating_loss, global_step)

    final_model_path = os.path.join(output_dir, "checkpoints", f"epoch_{global_step}_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Clean up the very last intermediate checkpoint if it's not the final one
    if last_checkpoint_path is not None and os.path.exists(last_checkpoint_path) and last_checkpoint_path != final_model_path:
        os.remove(last_checkpoint_path)
        print(f"Deleted last intermediate checkpoint: {last_checkpoint_path}")

    if all_train_results and any('active_expert_indices' in r for r in all_train_results):
        num_layers = config.model.get('depth', 1)
        num_experts = config.model.get('num_experts', 1)
        img_dir = os.path.join(output_dir, "img")

        # Plotting for Active Experts
        fig_scatter_active = plot_expert_scatter(all_train_results, num_layers, num_experts, 'active')
        fig_scatter_active.savefig(os.path.join(img_dir, "active_experts_scatter.png"))
        writer.add_figure('Expert Dynamics/Active Experts Scatter', fig_scatter_active, global_step)
        plt.close(fig_scatter_active)

        fig_heatmap_active = plot_expert_heatmap(all_train_results, num_layers, num_experts, 'active')
        fig_heatmap_active.savefig(os.path.join(img_dir, "active_experts_heatmap.png"))
        writer.add_figure('Expert Dynamics/Active Experts Heatmap', fig_heatmap_active, global_step)
        plt.close(fig_heatmap_active)

        # Plotting for Updated (Min-K) Experts
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
    parser.add_argument('--schedule', type=str, required=True, help='Path to the schedule configuration file')
    parser.add_argument('--model-config', type=str, required=True, help='Path to the model configuration file')
    parser.add_argument('--resume-from', type=str, default=None, help='Path to a checkpoint file to resume training from.')

    args = parser.parse_args()
    run_schedule(args.model_config, args.schedule, args.resume_from)
