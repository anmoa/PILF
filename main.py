import argparse
import os
import random
import subprocess
import time
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from configs.unified_config import BASE_CONFIG
from models import VisionTransformer
from models.moe_layers import BaseMoELayer
from utils.datasets import get_dataset
from utils.experience_buffer import PrototypingExperienceBuffer
from utils.logging.plotting import plot_core_metrics, plot_expert_dashboard
from utils.pi_calculator import PICalculator
from utils.strategies.backpropagation_strategies import (
    ActivatedSurpriseMinKStrategy,
    SurpriseMinKStrategy,
)
from utils.train_loops.base_train_loop import BaseTrainLoop
from utils.train_loops.pilf_train_loop import PILFTrainLoop
from utils.trainer import Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_schedule(schedule_path: str) -> Dict[str, Any]:
    schedule_module_path = f"schedules.{schedule_path.replace('.py', '')}"
    schedule_module = __import__(schedule_module_path, fromlist=["SCHEDULE", "schedule_config"])
    if hasattr(schedule_module, "SCHEDULE"):
        return schedule_module.SCHEDULE
    elif hasattr(schedule_module, "schedule_config"):
        return schedule_module.schedule_config
    else:
        raise AttributeError(f"Schedule file {schedule_path} must contain either a 'SCHEDULE' or 'schedule_config' dictionary.")


def log_and_plot_epoch(trainer: Trainer, model: nn.Module, output_dir: str, run_name: str, cycle: int, task_name: str, epoch: int):
    """Generates and saves plots for the current state of training."""
    print(f"--- Generating plots for epoch {epoch} of task {task_name} in cycle {cycle} ---")
    
    img_dir = os.path.join(output_dir, "img")
    os.makedirs(img_dir, exist_ok=True)

    # Core metrics
    fig_core = plot_core_metrics(trainer.epoch_results, trainer.validation_history, run_name)
    core_plot_filename = f"core_metrics_cycle_{cycle}_task_{task_name}_epoch_{epoch}.png"
    core_plot_path = os.path.join(img_dir, core_plot_filename)
    fig_core.savefig(core_plot_path)
    plt.close(fig_core)

    # Expert dashboard
    moe_blocks = [block for block in model.blocks if isinstance(block.mlp, BaseMoELayer)]
    if moe_blocks:
        num_layers = len(moe_blocks)
        num_experts = moe_blocks[0].mlp.num_experts
        fig_expert = plot_expert_dashboard(trainer.epoch_results, num_layers, num_experts)
        expert_plot_filename = f"expert_dashboard_cycle_{cycle}_task_{task_name}_epoch_{epoch}.png"
        expert_plot_path = os.path.join(img_dir, expert_plot_filename)
        fig_expert.savefig(expert_plot_path)
        plt.close(fig_expert)

def main():
    parser = argparse.ArgumentParser(description="PILF-2 Framework Training")
    parser.add_argument("--schedule", required=True, help="Path to the schedule file")
    parser.add_argument("--router", type=str, default="dense", choices=BASE_CONFIG["router_configs"].keys())
    parser.add_argument("--update", type=str, default="standard", choices=BASE_CONFIG["update_strategy_configs"].keys())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from", help="Path to a checkpoint to resume from")
    parser.add_argument("--no-tensorboard", action="store_true", help="Do not launch TensorBoard")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    schedule_config = load_schedule(args.schedule)
    
    model_cfg = {
        **BASE_CONFIG["model_config"], 
        **schedule_config.get("model_config", {}),
        **BASE_CONFIG["router_configs"][args.router], 
        **BASE_CONFIG["update_strategy_configs"][args.update]
    }
    train_cfg = schedule_config["train_config"]
    
    model = VisionTransformer(**model_cfg).to(device)

    main_params = [p for n, p in model.named_parameters() if "gating_transformer" not in n]
    main_optimizer = optim.Adam(main_params, lr=train_cfg["learning_rate"])

    gating_optimizer = None
    if model_cfg["router_type"] == "memory_gaussian_moe":
        gating_params = [p for n, p in model.named_parameters() if "gating_transformer" in n]
        if gating_params:
            gating_optimizer = optim.Adam(gating_params, lr=train_cfg.get("gating_learning_rate", train_cfg["learning_rate"]))

    pi_calculator = PICalculator(**schedule_config.get("pi_config", {}), device=device)
    
    strategy_components = []
    if args.update == "smk":
        strategy_components.append(SurpriseMinKStrategy(min_k=model_cfg.get("min_k", 1)))
    elif args.update == "asmk":
        strategy_components.append(ActivatedSurpriseMinKStrategy(min_k=model_cfg.get("min_k", 1)))

    run_name = f"{schedule_config['name']}-{args.router}-{args.update}-{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir = f"output/{run_name}"
    log_dir = f"{output_dir}/logs"
    writer = SummaryWriter(log_dir)

    if not args.no_tensorboard:
        subprocess.Popen(["tensorboard", "--logdir", "output"])
        print("Waiting 3 seconds for TensorBoard to start...")
        time.sleep(3)

    trainer = Trainer(
        model=model,
        optimizer=main_optimizer,
        gating_optimizer=gating_optimizer,
        loss_fn=nn.CrossEntropyLoss(),
        strategy_components=strategy_components,
        device=device,
        writer=writer,
        pi_calculator=pi_calculator,
    )

    if model_cfg["router_type"] == "memory_gaussian_moe":
        gating_config = model_cfg["gating_config"]
        pilf_schedule_config = schedule_config.get("pilf_config", {})
        gating_config.update(pilf_schedule_config)

        num_total_experts = sum(
            b.mlp.num_experts for b in model.blocks
            if hasattr(b, "mlp") and isinstance(b.mlp, BaseMoELayer)
        )

        experience_buffer = PrototypingExperienceBuffer(
            capacity=gating_config["total_buffer_size"],
            embed_dim=model_cfg["embed_dim"],
            num_experts=num_total_experts,
            device=device,
        )
        train_loop = PILFTrainLoop(trainer, experience_buffer, gating_config)
    else:
        train_loop = BaseTrainLoop(trainer)

    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)

    all_dataset_names = set(schedule_config.get("val_datasets", [])) | {task[0] for task in schedule_config.get("tasks", [])}
    datasets = {name: get_dataset(name, img_size=model_cfg["img_size"]) for name in all_dataset_names if name != "VALIDATE"}

    for cycle in range(schedule_config.get("num_cycles", 1)):
        for task_name, num_epochs in schedule_config["tasks"]:
            if task_name == "VALIDATE":
                for val_task in schedule_config.get("val_datasets", []):
                    val_loader = DataLoader(datasets[val_task], batch_size=train_cfg["batch_size"], shuffle=False)
                    buffer_to_pass = experience_buffer if model_cfg["router_type"] == "memory_gaussian_moe" else None
                    trainer.validate(val_loader, trainer.global_step, cycle, val_task, experience_buffer=buffer_to_pass)
                continue

            train_loader = DataLoader(datasets[task_name], batch_size=train_cfg["batch_size"], shuffle=True)
            for epoch in range(1, num_epochs + 1):
                print(f"--- Cycle {cycle+1}, Task {task_name}, Epoch {epoch}/{num_epochs} ---")
                train_loop.train_one_epoch(train_loader, epoch, task_name)
                log_and_plot_epoch(trainer, model, output_dir, run_name, cycle + 1, task_name, epoch)
    
    # Final checkpoint saving
    if schedule_config.get("num_cycles", 1) > 1 or len(schedule_config["tasks"]) > 1:
        trainer.save_checkpoint(run_name, epoch=schedule_config["tasks"][-1][1], is_final=True)
    
    writer.close()
    print("Training finished.")

if __name__ == "__main__":
    main()
