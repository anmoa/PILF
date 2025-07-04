import argparse
import atexit
import os
import random
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models import create_model
from utils.config import load_config
from utils.datasets import get_dataset
from utils.logging.plotting import plot_core_metrics, plot_expert_dashboard
from utils.pi_calculator import PICalculator
from utils.strategies.learning_rate_strategies import create_lr_strategy
from utils.train_loops.base_train_loop import BaseTrainLoop
from utils.train_loops.gaussian_train_loop import GaussianTrainLoop
from utils.train_loops.mgm_train_loop import MGMTrainLoop
from utils.trainer import Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


tensorboard_process: Optional[subprocess.Popen[bytes]] = None


def _cleanup_tensorboard():
    global tensorboard_process
    if tensorboard_process:
        try:
            print("Shutting down TensorBoard...")
            tensorboard_process.terminate()
            tensorboard_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tensorboard_process.kill()
            print("TensorBoard process did not terminate gracefully, killed.")
        finally:
            tensorboard_process = None


class ScheduleRunner:
    def __init__(self, config, trainer, train_loop, train_loaders, val_loaders, run_name):
        self.config = config
        self.trainer = trainer
        self.train_loop = train_loop
        self.train_loaders = train_loaders
        self.val_loaders = val_loaders
        self.run_name = run_name
        self.schedule_plan = self._create_schedule_plan()
        self.task_epochs_map = {
            task[0]: task[1] for task in self.config.schedule["tasks"]
        }

    def _create_schedule_plan(self) -> List[Tuple[str, int]]:
        schedule_config = cast(Dict[str, Any], self.config.schedule)
        tasks = schedule_config["tasks"]
        num_cycles = schedule_config.get("num_cycles", 1)

        plan = []
        for _ in range(num_cycles):
            for task_name, num_epochs in tasks:
                if task_name.upper() == "VALIDATE":
                    plan.append(("VALIDATE", 1))
                else:
                    for i in range(num_epochs):
                        plan.append((task_name, i + 1))
        return plan

    def run(self):
        global_step = self.trainer.global_step
        current_epoch_in_task = self.trainer.current_epoch_in_task

        for task_name, epoch_num_in_task in self.schedule_plan:
            if task_name.upper() == "VALIDATE":
                epoch_for_validation = (
                    max(current_epoch_in_task.values()) if current_epoch_in_task else 1
                )
                for val_task, val_loader in self.val_loaders.items():
                    self.trainer.validate(
                        val_loader, global_step, epoch_for_validation, val_task
                    )
                if self.trainer.is_moe_model:
                    self.trainer.log_expert_embeddings(global_step)
                self.trainer.save_checkpoint(self.run_name, epoch_for_validation, global_step)
                continue

            current_epoch_in_task[task_name] = (
                current_epoch_in_task.get(task_name, 0) + 1
            )
            epoch = current_epoch_in_task[task_name]

            schedule_config = cast(Dict[str, Any], self.config.schedule)
            total_epochs_for_task = self.task_epochs_map.get(task_name, "N/A")
            print(
                f"--- Starting Cycle Epoch {epoch_num_in_task}/{total_epochs_for_task}, Task: {task_name} (Overall Epoch {epoch}) ---"
            )

            global_step, epoch_results = self.train_loop.train_one_epoch(
                self.train_loaders[task_name],
                epoch,
                global_step,
                schedule_config["train_config"].get("accumulation_steps", 1),
                task_name,
            )
            self.trainer.epoch_results.extend(epoch_results)
            self.trainer.global_step = global_step
            self.trainer.current_epoch_in_task = current_epoch_in_task

    def save_final_plots(self, run_name: str):
        schedule_config = cast(Dict[str, Any], self.config.schedule)
        output_dir = os.path.join(
            str(schedule_config["train_config"]["output_dir"]), run_name, "img"
        )
        os.makedirs(output_dir, exist_ok=True)

        if not self.trainer.epoch_results:
            print("No training steps recorded, skipping plot generation.")
            return

        core_metrics_fig = plot_core_metrics(
            self.trainer.epoch_results,
            self.trainer.validation_history,
            run_name,
            self.trainer.num_layers,
            self.trainer.num_experts,
        )
        core_metrics_path = os.path.join(output_dir, f"{run_name}_Core_Metrics.png")
        core_metrics_fig.savefig(core_metrics_path)
        plt.close(core_metrics_fig)
        print(f"Saved core metrics plot: {core_metrics_path}")

        if self.trainer.is_moe_model:
            expert_dashboard_fig = plot_expert_dashboard(
                self.trainer.epoch_results,
                self.trainer.num_layers,
                self.trainer.num_experts,
            )
            expert_dashboard_path = os.path.join(
                output_dir, f"{run_name}_Expert_Dashboard.png"
            )
            expert_dashboard_fig.savefig(expert_dashboard_path)
            plt.close(expert_dashboard_fig)
            print(f"Saved expert dashboard plot: {expert_dashboard_path}")


def main():
    parser = argparse.ArgumentParser(description="PILF Framework Training")
    parser.add_argument("--schedule", required=True, help="Path to the schedule file")
    parser.add_argument(
        "--config",
        default="configs/unified_config.py",
        help="Path to the unified config file",
    )
    parser.add_argument(
        "--router", required=True, help="Routing strategy for MoE layers"
    )
    parser.add_argument("--update", required=True, help="Backpropagation strategy")
    parser.add_argument("--lrs", required=True, help="Learning rate strategy")
    parser.add_argument("--resume-from", help="Path to a checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        print(f"PyTorch CUDA version: {torch.version.cuda}")
        print(f"Is backend CuDNN enabled: {torch.backends.cudnn.enabled}")
        has_flash_attn = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        print(f"Backend Flash Attention 2 available: {has_flash_attn}")
        if has_flash_attn:
             print(f"Is backend Flash Attention enabled: {torch.backends.cuda.flash_sdp_enabled()}")
             print(f"Is backend Memory Efficient Attention enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}")


    schedule_config = cast(Dict[str, Any], config.schedule)
    train_task_names = [
        task[0] for task in schedule_config["tasks"] if task[0].upper() != "VALIDATE"
    ]
    val_task_names = schedule_config.get("val_datasets", [])
    all_dataset_names = sorted(list(set(train_task_names + val_task_names)))

    print("Loading datasets...")
    datasets = {
        name: get_dataset(
            name,
            schedule_config["train_config"]["batch_size"],
            config.model["img_size"],
        )
        for name in all_dataset_names
    }
    print("Datasets loaded.")

    def create_dataloader(task_names, split, shuffle):
        return {
            name: torch.utils.data.DataLoader(
                datasets[name][split],
                batch_size=schedule_config["train_config"]["batch_size"],
                shuffle=shuffle,
                num_workers=4,
                pin_memory=True,
            )
            for name in task_names
        }

    train_loaders = create_dataloader(train_task_names, "train", True)
    val_loaders = create_dataloader(val_task_names, "test", False)

    model = create_model(config.model).to(device)
    optimizer = optim.Adam(
        model.get_param_groups(), lr=schedule_config["train_config"]["learning_rate"]
    )
    loss_fn = nn.CrossEntropyLoss()

    pi_config = cast(Dict[str, Any], config.pi)

    strategy_components = []
    if config.pilr:
        strategy_components.append(create_lr_strategy(config.pilr))

    run_name = f"{schedule_config['name']}-{config.model.get('name', 'model')}-{args.router}-{args.update}-{args.lrs}-{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(f"runs/{run_name}")

    global tensorboard_process
    log_dir = f"runs/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    tb_command = ["tensorboard", "--logdir", log_dir, "--port", "6006"]
    try:
        tensorboard_process = subprocess.Popen(
            tb_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    except FileNotFoundError:
        print(
            "TensorBoard not found. Please make sure it is installed and in your PATH."
        )
        tensorboard_process = None
    atexit.register(_cleanup_tensorboard)
    print("TensorBoard launched. View at: http://127.0.0.1:6006")

    pi_calculator = PICalculator(
        alpha=pi_config.get("alpha", 1.0),
        gamma=pi_config.get("gamma", 0.5),
        device=device,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        strategy_components=strategy_components,
        device=device,
        writer=writer,
        num_layers=config.model.get("depth", 0),
        num_experts=config.model.get("num_experts", 0),
        pi_calculator=pi_calculator,
    )

    router_type = config.model.get("router_type", "").lower()
    if "memory_gaussian" in router_type:
        train_loop = MGMTrainLoop(trainer)
    elif "gaussian" in router_type:
        train_loop = GaussianTrainLoop(trainer)
    else:
        train_loop = BaseTrainLoop(trainer)

    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)

    runner = ScheduleRunner(config, trainer, train_loop, train_loaders, val_loaders, run_name)
    runner.run()

    final_epoch = max(runner.trainer.current_epoch_in_task.values()) if runner.trainer.current_epoch_in_task else 0
    trainer.save_checkpoint(run_name, final_epoch, runner.trainer.global_step, is_final=True)
    runner.save_final_plots(run_name)

    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
