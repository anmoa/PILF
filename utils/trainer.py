import os
from typing import Any, Dict, List, Sized, cast

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.logging.metrics_logger import TensorBoardLogger
from utils.logging.plotting import plot_expert_dashboard
from utils.logging.types import StepResult
from utils.pi_calculator import PICalculator
from utils.strategies.base_strategy import StrategyComponent


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        strategy_components: List[StrategyComponent],
        device: torch.device,
        writer: SummaryWriter,
        num_layers: int,
        num_experts: int,
        pi_calculator: PICalculator,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.strategy_components = strategy_components
        self.device = device
        self.writer = writer
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.pi_calculator = pi_calculator
        self.epoch_results: List[StepResult] = []
        self.validation_history: Dict[str, List[Dict[str, Any]]] = {}
        self.global_step: int = 0
        self.current_epoch_in_task: Dict[str, int] = {}

        self.is_moe_model = hasattr(self.model, "get_param_groups")
        self.gating_params: List[torch.nn.Parameter] = []
        self.expert_params: List[torch.nn.Parameter] = []
        if self.is_moe_model:
            for pg in self.model.get_param_groups():
                if pg["name"] == "gating":
                    self.gating_params.extend(pg["params"])
                elif pg["name"] == "experts":
                    self.expert_params.extend(pg["params"])

    def validate(
        self,
        val_loader: DataLoader,
        global_step: int,
        epoch: int,
        dataset_name: str = "Validation",
    ) -> Dict[str, Any]:
        self.model.eval()
        total_loss, correct = 0.0, 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                logits = output[0] if isinstance(output, tuple) else output

                loss = self.loss_fn(logits, target)
                total_loss += loss.item()

                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        avg_loss = total_loss / len(val_loader)
        dataset = val_loader.dataset
        num_samples = len(cast(Sized, dataset))
        accuracy = 100.0 * correct / num_samples

        print(
            f"{dataset_name} set: Avg loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )

        val_result = {
            "global_step": global_step,
            "epoch": epoch,
            "loss": avg_loss,
            "accuracy": accuracy,
        }

        if dataset_name not in self.validation_history:
            self.validation_history[dataset_name] = []
        self.validation_history[dataset_name].append(val_result)

        logger = TensorBoardLogger(self.writer, global_step)
        logger.log_metrics(
            {k: v for k, v in val_result.items() if isinstance(v, (int, float))},
            dataset_name,
            scope="Validation",
        )

        if self.is_moe_model and self.epoch_results:
            self.plot_epoch_dashboards(epoch, global_step)

        return val_result

    def plot_epoch_dashboards(self, epoch: int, global_step: int):
        if not self.epoch_results or not self.is_moe_model:
            return

        fig = plot_expert_dashboard(
            train_steps=self.epoch_results,
            num_layers=self.num_layers,
            num_experts=self.num_experts,
        )
        self.writer.add_figure(
            "Expert Activations/Dashboard", fig, global_step=global_step
        )
        plt.close(fig)

    def log_expert_embeddings(self, global_step: int):
        if not self.is_moe_model:
            return

        for i, block in enumerate(self.model.blocks):
            if hasattr(block.mlp, "expert_mus"):
                expert_mus = block.mlp.expert_mus.data
                expert_labels = [f"L{i}_E{j}" for j in range(expert_mus.size(0))]
                self.writer.add_embedding(
                    expert_mus,
                    metadata=expert_labels,
                    tag=f"Expert Embeddings Layer {i}",
                    global_step=global_step,
                )

    def save_checkpoint(self, run_name: str, epoch: int, global_step: int, is_final: bool = False):
        schedule_name = run_name.split('-')[0]
        checkpoint_dir = os.path.join("output", schedule_name, run_name, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        suffix = "final" if is_final else f"epoch_{epoch}"
        checkpoint_path = f"{checkpoint_dir}/{suffix}.pth"

        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "validation_history": self.validation_history,
            "epoch_results": self.epoch_results,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint path {checkpoint_path} does not exist. Starting from scratch.")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.current_epoch_in_task = checkpoint.get("current_epoch_in_task", {})
        self.validation_history = checkpoint.get("validation_history", {})
        self.epoch_results = checkpoint.get("epoch_results", [])

        print(f"Checkpoint loaded from {checkpoint_path}. Resuming from epoch {self.current_epoch_in_task}, global step {self.global_step}.")
