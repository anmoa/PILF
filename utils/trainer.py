import os
from typing import Any, Dict, List, Optional, Sized, cast

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.logging.metrics_logger import TensorBoardLogger
from utils.logging.types import StepResult, ValidationResult
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
        pi_calculator: PICalculator,
        gating_optimizer: Optional[optim.Optimizer] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.gating_optimizer = gating_optimizer
        self.loss_fn = loss_fn
        self.strategy_components = strategy_components
        self.device = device
        self.writer = writer
        self.pi_calculator = pi_calculator
        self.epoch_results: List[StepResult] = []
        self.validation_history: Dict[str, List[ValidationResult]] = {}
        self.global_step: int = 0

    def validate(
        self,
        val_loader: DataLoader,
        global_step: int,
        epoch: int,
        dataset_name: str = "Validation",
        experience_buffer: Optional[Any] = None,
    ) -> ValidationResult:
        self.model.eval()
        total_loss, correct = 0.0, 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, _ = self.model(data, experience_buffer=experience_buffer)
                loss = self.loss_fn(output, target)
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        avg_loss = total_loss / len(val_loader)
        num_samples = len(cast(Sized, val_loader.dataset))
        accuracy = 100.0 * correct / num_samples

        print(
            f"{dataset_name} set: Avg loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )

        val_result: ValidationResult = {
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
        return val_result

    def save_checkpoint(self, run_name: str, epoch: int, is_final: bool = False):
        checkpoint_dir = os.path.join("output", run_name, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        suffix = "final" if is_final else f"epoch_{epoch}"
        checkpoint_path = os.path.join(checkpoint_dir, f"{suffix}.pth")

        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "gating_optimizer_state_dict": self.gating_optimizer.state_dict() if self.gating_optimizer else None,
            "validation_history": self.validation_history,
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
        if self.gating_optimizer and checkpoint.get("gating_optimizer_state_dict"):
            self.gating_optimizer.load_state_dict(checkpoint["gating_optimizer_state_dict"])

        self.global_step = checkpoint.get("global_step", 0)
        self.validation_history = checkpoint.get("validation_history", {})
        
        print(f"Checkpoint loaded from {checkpoint_path}. Resuming from global step {self.global_step}.")
