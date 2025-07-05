from typing import Any, Dict, cast

import torch
from tqdm import tqdm

from utils.logging.metrics_logger import TensorBoardLogger
from utils.logging.types import StepResult
from utils.trainer import Trainer


class BaseTrainLoop:
    def __init__(self, trainer: Trainer):
        self.trainer = trainer
        self.model = trainer.model
        self.optimizer = trainer.optimizer
        self.loss_fn = trainer.loss_fn
        self.device = trainer.device
        self.writer = trainer.writer
        self.pi_calculator = trainer.pi_calculator
        self.strategy_components = trainer.strategy_components

    def train_one_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        epoch: int,
        task_name: str,
    ) -> None:
        self.model.train()
        
        train_loader_tqdm = tqdm(
            train_loader,
            desc=f"Epoch {epoch} ({task_name})",
            leave=False,
        )

        for data, target in train_loader_tqdm:
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            logits, all_routing_info = self.model(data)
            
            loss = self.loss_fn(logits, target)
            loss.backward()

            pi_metrics = self.pi_calculator.calculate(
                loss=loss,
                logits=logits,
                gradients=[p.grad for p in self.model.parameters() if p.grad is not None],
            )

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            step_result: StepResult = {}
            for component in self.strategy_components:
                component_metrics = component.apply(self.model, self.optimizer, pi_metrics)
                step_result.update(component_metrics)

            self.optimizer.step()

            pred = logits.argmax(dim=1, keepdim=True)
            accuracy = 100.0 * pred.eq(target.view_as(pred)).sum().item() / len(data)

            step_result.update({
                "global_step": self.trainer.global_step,
                "loss": loss.item(),
                "accuracy": accuracy,
                "task_name": task_name,
            })
            
            if all_routing_info:
                top_k_indices_by_layer = {
                    i: info["top_indices"].flatten().detach().cpu().tolist() 
                    for i, info in enumerate(all_routing_info)
                }
                step_result["top_k_expert_indices"] = top_k_indices_by_layer

            pi_metrics_items = {k: v.item() for k, v in pi_metrics.items()}
            step_result.update(pi_metrics_items)  # type: ignore

            self.trainer.epoch_results.append(step_result)
            
            logger = TensorBoardLogger(self.writer, self.trainer.global_step)
            logger.log_metrics(cast(Dict[str, Any], step_result), task_name, scope="Train")
            
            train_loader_tqdm.set_postfix(
                loss=f"{step_result.get('loss', 0.0):.4f}",
                acc=f"{step_result.get('accuracy', 0.0):.2f}%",
                pi=f"{step_result.get('pi_score', 0.0):.4f}",
            )
            self.trainer.global_step += 1