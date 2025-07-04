from typing import Any, Dict, List, Tuple, cast

import torch
from tqdm import tqdm

from utils.logging.metrics_logger import TensorBoardLogger
from utils.logging.types import StepResult
from utils.pi_calculator import PICalculator
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
        self.is_moe_model = trainer.is_moe_model

    def train_one_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        epoch: int,
        global_step: int,
        accumulation_steps: int,
        task_name: str,
    ) -> Tuple[int, List[StepResult]]:
        self.model.train()
        self.optimizer.zero_grad()

        epoch_results: List[StepResult] = []
        
        num_updates_per_epoch = len(train_loader) // accumulation_steps
        train_loader_tqdm = tqdm(
            range(num_updates_per_epoch),
            desc=f"Epoch {epoch} ({task_name})",
            leave=False,
        )
        data_iter = iter(train_loader)

        for _ in train_loader_tqdm:
            accumulated_loss = 0.0
            final_logits = None
            final_target = None
            final_routing_info = None

            for i in range(accumulation_steps):
                try:
                    data, target = next(data_iter)
                    data, target = data.to(self.device), target.to(self.device)
                    
                    output = self.model(data)
                    
                    logits, all_routing_info = (output[0], output[1]) if isinstance(output, tuple) else (output, None)
                    
                    loss = self.loss_fn(logits, target)
                    scaled_loss = loss / accumulation_steps
                    scaled_loss.backward(retain_graph=True)

                    accumulated_loss += scaled_loss.item()
                    if i == accumulation_steps - 1:
                        final_logits = logits
                        final_target = target
                        final_routing_info = all_routing_info
                
                except StopIteration:
                    break
            
            if final_logits is None:
                continue
            
            gating_grads = [p.grad.clone() for p in self.trainer.gating_params if p.grad is not None]
            expert_grads = [p.grad.clone() for p in self.trainer.expert_params if p.grad is not None]
            
            pi_metrics = self.pi_calculator.calculate(
                loss=torch.tensor(accumulated_loss, device=self.device),
                logits=final_logits,
                gradients={"gating": gating_grads, "experts": expert_grads},
            )

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

            pred = final_logits.argmax(dim=1, keepdim=True)
            if final_target is None:
                continue
            correct = pred.eq(final_target.view_as(pred)).sum().item()
            accuracy = 100.0 * correct / len(final_target)

            step_result: StepResult = {
                "global_step": global_step,
                "loss": accumulated_loss * accumulation_steps,
                "accuracy": accuracy,
                "task_name": task_name,
            }
            step_result.update(pi_metrics)

            for component in self.strategy_components:
                all_top_indices = [info["top_indices"] for info in final_routing_info] if final_routing_info else None
                component_metrics = component.apply(self.model, self.optimizer, pi_metrics, final_routing_info, all_top_indices)
                step_result.update(component_metrics)

            epoch_results.append(step_result)
            
            logger = TensorBoardLogger(self.writer, global_step)
            logger.log_metrics(cast(Dict[str, Any], step_result), task_name, scope="Train")
            
            train_loader_tqdm.set_postfix(
                loss=f"{step_result.get('loss', 0.0):.4f}",
                acc=f"{step_result.get('accuracy', 0.0):.2f}%",
                pi=f"{step_result.get('pi_score', 0.0):.4f}",
            )
            global_step += 1

        return global_step, epoch_results
