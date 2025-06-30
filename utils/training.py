from typing import Dict, List, Optional, Sized, Tuple, cast

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.pi_calculator import PiCalculator
from utils.gating_loss import GatingSelectionAccuracyLoss
from utils.strategies import StrategyComponent, SurpriseMinKStrategy
from utils.metrics import calculate_gating_selection_accuracy
from utils.logging import TensorBoardLogger
from utils.types import StepResult, ValidationResult

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        pi_monitor: PiCalculator,
        strategy_components: List[StrategyComponent],
        device: torch.device,
        writer: SummaryWriter,
        gating_loss_config: Optional[Dict[str, float]] = None, # This will be deprecated
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.pi_monitor = pi_monitor
        self.strategy_components = strategy_components
        self.device = device
        self.writer = writer
        self.min_k: Optional[int] = None
        self.top_k: Optional[int] = None

        for component in self.strategy_components:
            if isinstance(component, SurpriseMinKStrategy):
                self.min_k = component.min_k
        
        if hasattr(self.model, 'top_k'):
            self.top_k = self.model.top_k
        else:
            self.top_k = 1 # Default to 1 if not specified, though MoE models should have it

        self.is_moe_model = hasattr(self.model, 'get_param_groups')
        if self.is_moe_model:
            param_groups = self.model.get_param_groups()
            self.base_params = param_groups[0]['params']
            self.gating_params = param_groups[1]['params']
            self.expert_params = param_groups[2]['params']
            self.expert_and_base_params = self.base_params + self.expert_params
            
            if self.top_k is not None:
                self.gating_accuracy_loss_fn = GatingSelectionAccuracyLoss(top_k=self.top_k)
            else:
                self.gating_accuracy_loss_fn = GatingSelectionAccuracyLoss(top_k=1)

    def train_one_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        global_step: int,
        accumulation_steps: int,
        task_name: str,
    ) -> Tuple[int, List[StepResult]]:
        self.model.train()
        self.optimizer.zero_grad()

        epoch_results: List[StepResult] = []

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch} ({task_name})", leave=False)
        for batch_idx, (data, target) in enumerate(train_loader_tqdm):
            data, target = data.to(self.device), target.to(self.device)

            output = self.model(data)
            
            all_routing_info = None
            if isinstance(output, tuple) and len(output) > 1 and isinstance(output[1], list):
                logits, all_routing_info = output
            else:
                logits = output[0] if isinstance(output, tuple) else output

            expert_loss = self.loss_fn(logits, target)
            
            gating_loss_val = 0.0
            min_k_expert_indices_per_layer = None

            expert_loss.backward(retain_graph=True)
            
            pi_metrics_pre = {} 

            for component in self.strategy_components:
                if isinstance(component, SurpriseMinKStrategy):
                    all_top_indices_pre = [info['top_indices'] for info in all_routing_info] if all_routing_info else None
                    component_metrics = component.apply(self.model, self.optimizer, pi_metrics_pre, all_routing_info, all_top_indices_pre)
                    min_k_expert_indices_per_layer = component_metrics.get('min_k_expert_indices')

            gating_loss_val, expert_grads, gating_grads = self._compute_grads(
                expert_loss, all_routing_info, min_k_expert_indices_per_layer, accumulation_steps
            )

            if (batch_idx + 1) % accumulation_steps == 0:
                self._assign_grads_and_step(expert_grads, gating_grads)
                
                if hasattr(self.model, 'zero_inactive_expert_grads') and all_routing_info:
                    self.model.zero_inactive_expert_grads(all_routing_info)
                
                all_grads = [p.grad for p in self.model.parameters() if p.grad is not None]
                pi_metrics = self.pi_monitor.calculate(all_grads, expert_loss, logits)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                
                pred = logits.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()
                accuracy = 100. * correct / len(data)
                
                step_result: StepResult = {
                    "global_step": global_step,
                    "loss": expert_loss.item(),
                    "gating_selection_accuracy": gating_loss_val,
                    "accuracy": accuracy,
                    "task_name": task_name,
                    **pi_metrics,
                }
                
                for component in self.strategy_components:
                    all_top_indices = [info['top_indices'] for info in all_routing_info] if all_routing_info else None
                    component_metrics = component.apply(self.model, self.optimizer, pi_metrics, all_routing_info, all_top_indices)
                    step_result.update(component_metrics)
                
                if self.min_k is not None and all_routing_info:
                    min_k_indices_from_step = step_result.get('min_k_expert_indices')
                    if min_k_indices_from_step:
                        gating_selection_accuracy_val = calculate_gating_selection_accuracy(
                            all_routing_info, min_k_indices_from_step, self.top_k
                        )
                        step_result["gating_selection_accuracy"] = gating_selection_accuracy_val

                epoch_results.append(step_result)
                global_step += 1

                logger = TensorBoardLogger(self.writer, global_step)
                logger.log_metrics(step_result, task_name, scope="Train")

                train_loader_tqdm.set_postfix(
                    loss=f"{step_result.get('loss', 0.0):.4f}",
                    acc=f"{step_result.get('accuracy', 0.0):.2f}%",
                    pi=f"{step_result.get('pi_score', 0.0):.4f}",
                    surprise=f"{step_result.get('surprise', 0.0):.4f}",
                )
        
        return global_step, epoch_results

    def validate(self, val_loader: DataLoader, global_step: int, dataset_name: str = "Validation") -> ValidationResult:
        self.model.eval()
        total_loss, correct = 0.0, 0
        all_pi_scores: List[float] = []
        all_surprises: List[float] = []
        all_taus: List[float] = []
        all_gating_taus: List[float] = []
        all_gating_selection_accuracies: List[float] = []

        for data, target in val_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            with torch.enable_grad():
                output = self.model(data)
                all_routing_info = None
                if isinstance(output, tuple) and len(output) > 1 and isinstance(output[1], list):
                    logits, all_routing_info = output
                else:
                    logits = output[0] if isinstance(output, tuple) else output

                loss_epsilon = self.loss_fn(logits, target)
                loss_epsilon.backward()
                
                all_grads = [p.grad for p in self.model.parameters() if p.grad is not None]
                pi_metrics = self.pi_monitor.calculate(all_grads, loss_epsilon, logits)
                
                all_pi_scores.append(pi_metrics['pi_score'])
                all_surprises.append(pi_metrics['surprise'])
                all_taus.append(pi_metrics['tau'])
                if 'gating_tau' in pi_metrics:
                    all_gating_taus.append(pi_metrics['gating_tau'])

                if self.min_k is not None and all_routing_info:
                    # In validation, we don't have surprise-based min_k, so we check against top_k from logits
                    min_k_indices_from_logits = {}
                    for i, info in enumerate(all_routing_info):
                        _, top_indices = torch.topk(info['log_probs'], self.min_k, dim=-1)
                        min_k_indices_from_logits[i] = top_indices.cpu().flatten().tolist()

                    gating_selection_accuracy_val = calculate_gating_selection_accuracy(
                        all_routing_info, min_k_indices_from_logits, self.top_k
                    )
                    all_gating_selection_accuracies.append(gating_selection_accuracy_val)

            with torch.no_grad():
                total_loss += loss_epsilon.item()
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        avg_loss = total_loss / len(val_loader)
        
        dataset = val_loader.dataset
        num_samples = len(cast(Sized, dataset))
        accuracy = 100. * correct / num_samples
        avg_pi = sum(all_pi_scores) / len(all_pi_scores) if all_pi_scores else 0.0
        avg_surprise = sum(all_surprises) / len(all_surprises) if all_surprises else 0.0
        avg_tau = sum(all_taus) / len(all_taus) if all_taus else 0.0
        avg_gating_tau = sum(all_gating_taus) / len(all_gating_taus) if all_gating_taus else None
        avg_gating_selection_accuracy = sum(all_gating_selection_accuracies) / len(all_gating_selection_accuracies) if all_gating_selection_accuracies else None

        summary_parts = [
            f"{dataset_name} set:",
            f"Avg loss: {avg_loss:.4f}",
            f"Accuracy: {accuracy:.2f}%",
            f"Avg PI: {avg_pi:.4f}" if avg_pi is not None else "",
            f"Avg Surprise: {avg_surprise:.4f}" if avg_surprise is not None else "",
            f"Avg Tau: {avg_tau:.4f}" if avg_tau is not None else ""
        ]
        if avg_gating_tau is not None and avg_gating_tau > 0:
            summary_parts.append(f"Avg Gating Tau: {avg_gating_tau:.4f}")
        if avg_gating_selection_accuracy is not None:
            summary_parts.append(f"Avg Gating Selection Accuracy: {avg_gating_selection_accuracy:.2f}%")
        
        print(", ".join(filter(None, summary_parts)))

        val_metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "pi_score": avg_pi,
            "surprise": avg_surprise,
            "tau": avg_tau,
            "gating_tau": avg_gating_tau,
            "gating_selection_accuracy": avg_gating_selection_accuracy,
        }
        logger = TensorBoardLogger(self.writer, global_step)
        logger.log_metrics({k: v for k, v in val_metrics.items() if v is not None}, dataset_name, scope="Validation")

        return avg_loss, accuracy, avg_pi, avg_surprise, avg_tau, avg_gating_tau, avg_gating_selection_accuracy

    def _compute_grads(self, expert_loss, all_routing_info, min_k_expert_indices_per_layer, accumulation_steps):
        gating_loss_val = 0.0
        if self.is_moe_model and all_routing_info and min_k_expert_indices_per_layer:
            total_gating_loss = torch.tensor(0.0, device=self.device)
            for i, info in enumerate(all_routing_info):
                if i in min_k_expert_indices_per_layer:
                    layer_min_k_indices = {i: min_k_expert_indices_per_layer[i]}
                    total_gating_loss += self.gating_accuracy_loss_fn(info['log_probs'], layer_min_k_indices)
            
            gating_loss = total_gating_loss / len(all_routing_info)
            gating_loss_val = gating_loss.item()

            expert_grads = torch.autograd.grad(expert_loss / accumulation_steps, self.expert_and_base_params, retain_graph=True, allow_unused=True)
            gating_grads = torch.autograd.grad(gating_loss / accumulation_steps, self.gating_params, allow_unused=True)
        else:
            expert_grads = torch.autograd.grad(expert_loss / accumulation_steps, list(self.model.parameters()), allow_unused=True)
            gating_grads = tuple([None] * len(self.gating_params))
        
        return gating_loss_val, expert_grads, gating_grads

    def _assign_grads_and_step(self, expert_grads, gating_grads):
        self.optimizer.zero_grad()
        if self.is_moe_model:
            for param, grad in zip(self.expert_and_base_params, expert_grads):
                if grad is not None:
                    param.grad = grad
            for param, grad in zip(self.gating_params, gating_grads):
                if grad is not None:
                    if param.grad is not None:
                        param.grad += grad
                    else:
                        param.grad = grad
        else:
            for param, grad in zip(self.model.parameters(), expert_grads):
                if grad is not None:
                    param.grad = grad
