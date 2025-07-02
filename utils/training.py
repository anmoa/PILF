from typing import Any, Dict, List, Optional, Sized, Tuple, cast

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.gating_loss import GatingSelectionLoss
from utils.logging import TensorBoardLogger
from utils.pi_calculator import PiCalculator
from utils.strategies.backpropagation_strategies import SurpriseMinKStrategy
from utils.strategies.base_strategy import StrategyComponent
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
            
            self.gating_accuracy_loss_fn = GatingSelectionLoss()

    def train_one_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        global_step: int,
        accumulation_steps: int,
        task_name: str,
    ) -> Tuple[int, List[StepResult]]:
        self.model.train()
        self.optimizer.zero_grad() # Clear gradients at the start of each epoch

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

            expert_loss.backward(retain_graph=True) # Compute gradients for expert and base parameters
            
            # Calculate PI metrics before applying strategies that might modify gradients
            all_grads = [p.grad for p in self.model.parameters() if p.grad is not None]
            pi_metrics = self.pi_monitor.calculate(all_grads, expert_loss, logits)

            min_k_expert_indices_per_layer = None
            # Apply strategies that might modify gradients (e.g., SurpriseMinKStrategy)
            for component in self.strategy_components:
                if isinstance(component, SurpriseMinKStrategy):
                    all_top_indices_pre = [info['top_indices'] for info in all_routing_info] if all_routing_info else None
                    component_metrics = component.apply(self.model, self.optimizer, pi_metrics, all_routing_info, all_top_indices_pre)
                    min_k_expert_indices_per_layer = component_metrics.get('min_k_expert_indices')
            
            gating_loss_val, gating_loss_tensor = self._compute_gating_loss(
                all_routing_info, min_k_expert_indices_per_layer, accumulation_steps
            )
            if gating_loss_tensor is not None:
                gating_loss_tensor.backward() # Compute gradients for gating parameters, accumulating if needed

            if (batch_idx + 1) % accumulation_steps == 0: # Perform optimization step
                if hasattr(self.model, 'zero_inactive_expert_grads') and all_routing_info:
                    self.model.zero_inactive_expert_grads(all_routing_info) # Zero out gradients for inactive experts
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # Clip gradients
                self.optimizer.step() # Perform optimizer step
                self.optimizer.zero_grad() # Clear gradients after step
                
                pred = logits.argmax(dim=1, keepdim=True) # Calculate accuracy
                correct = pred.eq(target.view_as(pred)).sum().item()
                accuracy = 100. * correct / len(data)
                
                step_result: StepResult = {
                    "global_step": global_step,
                    "loss": expert_loss.item(),
                    "gating_selection_accuracy": gating_loss_val, # Now represents the BCE loss
                    "accuracy": accuracy,
                    "task_name": task_name,
                }
                step_result.update(cast(StepResult, pi_metrics))
                
                # Apply strategies that might modify learning rates or collect additional metrics
                for component in self.strategy_components:
                    all_top_indices = [info['top_indices'] for info in all_routing_info] if all_routing_info else None
                    component_metrics = component.apply(self.model, self.optimizer, pi_metrics, all_routing_info, all_top_indices)
                    step_result.update(component_metrics)

                epoch_results.append(step_result)
                global_step += 1

                logger = TensorBoardLogger(self.writer, global_step)
                logger.log_metrics(cast(Dict[str, Any], step_result), task_name, scope="Train")

                train_loader_tqdm.set_postfix(
                    loss=f"{step_result.get('loss', 0.0):.4f}",
                    acc=f"{step_result.get('accuracy', 0.0):.2f}%",
                    pi=f"{step_result.get('pi_score', 0.0):.4f}",
                    surprise=f"{step_result.get('surprise', 0.0):.4f}",
                    lr_mod=f"{step_result.get('lr_mod', 1.0):.4f}" # Add lr_mod to postfix
                )
        
        return global_step, epoch_results

    def validate(self, val_loader: DataLoader, global_step: int, dataset_name: str = "Validation") -> ValidationResult:
        self.model.eval()
        total_loss, correct = 0.0, 0
        all_pi_scores: List[float] = []
        all_surprises: List[float] = []
        all_taus: List[float] = []
        all_gating_taus: List[float] = []
        all_gating_losses: List[float] = []

        for data, target in val_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            with torch.enable_grad(): # Keep grad enabled for pi_metrics calculation
                output = self.model(data)
                all_routing_info = None
                if isinstance(output, tuple) and len(output) > 1 and isinstance(output[1], list):
                    logits, all_routing_info = output
                else:
                    logits = output[0] if isinstance(output, tuple) else output

                loss_epsilon = self.loss_fn(logits, target)
                loss_epsilon.backward() # Compute gradients for pi_metrics

                all_grads = [p.grad for p in self.model.parameters() if p.grad is not None]
                pi_metrics = self.pi_monitor.calculate(all_grads, loss_epsilon, logits)
                
                all_pi_scores.append(pi_metrics['pi_score'])
                all_surprises.append(pi_metrics['surprise'])
                all_taus.append(pi_metrics['tau'])
                if 'gating_tau' in pi_metrics:
                    all_gating_taus.append(pi_metrics['gating_tau'])

                # Calculate gating loss for validation
                gating_loss_val = 0.0
                if self.is_moe_model and all_routing_info and self.min_k is not None:
                    total_gating_loss = torch.tensor(0.0, device=self.device)
                    for i, info in enumerate(all_routing_info):
                        # In validation, we don't have surprise-based min_k, so we use top_k from logits
                        _, top_indices = torch.topk(info['log_probs'], self.min_k, dim=-1)
                        min_k_indices_for_val = {i: top_indices.cpu().flatten().tolist()}
                        total_gating_loss += self.gating_accuracy_loss_fn(info['log_probs'], min_k_indices_for_val, i)
                    
                    if len(all_routing_info) > 0:
                        gating_loss_val = (total_gating_loss / len(all_routing_info)).item()
                all_gating_losses.append(gating_loss_val)

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
        avg_gating_loss = sum(all_gating_losses) / len(all_gating_losses) if all_gating_losses else None

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
        if avg_gating_loss is not None:
            summary_parts.append(f"Avg Gating Loss: {avg_gating_loss:.4f}")
        
        print(", ".join(filter(None, summary_parts)))

        val_metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "pi_score": avg_pi,
            "surprise": avg_surprise,
            "tau": avg_tau,
            "gating_tau": avg_gating_tau,
            "gating_loss": avg_gating_loss,
        }
        logger = TensorBoardLogger(self.writer, global_step)
        logger.log_metrics({k: v for k, v in val_metrics.items() if v is not None}, dataset_name, scope="Validation")

        return avg_loss, accuracy, avg_pi, avg_surprise, avg_tau, avg_gating_tau, avg_gating_loss

    def _compute_gating_loss(self, all_routing_info, min_k_expert_indices_per_layer, accumulation_steps) -> Tuple[float, Optional[torch.Tensor]]:
        gating_loss_val = 0.0
        gating_loss_tensor: Optional[torch.Tensor] = None

        if self.is_moe_model and all_routing_info and min_k_expert_indices_per_layer:
            total_gating_loss = torch.tensor(0.0, device=self.device)
            for i, info in enumerate(all_routing_info):
                if i in min_k_expert_indices_per_layer:
                    layer_min_k_indices = {i: min_k_expert_indices_per_layer[i]}
                    total_gating_loss += self.gating_accuracy_loss_fn(info['log_probs'], layer_min_k_indices, i)
            
            gating_loss_tensor = total_gating_loss / len(all_routing_info) / accumulation_steps
            gating_loss_val = gating_loss_tensor.item()
        
        return gating_loss_val, gating_loss_tensor

    # This method is no longer needed as gradients are directly accumulated and then zeroed by the optimizer.
    # def _assign_grads_and_step(self, expert_grads, gating_grads):
    #     pass
