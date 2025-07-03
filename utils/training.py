from typing import Any, Dict, List, Optional, Sized, Tuple, cast

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.gating_loss import TopKMinKLoss
from utils.logging import TensorBoardLogger
from utils.pi_calculator import GlobalPiCalculator, LocalPiCalculator
from utils.strategies.backpropagation_strategies import SurpriseMinKStrategy
from utils.strategies.base_strategy import StrategyComponent
from utils.types import StepResult, ValidationResult


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        global_pi_calculator: GlobalPiCalculator,
        local_pi_calculator: LocalPiCalculator,
        strategy_components: List[StrategyComponent],
        device: torch.device,
        writer: SummaryWriter,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.global_pi_calculator = global_pi_calculator
        self.local_pi_calculator = local_pi_calculator
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
            self.top_k = 1

        self.is_moe_model = hasattr(self.model, 'get_param_groups')
        if self.is_moe_model:
            param_groups = self.model.get_param_groups()
            self.base_params = param_groups[0]['params']
            self.gating_params = param_groups[1]['params']
            self.expert_params = param_groups[2]['params']
            self.expert_and_base_params = self.base_params + self.expert_params
            
            self.top_k_min_k_loss_fn = TopKMinKLoss()

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
        expert_pi_components: List[Dict[str, float]] = []
        gating_pi_components: List[Dict[str, float]] = []
        router_surprise_sum = 0.0
        router_surprise_count = 0

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
            
            # Calculate local PI components for expert network
            expert_grads = [p.grad for p in self.expert_and_base_params if p.grad is not None]
            expert_pi_components.append(self.local_pi_calculator.calculate(expert_grads, expert_loss, logits))

            min_k_expert_indices_per_layer = None
            for component in self.strategy_components:
                if isinstance(component, SurpriseMinKStrategy):
                    all_top_indices_pre = [info['top_indices'] for info in all_routing_info] if all_routing_info else None
                    component_metrics = component.apply(self.model, self.optimizer, {}, all_routing_info, all_top_indices_pre) # PI metrics are not needed here
                    min_k_expert_indices_per_layer = component_metrics.get('min_k_expert_indices')
            
            gating_loss_val, gating_loss_tensor = self._compute_gating_loss(
                all_routing_info, min_k_expert_indices_per_layer, accumulation_steps
            )
            if gating_loss_tensor is not None:
                gating_loss_tensor.backward(retain_graph=True) # Retain graph for gating PI calculation

            # Calculate local PI components for gating network
            gating_grads = [p.grad for p in self.gating_params if p.grad is not None]
            gating_local_pi = self.local_pi_calculator.calculate(gating_grads, gating_loss_tensor if gating_loss_tensor is not None else torch.tensor(0.0), logits)
            gating_pi_components.append(gating_local_pi)
            router_surprise_sum += gating_local_pi['surprise']
            router_surprise_count += 1


            if (batch_idx + 1) % accumulation_steps == 0:
                # Calculate global PI after accumulating local components (only for expert network)
                global_pi_metrics = self.global_pi_calculator.calculate(expert_pi_components)
                expert_pi_components = [] # Reset for next accumulation step

                if hasattr(self.model, 'zero_inactive_expert_grads') and all_routing_info:
                    self.model.zero_inactive_expert_grads(all_routing_info)
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                pred = logits.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()
                accuracy = 100. * correct / len(data)
                
                step_result: StepResult = {
                    "global_step": global_step,
                    "loss": expert_loss.item(),
                    "gating_selection_accuracy": gating_loss_val,
                    "accuracy": accuracy,
                    "task_name": task_name,
                    "router_surprise": router_surprise_sum / router_surprise_count if router_surprise_count > 0 else 0.0,
                }
                step_result.update(cast(StepResult, global_pi_metrics))
                
                for component in self.strategy_components:
                    all_top_indices = [info['top_indices'] for info in all_routing_info] if all_routing_info else None
                    component_metrics = component.apply(self.model, self.optimizer, global_pi_metrics, all_routing_info, all_top_indices)
                    step_result.update(component_metrics)

                epoch_results.append(step_result)
                global_step += 1

                logger = TensorBoardLogger(self.writer, global_step)
                logger.log_metrics(cast(Dict[str, Any], step_result), task_name, scope="Train")

                train_loader_tqdm.set_postfix(
                    loss=f"{step_result.get('loss', 0.0):.4f}",
                    acc=f"{step_result.get('accuracy', 0.0):.2f}%",
                    pi=f"{step_result.get('pi_score', 0.0):.4f}",
                    router_surprise=f"{step_result.get('router_surprise', 0.0):.4f}",
                    lr_mod=f"{step_result.get('lr_mod', 1.0):.4f}"
                )
                router_surprise_sum = 0.0
                router_surprise_count = 0
        
        return global_step, epoch_results

    def _compute_gating_loss(
        self,
        all_routing_info: Optional[List[Dict[str, torch.Tensor]]],
        min_k_expert_indices_per_layer: Optional[Dict[int, List[int]]],
        accumulation_steps: int,
    ) -> Tuple[float, Optional[torch.Tensor]]:
        if not self.is_moe_model or all_routing_info is None or min_k_expert_indices_per_layer is None:
            return 0.0, None

        total_gating_loss = torch.tensor(0.0, device=self.device)
        for i, info in enumerate(all_routing_info):
            top_k_indices = info['top_indices']
            min_k_indices = min_k_expert_indices_per_layer.get(i, [])
            total_gating_loss += self.top_k_min_k_loss_fn(info['log_probs'], top_k_indices, min_k_indices, i)

        if len(all_routing_info) > 0:
            avg_gating_loss = total_gating_loss / len(all_routing_info)
            # Scale the loss by accumulation steps to account for gradient accumulation
            avg_gating_loss = avg_gating_loss / accumulation_steps
            return avg_gating_loss.item(), avg_gating_loss
        return 0.0, None

    def validate(self, val_loader: DataLoader, global_step: int, dataset_name: str = "Validation") -> ValidationResult:
        self.model.eval()
        total_loss, correct = 0.0, 0
        all_pi_scores: List[float] = []
        all_router_surprises: List[float] = [] 
        all_taus: List[float] = []
        all_gating_taus: List[float] = []
        all_gating_losses: List[float] = []
        
        expert_local_pi_components: List[Dict[str, float]] = []
        gating_local_pi_components: List[Dict[str, float]] = []

        for data, target in val_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            with torch.enable_grad(): # Keep grad enabled for PI calculation
                output = self.model(data)
                all_routing_info = None
                if isinstance(output, tuple) and len(output) > 1 and isinstance(output[1], list):
                    logits, all_routing_info = output
                else:
                    logits = output[0] if isinstance(output, tuple) else output

                loss_epsilon = self.loss_fn(logits, target)
                loss_epsilon.backward(retain_graph=True) # Retain graph for gating PI calculation

                # Calculate local PI components for expert network
                expert_grads = [p.grad for p in self.expert_and_base_params if p.grad is not None]
                expert_local_pi_components.append(self.local_pi_calculator.calculate(expert_grads, loss_epsilon, logits))
                
                gating_loss_val = 0.0
                if self.is_moe_model and all_routing_info and self.min_k is not None:
                    total_gating_loss = torch.tensor(0.0, device=self.device)
                    for i, info in enumerate(all_routing_info):
                        top_k_indices_for_val = info['top_indices']
                        # In validation, min_k_expert_indices should reflect the actual top_k chosen,
                        # as there's no backprop to influence min_k selection.
                        min_k_indices_for_val = {i: top_k_indices_for_val.cpu().flatten().tolist()} 
                        total_gating_loss += self.top_k_min_k_loss_fn(info['log_probs'], top_k_indices_for_val, min_k_indices_for_val, i)
                    
                    if len(all_routing_info) > 0:
                        gating_loss_val = (total_gating_loss / len(all_routing_info)).item()
                    
                    # Removed total_gating_loss.backward() as per user feedback for validation
                all_gating_losses.append(gating_loss_val)

                # Calculate local PI components for gating network
                gating_grads = [p.grad for p in self.gating_params if p.grad is not None]
                gating_local_pi = self.local_pi_calculator.calculate(gating_grads, total_gating_loss if self.is_moe_model and all_routing_info and self.min_k is not None else torch.tensor(0.0), logits)
                gating_local_pi_components.append(gating_local_pi)

                # Calculate global PI for expert network
                global_pi_metrics = self.global_pi_calculator.calculate(expert_local_pi_components)
                all_pi_scores.append(global_pi_metrics['pi_score'])
                all_taus.append(global_pi_metrics['tau'])

                # Collect router surprise separately
                all_router_surprises.append(gating_local_pi['surprise'])
                if 'gating_tau' in gating_local_pi: 
                    all_gating_taus.append(gating_local_pi['tau']) 

            with torch.no_grad():
                total_loss += loss_epsilon.item()
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        avg_loss = total_loss / len(val_loader)
        
        dataset = val_loader.dataset
        num_samples = len(cast(Sized, dataset))
        accuracy = 100. * correct / num_samples
        avg_pi = sum(all_pi_scores) / len(all_pi_scores) if all_pi_scores else 0.0
        avg_router_surprise = sum(all_router_surprises) / len(all_router_surprises) if all_router_surprises else 0.0 
        avg_tau = sum(all_taus) / len(all_taus) if all_taus else 0.0
        avg_gating_tau = sum(all_gating_taus) / len(all_gating_taus) if all_gating_taus else None
        avg_gating_loss = sum(all_gating_losses) / len(all_gating_losses) if all_gating_losses else None

        summary_parts = [
            f"{dataset_name} set:",
            f"Avg loss: {avg_loss:.4f}",
            f"Accuracy: {accuracy:.2f}%",
            f"Avg PI: {avg_pi:.4f}" if avg_pi is not None else "",
            f"Avg Router Surprise: {avg_router_surprise:.4f}" if avg_router_surprise is not None else "", 
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
            "router_surprise": avg_router_surprise, 
            "tau": avg_tau,
            "gating_tau": avg_gating_tau,
            "gating_loss": avg_gating_loss,
        }
        logger = TensorBoardLogger(self.writer, global_step)
        logger.log_metrics({k: v for k, v in val_metrics.items() if v is not None}, dataset_name, scope="Validation")

        return avg_loss, accuracy, avg_pi, avg_router_surprise, avg_tau, avg_gating_tau, avg_gating_loss
