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
from utils.strategies.backpropagation_strategies import (
    SurpriseMinKStrategy,
)
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
        use_narrative_generator: bool = True,
    ) -> Tuple[int, List[StepResult]]:
        self.model.train()
        self.optimizer.zero_grad()

        epoch_results: List[StepResult] = []
        expert_pi_components: List[Dict[str, torch.Tensor]] = []
        gating_pi_components: List[Dict[str, torch.Tensor]] = []
        router_surprise_sum = 0.0
        router_surprise_count = 0
        gating_tau_sum = 0.0
        gating_tau_count = 0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch} ({task_name})", leave=False)
        for batch_idx, (data, target) in enumerate(train_loader_tqdm):
            data, target = data.to(self.device), target.to(self.device)

            current_global_pi_metrics = self.global_pi_calculator.calculate(expert_pi_components) if expert_pi_components else {"pi_score": torch.tensor(0.0, device=self.device)}
            current_pi_score = current_global_pi_metrics.get("pi_score", torch.tensor(0.0, device=self.device))

            output = self.model(data, pi_score=current_pi_score, use_narrative_generator=use_narrative_generator)
            
            all_routing_info = None
            if isinstance(output, tuple) and len(output) > 1 and isinstance(output[1], list):
                logits, all_routing_info = output
            else:
                logits = output[0] if isinstance(output, tuple) else output

            expert_loss = self.loss_fn(logits, target)
            
            vae_kl_loss = torch.tensor(0.0, device=self.device)
            if all_routing_info:
                for info in all_routing_info:
                    if 'vae_kl_loss' in info:
                        vae_kl_loss += info['vae_kl_loss']
            
            total_loss = expert_loss + vae_kl_loss
            
            gating_loss_val = torch.tensor(0.0, device=self.device)
            min_k_expert_indices_per_layer = None

            # Apply strategy components that might modify gradients before backward
            for component in self.strategy_components:
                if isinstance(component, SurpriseMinKStrategy):
                    all_top_indices_pre = [info['top_indices'] for info in all_routing_info] if all_routing_info else None
                    component_metrics = component.apply(self.model, self.optimizer, {}, all_routing_info, all_top_indices_pre)
                    min_k_expert_indices_per_layer = component_metrics.get('min_k_expert_indices')
            
            # Compute gating loss and add to total loss
            if self.is_moe_model and all_routing_info and min_k_expert_indices_per_layer is not None:
                total_gating_loss = torch.tensor(0.0, device=self.device)
                for i, info in enumerate(all_routing_info):
                    top_k_indices = info['top_indices']
                    min_k_indices = min_k_expert_indices_per_layer.get(i, [])
                    total_gating_loss += self.top_k_min_k_loss_fn(info['log_probs'], top_k_indices, min_k_indices, i)
                
                if len(all_routing_info) > 0:
                    gating_loss_val = total_gating_loss / len(all_routing_info)
                    total_loss += gating_loss_val / accumulation_steps # Add gating loss to total loss
            
            total_loss.backward() # Single backward pass
            
            # Collect gradients after backward pass
            expert_grads = [p.grad for p in self.expert_and_base_params if p.grad is not None]
            expert_pi_components.append(self.local_pi_calculator.calculate(expert_grads, expert_loss, logits))

            gating_grads = [p.grad for p in self.gating_params if p.grad is not None]
            gating_local_pi = self.local_pi_calculator.calculate(gating_grads, gating_loss_val, logits)
            gating_pi_components.append(gating_local_pi)
            router_surprise_sum += gating_local_pi['surprise'].item()
            router_surprise_count += 1
            if 'tau' in gating_local_pi:
                gating_tau_sum += gating_local_pi['tau'].item()
                gating_tau_count += 1


            if (batch_idx + 1) % accumulation_steps == 0:
                global_pi_metrics = self.global_pi_calculator.calculate(expert_pi_components)
                expert_pi_components = []

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
                    "gating_selection_accuracy": gating_loss_val.item(), # .item() for logging
                    "accuracy": accuracy,
                    "task_name": task_name,
                    "router_surprise": router_surprise_sum / router_surprise_count if router_surprise_count > 0 else 0.0,
                    "vae_kl_loss": vae_kl_loss.item() if isinstance(vae_kl_loss, torch.Tensor) else vae_kl_loss,
                    "gating_tau": gating_tau_sum / gating_tau_count if gating_tau_count > 0 else 0.0,
                }
                step_result.update(cast(StepResult, {k: v.item() for k, v in global_pi_metrics.items()}))
                
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
                    vae_kl_loss=f"{step_result.get('vae_kl_loss', 0.0):.4f}",
                    lr_mod=f"{step_result.get('lr_mod', 1.0):.4f}"
                )
                router_surprise_sum = 0.0
                router_surprise_count = 0
                gating_tau_sum = 0.0
                gating_tau_count = 0
        
        return global_step, epoch_results

    def _compute_gating_loss(
        self,
        all_routing_info: Optional[List[Dict[str, torch.Tensor]]],
        min_k_expert_indices_per_layer: Optional[Dict[int, List[int]]],
        accumulation_steps: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]: # Changed return type
        if not self.is_moe_model or all_routing_info is None or min_k_expert_indices_per_layer is None:
            return torch.tensor(0.0, device=self.device), None # Changed return type

        total_gating_loss = torch.tensor(0.0, device=self.device)
        for i, info in enumerate(all_routing_info):
            top_k_indices = info['top_indices']
            min_k_indices = min_k_expert_indices_per_layer.get(i, [])
            total_gating_loss += self.top_k_min_k_loss_fn(info['log_probs'], top_k_indices, min_k_indices, i)

        if len(all_routing_info) > 0:
            avg_gating_loss = total_gating_loss / len(all_routing_info)
            # avg_gating_loss = avg_gating_loss / accumulation_steps # Removed division here, moved to total_loss
            return avg_gating_loss, avg_gating_loss # Return tensor
        return torch.tensor(0.0, device=self.device), None # Changed return type

    def validate(self, val_loader: DataLoader, global_step: int, dataset_name: str = "Validation") -> ValidationResult:
        self.model.eval()
        total_loss, correct = 0.0, 0
        all_router_surprises: List[float] = []
        all_taus: List[float] = []
        all_gating_taus: List[float] = []
        all_gating_losses: List[float] = []
        
        expert_local_pi_components: List[Dict[str, torch.Tensor]] = []
        
        for data, target in val_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            output = self.model(data, use_narrative_generator=True)
            all_routing_info = None
            if isinstance(output, tuple) and len(output) > 1 and isinstance(output[1], list):
                logits, all_routing_info = output
            else:
                logits = output[0] if isinstance(output, tuple) else output

            expert_loss = self.loss_fn(logits, target)
            
            gating_loss_val = torch.tensor(0.0, device=self.device)
            if self.is_moe_model and all_routing_info and self.min_k is not None:
                total_gating_loss = torch.tensor(0.0, device=self.device)
                for i, info in enumerate(all_routing_info):
                    top_k_indices_for_val = info['top_indices']
                    min_k_indices_for_val = {i: top_k_indices_for_val.cpu().flatten().tolist()} 
                    total_gating_loss += self.top_k_min_k_loss_fn(info['log_probs'], top_k_indices_for_val, min_k_indices_for_val, i)
                
                if len(all_routing_info) > 0:
                    gating_loss_val = total_gating_loss / len(all_routing_info)
                
            all_gating_losses.append(gating_loss_val.item()) # .item() for logging

            # Calculate PI components for experts and gating
            # Need to enable grad temporarily for surprise calculation if not already in training mode
            with torch.enable_grad():
                # Re-run forward pass to get gradients for PI calculation if not in training mode
                # This is a simplification, ideally gradients should be captured during the main forward pass
                # For validation, we typically don't need gradients for optimization, only for metrics
                # If surprise is needed, a separate forward pass with create_graph=True might be necessary
                # For now, we'll assume gradients are available from the main forward pass if model.eval() is not strictly enforced
                # Or, we can calculate surprise based on parameters directly if they have grads from previous training step
                
                # For validation, we usually don't compute gradients for surprise.
                # If surprise is required for validation metrics, it should be handled carefully.
                # For now, we'll just pass empty gradients for surprise in validation.
                expert_local_pi_components.append(self.local_pi_calculator.calculate([], expert_loss, logits))
                gating_local_pi = self.local_pi_calculator.calculate([], gating_loss_val, logits)
            
            all_router_surprises.append(gating_local_pi['surprise'].item())
            if 'tau' in gating_local_pi:
                all_gating_taus.append(gating_local_pi['tau'].item())

            with torch.no_grad():
                total_loss += expert_loss.item()
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        avg_loss = total_loss / len(val_loader)
        
        dataset = val_loader.dataset
        num_samples = len(cast(Sized, dataset))
        accuracy = 100. * correct / num_samples
        
        # Calculate global PI metrics for validation
        global_pi_metrics_val = self.global_pi_calculator.calculate(expert_local_pi_components)
        avg_pi = global_pi_metrics_val.get("pi_score", torch.tensor(0.0, device=self.device)).item()
        
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