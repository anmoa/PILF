import collections  # Import collections for deque
from typing import Any, Dict, List, Optional, Sized, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F  # Import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.memory_gaussian_moe import (
    MemoryGaussianMoEVisionTransformer,  # Import MemoryGaussianMoEVisionTransformer
)
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
        routing_history_length: int = 10, # New parameter for history length
        historical_routing_loss_weight: float = 0.1, # New parameter for loss weight
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
        self.routing_history_length = routing_history_length # Now refers to number of epochs
        self.historical_routing_loss_weight = historical_routing_loss_weight
        # Stores aggregated routing weights per MoE layer, per task, per epoch
        # Key: task_name, Value: deque of lists of aggregated routing weights (one list per MoE layer)
        self.routing_history_buffers: Dict[str, collections.deque[List[torch.Tensor]]] = collections.defaultdict(lambda: collections.deque(maxlen=self.routing_history_length))

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

            output = self.model(data)
            
            all_routing_info = None
            if isinstance(output, tuple) and len(output) > 1 and isinstance(output[1], list):
                logits, all_routing_info = output
            else:
                logits = output[0] if isinstance(output, tuple) else output

            expert_loss = self.loss_fn(logits, target)
            
            historical_routing_loss_val = torch.tensor(0.0, device=self.device)

            if all_routing_info:
                # Temporarily store current batch's routing weights for aggregation at epoch end
                if not hasattr(self, '_current_epoch_routing_weights'):
                    self._current_epoch_routing_weights = collections.defaultdict(list)
                for i, layer_info in enumerate(all_routing_info):
                    if "weights" in layer_info:
                        self._current_epoch_routing_weights[i].append(layer_info["weights"].detach())

                # Calculate historical routing loss (using epoch-level history)
                current_task_name = task_name # Use the task_name passed to train_one_epoch
                if isinstance(self.model, MemoryGaussianMoEVisionTransformer) and \
                   current_task_name in self.routing_history_buffers and \
                   len(self.routing_history_buffers[current_task_name]) > 0:
                    
                    # Get the latest PI score for the current task
                    # Use current_pi_score from the beginning of the batch loop
                    current_pi_score = current_global_pi_metrics.get("pi_score", torch.tensor(1.0, device=self.device))
                    
                    # Weight for historical influence: (1 - PI)
                    historical_influence_weight = (1.0 - current_pi_score).clamp(0.0, 1.0) # Ensure between 0 and 1
                    
                    # Aggregate historical weights for the current task
                    # routing_history_buffers stores List[torch.Tensor] for each epoch
                    # We need to average across epochs first, then across tokens/batches
                    
                    # Stack all historical epoch-level aggregated weights for this task
                    # Each element in self.routing_history_buffers[current_task_name] is a List[torch.Tensor]
                    # where each torch.Tensor is the aggregated routing weight for one MoE layer for one epoch
                    all_historical_epoch_weights_per_layer = collections.defaultdict(list)
                    for epoch_weights_list in self.routing_history_buffers[current_task_name]:
                        for layer_idx, layer_agg_weights in enumerate(epoch_weights_list):
                            all_historical_epoch_weights_per_layer[layer_idx].append(layer_agg_weights)
                    
                    for i, layer_info in enumerate(all_routing_info):
                        if "weights" in layer_info and i in all_historical_epoch_weights_per_layer:
                            current_weights = layer_info["weights"]
                            
                            # Average across all historical epoch-level aggregated weights for this specific layer
                            historical_weights_aggregated_for_layer = torch.mean(torch.stack(all_historical_epoch_weights_per_layer[i]), dim=0)
                            
                            # Mix current and historical weights based on PI
                            # This mixing happens at the current batch level for the loss calculation
                            mixed_weights = (1.0 - historical_influence_weight) * current_weights + \
                                            historical_influence_weight * historical_weights_aggregated_for_layer
                            
                            # Calculate KL divergence between current weights and mixed weights
                            # Add a small epsilon for numerical stability if using log
                            epsilon = 1e-9
                            kl_div = F.kl_div(F.log_softmax(current_weights + epsilon, dim=-1),
                                              F.log_softmax(mixed_weights + epsilon, dim=-1),
                                              reduction='batchmean')
                            historical_routing_loss_val += kl_div
            
            total_loss = expert_loss
            
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
            
            # Add historical routing loss to total loss
            if isinstance(self.model, MemoryGaussianMoEVisionTransformer) and historical_routing_loss_val > 0:
                total_loss += self.historical_routing_loss_weight * historical_routing_loss_val / accumulation_steps
            
            total_loss.backward() # Single backward pass
            
            # Collect gradients after backward pass
            expert_grads = [p.grad for p in self.expert_and_base_params if p.grad is not None]
            expert_pi_components.append(self.local_pi_calculator.calculate(expert_grads, expert_loss, logits))

            if self.is_moe_model: # Only collect gating grads for MoE models
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
                    "gating_tau": gating_tau_sum / gating_tau_count if gating_tau_count > 0 else 0.0,
                    "historical_routing_loss": historical_routing_loss_val.item() if isinstance(historical_routing_loss_val, torch.Tensor) else historical_routing_loss_val, # Log historical routing loss
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
                    historical_routing_loss=f"{step_result.get('historical_routing_loss', 0.0):.4f}", # Display historical routing loss
                    lr_mod=f"{step_result.get('lr_mod', 1.0):.4f}"
                )
                router_surprise_sum = 0.0
                router_surprise_count = 0
                gating_tau_sum = 0.0
                gating_tau_count = 0
        
        # Aggregate and store epoch-level routing weights at the end of the epoch
        if hasattr(self, '_current_epoch_routing_weights') and self._current_epoch_routing_weights:
            aggregated_epoch_weights_list: List[torch.Tensor] = []
            for i in sorted(self._current_epoch_routing_weights.keys()):
                # Concatenate all batch weights for this layer in the current epoch
                if self._current_epoch_routing_weights[i]:
                    # Stack first, then mean across the batch dimension (dim=0)
                    # This handles variable batch sizes by averaging over all tokens collected in the epoch
                    all_batch_weights_for_layer = torch.cat(self._current_epoch_routing_weights[i], dim=0)
                    aggregated_layer_weights = torch.mean(all_batch_weights_for_layer, dim=0)
                    aggregated_epoch_weights_list.append(aggregated_layer_weights)
            
            if aggregated_epoch_weights_list:
                self.routing_history_buffers[task_name].append(aggregated_epoch_weights_list)
            
            # Clear for next epoch
            del self._current_epoch_routing_weights

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
            
            output = self.model(data)
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