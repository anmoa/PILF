from typing import TYPE_CHECKING, Any, Dict, List, cast

import torch
from tqdm import tqdm

if TYPE_CHECKING:
    from models.moe_layers import MemoryGaussianMoELayer
from utils.experience_buffer import PrototypingExperienceBuffer
from utils.logging.metrics_logger import TensorBoardLogger
from utils.logging.types import StepResult
from utils.strategies.backpropagation_strategies import (
    ActivatedSurpriseMinKStrategy,
    SurpriseMinKStrategy,
)
from utils.trainer import Trainer


def format_buffer_dist_bar(
    num_items: int, capacity: int, num_prototypes: int, bar_width: int = 20
) -> str:
    if capacity == 0:
        return "Buffer: [ " + " " * bar_width + " ] 0/0 (0.0%) | Prototypes: 0"

    if capacity == 0:
        return "Buffer: [ " + " " * bar_width + " ] 0/0 (0.0%) | Prototypes: 0"

    occupancy_ratio = min(1.0, num_items / capacity)
    occupied_width = int(bar_width * occupancy_ratio)
    
    bar_char = '█'
    bar_str = f"[{bar_char * occupied_width}{' ' * (bar_width - occupied_width)}]"
    
    occupancy_str = f"{num_items}/{capacity} ({occupancy_ratio:.1%})"
    prototypes_str = f"Prototypes: {num_prototypes}"
    
    return f"Buffer: {bar_str} {occupancy_str} | {prototypes_str}"


class PILFTrainLoop:
    def __init__(
        self,
        trainer: Trainer,
        experience_buffer: PrototypingExperienceBuffer,
        pilf_config: Dict[str, Any],
    ):
        self.trainer = trainer
        self.model = trainer.model
        self.optimizer = trainer.optimizer
        self.gating_optimizer = trainer.gating_optimizer
        self.loss_fn = trainer.loss_fn
        self.device = trainer.device
        self.writer = trainer.writer
        self.pi_calculator = trainer.pi_calculator
        self.experience_buffer = experience_buffer
        self.pilf_config = pilf_config
        self.smk_strategy = next(
            (
                s
                for s in trainer.strategy_components
                if isinstance(s, (SurpriseMinKStrategy, ActivatedSurpriseMinKStrategy))
            ),
            None,
        )
        self.meta_learning_freq = pilf_config.get("meta_learning_freq", 1)
        self.gating_loss_cache: Dict[Tuple[str, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def train_one_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        epoch: int,
        task_name: str,
    ) -> None:
        self.model.train()
        
        status_bar = tqdm(total=0, position=0, bar_format='{desc}')
        
        train_loader_tqdm = tqdm(
            train_loader, desc=f"Epoch {epoch} ({task_name})", leave=False, position=1
        )

        for i, (data, target) in enumerate(train_loader_tqdm):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            logits, all_routing_info = self.model(data, self.experience_buffer)
            expert_loss = self.loss_fn(logits, target)
            expert_loss.backward()

            all_grads = [p.grad for n, p in self.model.named_parameters() if p.grad is not None and "gating_transformer" not in n]
            pi_metrics = self.pi_calculator.calculate(loss=expert_loss, logits=logits, gradients=all_grads)

            smk_metrics: StepResult = {}
            top_k_indices_by_layer: Dict[int, List[int]] = {}
            if all_routing_info:
                top_k_indices_by_layer = {i: info["top_indices"].flatten().detach().cpu().tolist() for i, info in enumerate(all_routing_info)}

            if self.smk_strategy and all_routing_info:
                smk_metrics = self.smk_strategy.apply(self.model, self.optimizer, {}, activated_experts=top_k_indices_by_layer)
            
            self.optimizer.step()

            with torch.no_grad():
                if all_routing_info and smk_metrics and "surprise_min_k_expert_indices" in smk_metrics:
                    pi_score_tensor = pi_metrics.get("pi_score", torch.zeros(1, device=self.device))
                    min_k_indices_by_layer = smk_metrics["surprise_min_k_expert_indices"]
                    if min_k_indices_by_layer is not None:
                        for layer_idx, layer_info in enumerate(all_routing_info):
                            if layer_idx in min_k_indices_by_layer:
                                batch_size = data.shape[0]
                                
                                q_aggregated = layer_info["q_embedding"]
                                
                                min_k_indices = min_k_indices_by_layer[layer_idx]
                                if not min_k_indices: continue

                                num_total_experts = sum(b.mlp.num_experts for b in self.model.blocks if hasattr(b, "mlp") and "BaseMoELayer" in [c.__name__ for c in b.mlp.__class__.__mro__])
                                target_actions_multihot = torch.zeros(batch_size, num_total_experts, device=self.device)
                                
                                offset = 0
                                for b_idx, block in enumerate(self.model.blocks):
                                    if hasattr(block, "mlp") and "BaseMoELayer" in [c.__name__ for c in block.mlp.__class__.__mro__]:
                                        if layer_idx == b_idx and min_k_indices:
                                            min_k_tensor = torch.tensor(min_k_indices, device=self.device)
                                            global_indices = min_k_tensor + offset
                                            sample_indices = torch.arange(batch_size, device=self.device).unsqueeze(1)
                                            target_actions_multihot[sample_indices, global_indices] = 1.0
                                        offset += block.mlp.num_experts


                                self.experience_buffer.add(q_aggregated, target_actions_multihot, pi_score_tensor.expand(batch_size))

            buffer_dist_bar = format_buffer_dist_bar(
                self.experience_buffer.current_size,
                self.experience_buffer.capacity,
                self.experience_buffer.num_active_prototypes,
            )
            status_bar.set_description_str(buffer_dist_bar)

            gating_loss = torch.tensor(0.0, device=self.device)
            if self.gating_optimizer and self.experience_buffer.current_size > 0 and (i + 1) % self.meta_learning_freq == 0:
                self.gating_optimizer.zero_grad()
                
                cache_key = (task_name, epoch)
                if cache_key in self.gating_loss_cache:
                    q_rehearsal, a_rehearsal, sampled_indices = self.gating_loss_cache[cache_key]
                else:
                    q_rehearsal, a_rehearsal, sampled_indices = self.experience_buffer.sample(self.experience_buffer.current_size)
                    self.gating_loss_cache[cache_key] = (q_rehearsal, a_rehearsal, sampled_indices)

                moe_blocks = [
                    b
                    for b in self.model.blocks
                    if hasattr(b, "mlp")
                    and "MemoryGaussianMoELayer" in [c.__name__ for c in b.mlp.__class__.__mro__]
                ]
                if moe_blocks:
                    total_gating_loss = torch.tensor(0.0, device=self.device)
                    num_gating_layers = 0
                    action_offset = 0
                    for block in moe_blocks:
                        mlp_layer = cast("MemoryGaussianMoELayer", block.mlp)
                        num_experts_in_layer = mlp_layer.num_experts
                        
                        actions_for_layer = a_rehearsal[:, action_offset : action_offset + num_experts_in_layer]
                        
                        gating_loss_for_layer = mlp_layer.meta_train(q_rehearsal, actions_for_layer, self.experience_buffer)
                        if torch.isfinite(gating_loss_for_layer):
                            total_gating_loss += gating_loss_for_layer
                            num_gating_layers += 1
                        
                        action_offset += num_experts_in_layer
                    
                    if num_gating_layers > 0:
                        gating_loss = total_gating_loss / num_gating_layers
                        if torch.isfinite(gating_loss):
                            gating_loss.backward()
                            self.gating_optimizer.step()
                            self.experience_buffer.decay_priorities(sampled_indices)

            pred = logits.argmax(dim=1, keepdim=True)
            accuracy = 100.0 * pred.eq(target.view_as(pred)).sum().item() / len(data)

            routing_accuracy = 0.0
            if all_routing_info and smk_metrics and "surprise_min_k_expert_indices" in smk_metrics:
                min_k_indices_all_layers = smk_metrics["surprise_min_k_expert_indices"]
                if min_k_indices_all_layers is not None:
                    total_intersection = 0
                    total_top_k = 0

                    for layer_idx, layer_info in enumerate(all_routing_info):
                        if layer_idx in min_k_indices_all_layers and "top_indices" in layer_info:
                            top_indices_per_token = layer_info["top_indices"]
                            min_k_indices_for_layer = min_k_indices_all_layers.get(layer_idx)
                            if min_k_indices_for_layer:
                                min_k_set_for_layer = set(min_k_indices_for_layer)
                                min_k_tensor = torch.tensor(list(min_k_set_for_layer), device=top_indices_per_token.device)
                                intersection = torch.isin(top_indices_per_token, min_k_tensor).sum()
                                total_intersection += intersection.item()
                            total_top_k += top_indices_per_token.numel()
                    
                    if total_top_k > 0:
                        routing_accuracy = 100.0 * total_intersection / total_top_k

            step_result: StepResult = {
                "global_step": self.trainer.global_step, "loss": expert_loss.item(),
                "gating_loss": gating_loss.item(), "accuracy": accuracy, 
                "routing_accuracy": routing_accuracy, "task_name": task_name,
            }
            
            if top_k_indices_by_layer:
                step_result["top_k_expert_indices"] = top_k_indices_by_layer

            for key, value in pi_metrics.items():
                if key in StepResult.__annotations__:
                    step_result[key] = value.item() #type: ignore
            step_result.update(smk_metrics)

            self.trainer.epoch_results.append(step_result)
            logger = TensorBoardLogger(self.writer, self.trainer.global_step)
            logger.log_metrics(cast(Dict[str, Any], step_result), task_name, "Train")

            train_loader_tqdm.set_postfix(loss=f"{expert_loss.item():.4f}", acc=f"{accuracy:.2f}%")
            self.trainer.global_step += 1
        
        status_bar.close()
