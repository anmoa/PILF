from typing import Any, Dict, List, cast

import torch
from tqdm import tqdm

from models.moe_layers import MemoryGaussianMoELayer
from utils.experience_buffer import MultiTaskExperienceBuffer
from utils.logging.metrics_logger import TensorBoardLogger
from utils.logging.types import StepResult
from utils.strategies.backpropagation_strategies import SurpriseMinKStrategy
from utils.trainer import Trainer


def format_buffer_dist_bar(task_indices: Dict[str, list], total_size: int, bar_width: int = 20) -> str:
    dist_str = ""
    total_items = sum(len(v) for v in task_indices.values())
    
    colors = ['\x1b[47m', '\x1b[100m'] # White, Bright Black
    
    for i, (task, indices) in enumerate(task_indices.items()):
        
        task_len = len(indices)
        if total_items == 0:
            proportion = 0.0
        else:
            proportion = task_len / total_items
        
        bar_segment_width = int(proportion * bar_width)
        
        color = colors[i % len(colors)]
        reset_color = '\x1b[0m'
        
        dist_str += f" {task}: {task_len} {color}{' ' * bar_segment_width}{reset_color}"

    return f"Buffer: [{dist_str.strip()}]"


class PILFTrainLoop:
    def __init__(
        self,
        trainer: Trainer,
        experience_buffer: MultiTaskExperienceBuffer,
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
            (s for s in trainer.strategy_components if isinstance(s, SurpriseMinKStrategy)),
            None,
        )
        self.meta_learning_freq = pilf_config.get("meta_learning_freq", 5)

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
                smk_metrics = self.smk_strategy.apply(self.model, self.optimizer, {})
            
            self.optimizer.step()

            with torch.no_grad():
                priority = pi_metrics.get("pi_score", torch.tensor(0.0))
                
                if all_routing_info and "surprise_min_k_expert_indices" in smk_metrics:
                    min_k_indices_by_layer = smk_metrics["surprise_min_k_expert_indices"]
                    if min_k_indices_by_layer:
                        for layer_idx, layer_info in enumerate(all_routing_info):
                            if layer_idx in min_k_indices_by_layer:
                                q_embeddings = layer_info["q_embedding"]
                                batch_size = data.shape[0]
                                num_tokens = q_embeddings.shape[0] // batch_size
                                embed_dim = q_embeddings.shape[1]
                                q_reshaped = q_embeddings.view(batch_size, num_tokens, embed_dim)
                                q_aggregated = q_reshaped.mean(dim=1)
                                actions = torch.tensor(min_k_indices_by_layer[layer_idx], device=self.device, dtype=torch.long)
                                
                                if actions.numel() > 0:
                                    q_to_store = q_aggregated.repeat_interleave(actions.numel(), dim=0)
                                    actions_to_store = actions.repeat(q_aggregated.size(0))
                                    # The priority for replacement is the INVERSE of PI score. Low PI = high priority.
                                    priorities_to_store = (1.0 - priority).expand_as(actions_to_store).float()
                                    self.experience_buffer.add(q_to_store, actions_to_store, priorities_to_store, task_name)

            gating_loss = torch.tensor(0.0, device=self.device)
            if self.gating_optimizer and self.experience_buffer.current_size > 0 and (i + 1) % self.meta_learning_freq == 0:
                self.gating_optimizer.zero_grad()
                q_rehearsal, a_rehearsal, indices_by_task = self.experience_buffer.sample(self.experience_buffer.current_size)
                
                buffer_dist_bar = format_buffer_dist_bar(self.experience_buffer.task_indices, self.experience_buffer.total_buffer_size)
                status_bar.set_description_str(buffer_dist_bar)

                total_gating_loss = torch.tensor(0.0, device=self.device)
                num_gating_layers = 0
                
                moe_blocks = [b for b in self.model.blocks if isinstance(b.mlp, MemoryGaussianMoELayer)]
                if not moe_blocks: continue

                td_errors_by_task: Dict[str, List[torch.Tensor]] = {name: [] for name in self.experience_buffer.task_names}

                for block in moe_blocks:
                    mlp_layer = cast(MemoryGaussianMoELayer, block.mlp)
                    gating_loss_for_layer, td_error = mlp_layer.meta_train(q_rehearsal, a_rehearsal)
                    
                    if torch.isfinite(gating_loss_for_layer):
                        total_gating_loss += gating_loss_for_layer
                        num_gating_layers += 1
                        
                        start_idx = 0
                        for task, indices in indices_by_task.items():
                            end_idx = start_idx + len(indices)
                            td_errors_by_task[task].append(td_error[start_idx:end_idx])
                            start_idx = end_idx
                
                avg_td_errors = {name: torch.stack(errors).mean(0) for name, errors in td_errors_by_task.items() if errors}
                self.experience_buffer.update_priorities(indices_by_task, avg_td_errors)

                if num_gating_layers > 0:
                    gating_loss = total_gating_loss / num_gating_layers
                    if torch.isfinite(gating_loss):
                        gating_loss.backward()
                        self.gating_optimizer.step()

            pred = logits.argmax(dim=1, keepdim=True)
            accuracy = 100.0 * pred.eq(target.view_as(pred)).sum().item() / len(data)

            routing_accuracy = 0.0
            if top_k_indices_by_layer and "surprise_min_k_expert_indices" in smk_metrics:
                min_k_indices = smk_metrics["surprise_min_k_expert_indices"]
                total_intersection = 0
                total_top_k = 0
                
                for layer_idx, top_k_ids in top_k_indices_by_layer.items():
                    if min_k_indices and layer_idx in min_k_indices:
                        top_k_set = set(top_k_ids)
                        min_k_set = set(min_k_indices[layer_idx])
                        total_intersection += len(top_k_set.intersection(min_k_set))
                        total_top_k += len(top_k_set)
                
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