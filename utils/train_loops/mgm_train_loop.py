from typing import Any, Dict, List, Tuple, cast

import torch
from tqdm import tqdm

from models.moe_layers import BaseMoELayer, MemoryGaussianMoELayer
from utils.gating_loss import CompositeGatingLoss
from utils.logging.metrics_logger import TensorBoardLogger
from utils.logging.types import StepResult
from utils.strategies.backpropagation_strategies import SurpriseMinKStrategy
from utils.train_loops.base_train_loop import BaseTrainLoop


class MGMTrainLoop(BaseTrainLoop):
    def __init__(self, trainer):
        super().__init__(trainer)
        self.composite_gating_loss_fn = CompositeGatingLoss(
            rehearsal_weight=self.trainer.pi_calculator.gamma
        )
        self.mgm_layers = [
            block.mlp
            for block in self.model.blocks
            if isinstance(block.mlp, MemoryGaussianMoELayer)
        ]
        
        smk_strategy = next(
            (s for s in self.strategy_components if isinstance(s, SurpriseMinKStrategy)),
            None,
        )
        self.min_k_frac = 0.5
        if smk_strategy and hasattr(smk_strategy, 'min_k_frac'):
             self.min_k_frac = smk_strategy.min_k_frac


    def _calculate_min_k_indices(self) -> Dict[int, List[int]]:
        min_k_expert_indices: Dict[int, List[int]] = {}
        moe_blocks = [
            block
            for block in self.model.blocks
            if hasattr(block, "mlp") and isinstance(block.mlp, BaseMoELayer)
        ]

        for i, block in enumerate(moe_blocks):
            layer_surprises = []
            current_moe_layer = block.mlp
            num_experts = current_moe_layer.num_experts
            k = max(1, int(self.min_k_frac * num_experts))

            for expert_idx, expert_module in enumerate(current_moe_layer.experts):
                expert_grads = [
                    p.grad for p in expert_module.parameters() if p.grad is not None
                ]
                if not expert_grads:
                    surprise = float("inf")
                else:
                    total_norm_sq = torch.stack(
                        [torch.sum(g**2) for g in expert_grads]
                    ).sum()
                    surprise = torch.sqrt(total_norm_sq).item()
                layer_surprises.append((surprise, expert_idx))

            layer_surprises.sort(key=lambda x: x[0])
            min_k_indices_for_layer = [idx for _, idx in layer_surprises[:k]]
            min_k_expert_indices[i] = min_k_indices_for_layer

        return min_k_expert_indices

    def train_one_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        epoch: int,
        global_step: int,
        accumulation_steps: int,
        task_name: str,
    ) -> Tuple[int, List[StepResult]]:
        self.model.train()
        epoch_results: List[StepResult] = []

        for layer in self.mgm_layers:
            layer.decay_memory()

        train_loader_tqdm = tqdm(
            train_loader, desc=f"Epoch {epoch} ({task_name})", leave=False
        )
        for _, (data, target) in enumerate(train_loader_tqdm):
            self.optimizer.zero_grad()
            data, target = data.to(self.device), target.to(self.device)

            output = self.model(data)
            logits, all_routing_info = output
            expert_loss = self.loss_fn(logits, target)

            expert_loss.backward(retain_graph=True)

            with torch.no_grad():
                expert_grads = [
                    p.grad.clone()
                    for pg in self.optimizer.param_groups
                    if pg["name"] == "experts"
                    for p in pg["params"]
                    if p.grad is not None
                ]
                pi_metrics = self.pi_calculator.calculate(
                    loss=expert_loss, logits=logits, gradients=expert_grads
                )

            min_k_expert_indices_per_layer = self._calculate_min_k_indices()

            pred = logits.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy = 100.0 * correct / len(data)

            with torch.no_grad():
                if all_routing_info:
                    for i, routing_info in enumerate(all_routing_info):
                        if (
                            i < len(self.mgm_layers)
                            and i in min_k_expert_indices_per_layer
                        ):
                            q_embedding = routing_info.get("q_embedding")
                            if q_embedding is not None:
                                min_k_indices = torch.tensor(
                                    min_k_expert_indices_per_layer[i],
                                    device=self.device,
                                    dtype=torch.long,
                                )
                                self.mgm_layers[i].update_memory(
                                    pi_metrics["tau"],
                                    q_embedding.mean(dim=0),
                                    min_k_indices,
                                )

            gating_loss = self.composite_gating_loss_fn(
                all_routing_info, min_k_expert_indices_per_layer
            )

            if gating_loss > 0:
                gating_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            top_k_indices_dict = (
                {
                    i: info["top_indices"].detach().cpu().numpy()
                    for i, info in enumerate(all_routing_info)
                }
                if all_routing_info
                else {}
            )

            step_result: StepResult = {
                "global_step": global_step,
                "loss": expert_loss.item(),
                "gating_loss": gating_loss.item(),
                "accuracy": accuracy,
                "task_name": task_name,
                "top_k_expert_indices": top_k_indices_dict,
                "surprise_min_k_expert_indices": min_k_expert_indices_per_layer,
            }
            step_result["pi_score"] = pi_metrics["pi_score"].item()
            step_result["surprise"] = pi_metrics["surprise"].item()
            step_result["tau"] = pi_metrics["tau"].item()

            epoch_results.append(step_result)
            global_step += 1

            logger = TensorBoardLogger(self.writer, global_step)
            logger.log_metrics(cast(Dict[str, Any], step_result), task_name, scope="Train")

            train_loader_tqdm.set_postfix(
                loss=f"{expert_loss.item():.4f}",
                g_loss=f"{gating_loss.item():.4f}",
                acc=f"{accuracy:.2f}%",
                pi=f"{pi_metrics.get('pi_score', 0.0):.4f}",
            )

        return global_step, epoch_results