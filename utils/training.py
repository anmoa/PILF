from typing import Any, List, Optional, Sized, Tuple, cast

import torch
import torch.nn as nn
import torch.optim as optim
from sigma_pi import SigmaPI
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.strategies import StrategyComponent
from utils.types import StepResult, ValidationResult


def train(
    model: nn.Module, 
    device: torch.device, 
    train_loader: DataLoader, 
    optimizer: optim.Optimizer, 
    epoch: int, 
    loss_fn: nn.Module, 
    pi_monitor: SigmaPI, 
    strategy_components: List[StrategyComponent],
    writer: "SummaryWriter",
    global_step: int, 
    accumulation_steps: int,
    task_name: str
) -> Tuple[int, List[StepResult]]:
    model.train()
    optimizer.zero_grad()

    epoch_results: List[StepResult] = []

    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch} ({task_name})", leave=False)
    for batch_idx, (data, target) in enumerate(train_loader_tqdm):
        data, target = data.to(device), target.to(device)

        output = model(data)
        
        all_top_indices, all_log_probs, all_gating_logits = None, None, None
        if isinstance(output, tuple) and len(output) == 3:
            logits, all_top_indices, all_log_probs = output
            all_gating_logits = all_log_probs
        elif isinstance(output, tuple):
            logits, all_gating_logits = output
        else:
            logits = output

        loss = loss_fn(logits, target)
        loss_normalized = loss / accumulation_steps
        loss_normalized.backward()

        pred = logits.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        accuracy = 100. * correct / len(data)

        if (batch_idx + 1) % accumulation_steps == 0:
            pi_metrics = pi_monitor.calculate(model, loss, logits)
            
            step_result: StepResult = {
                "global_step": global_step,
                "loss": loss.item(),
                "accuracy": accuracy,
                "task_name": task_name,
                **pi_metrics,
            }

            # Apply strategy components in order
            for component in strategy_components:
                component_metrics = component.apply(model, optimizer, pi_metrics, all_gating_logits, all_top_indices)
                step_result.update(component_metrics)
            
            optimizer.zero_grad() # Moved here to be after all strategy applications

            if 'pi_score' in step_result: # Check in step_result as it's updated by strategies
                pi_metrics['pi_score'] = cast(float, step_result['pi_score'])
            if 'surprise' in step_result:
                pi_metrics['surprise'] = cast(float, step_result['surprise'])

            epoch_results.append(step_result)
            global_step += 1

            # --- Log metrics to TensorBoard at each step ---
            for key, value in step_result.items():
                if key in ['cognitive_cost', 'epsilon']: # Skip logging cognitive_cost and epsilon
                    continue
                if isinstance(value, (int, float)):
                    tag = f'Predictive Integrity/train/{task_name}' if key == 'pi_score' else f'{key.replace("_", " ").title()}/train/{task_name}'
                    writer.add_scalar(tag, value, global_step)
            writer.flush()

            train_loader_tqdm.set_postfix(
                loss=f"{step_result.get('loss', 0.0):.4f}",
                acc=f"{step_result.get('accuracy', 0.0):.2f}%",
                pi=f"{step_result.get('pi_score', 0.0):.4f}",
                surprise=f"{step_result.get('surprise', 0.0):.4f}",
            )
    
    if epoch_results:
        def safe_avg(key: str) -> Optional[float]:
            values = [r.get(cast(Any, key)) for r in epoch_results if r.get(cast(Any, key)) is not None]
            if not values:
                return None
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            return sum(numeric_values) / len(numeric_values) if numeric_values else None

        summary_parts = [
            f"Train Epoch {epoch} Summary:",
            f"Avg loss: {safe_avg('loss'):.4f}",
            f"Avg Accuracy: {safe_avg('accuracy'):.2f}%",
            f"Avg PI: {safe_avg('pi_score'):.4f}",
            f"Avg Surprise: {safe_avg('surprise'):.4f}",
        ]
        print(", ".join(filter(None, summary_parts)))

    return global_step, epoch_results

def validate(model: nn.Module, device: torch.device, val_loader: DataLoader, loss_fn: nn.Module, pi_monitor: SigmaPI, dataset_name: str = "Validation") -> ValidationResult:
    model.eval()
    total_loss, correct = 0.0, 0
    all_pi_scores: List[float] = []
    all_surprises: List[float] = []
    all_taus: List[float] = []
    all_gating_taus: List[float] = []

    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        
        with torch.enable_grad():
            output = model(data)
            logits = output[0] if isinstance(output, tuple) else output
            loss_epsilon = loss_fn(logits, target)
            
            model.zero_grad()
            loss_epsilon.backward()
            
            pi_metrics = pi_monitor.calculate(model, loss_epsilon, logits)
            all_pi_scores.append(pi_metrics['pi_score'])
            all_surprises.append(pi_metrics['surprise'])
            all_taus.append(pi_metrics['tau'])
            if 'gating_tau' in pi_metrics:
                all_gating_taus.append(pi_metrics['gating_tau'])

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
    
    print(", ".join(filter(None, summary_parts)))
    return avg_loss, accuracy, avg_pi, avg_surprise, avg_tau, avg_gating_tau
