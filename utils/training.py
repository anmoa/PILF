import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader
import os
from sigma_pi import SigmaPI 
from utils.plotting import plot_metrics
from typing import Dict, List, Tuple, Any, Optional, Sized

def gaussian_modulation(surprise: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * ((surprise - mu) / (sigma + 1e-9)) ** 2)

def train(model: nn.Module, device: torch.device, train_loader: DataLoader, optimizer: optim.Optimizer, epoch: int, loss_fn: nn.Module, pi_monitor: SigmaPI, 
          step_metrics: Dict[str, List[Tuple[int, float]]], epoch_metrics: Dict[str, List[Tuple[int, float]]], global_step: int, accumulation_steps: int, 
          use_pilr: bool = False, initial_surprise_ema: Optional[float] = None, sigma_threshold: float = 1.0, pilr_mode: str = 'lr_scheduler') -> Tuple[int, List[Tuple[int, float]], List[Tuple[int, str]]] | int:
    model.train()
    optimizer.zero_grad()

    for key in ['train_loss', 'train_acc', 'train_pi', 'train_surprise', 'train_tau']:
        step_metrics.setdefault(key, [])

    ema_surprise: Optional[torch.Tensor] = None
    ema_surprise_sq: Optional[torch.Tensor] = None
    if initial_surprise_ema is not None:
        ema_surprise = torch.tensor(initial_surprise_ema, device=device)
        ema_surprise_sq = torch.tensor(initial_surprise_ema ** 2, device=device)
    
    ema_alpha = 0.1
    base_lr = optimizer.param_groups[0]['lr']

    epoch_summary: Dict[str, List[Any]] = {
        'loss': [], 'acc': [], 'pi': [], 'surprise': [], 'tau': [],
        'consolidate': [], 'ignore': [], 'reject': [],
        'surprise_values': [], 'decisions': [], 'lr_mod': []
    }

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        output = model(data)
        logits = output[0] if isinstance(output, tuple) else output
        all_gating_logits = output[1] if isinstance(output, tuple) else None

        loss = loss_fn(logits, target)
        loss_normalized = loss / accumulation_steps
        loss_normalized.backward()

        pred = logits.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        accuracy = 100. * correct / len(data)

        if (batch_idx + 1) % accumulation_steps == 0:
            if use_pilr:
                if all_gating_logits is None:
                    raise ValueError("PILR requires a model that returns gating logits.")
                model.zero_inactive_expert_grads(all_gating_logits)
            
            pi_metrics = pi_monitor.calculate(model, loss, logits)
            
            if use_pilr:
                surprise = torch.tensor(pi_metrics['surprise'], device=device)
                if ema_surprise is None:
                    ema_surprise = surprise
                    ema_surprise_sq = surprise ** 2
                else:
                    assert ema_surprise is not None, "ema_surprise must be a Tensor here"
                    assert ema_surprise_sq is not None, "ema_surprise_sq must be a Tensor here"
                    ema_surprise = ema_alpha * surprise + (1 - ema_alpha) * ema_surprise
                    ema_surprise_sq = ema_alpha * (surprise ** 2) + (1 - ema_alpha) * ema_surprise_sq

                variance = ema_surprise_sq - ema_surprise ** 2
                std_dev = torch.sqrt(torch.clamp(variance, min=0))

                if pilr_mode == 'gate':
                    s_low = ema_surprise - sigma_threshold * std_dev
                    s_high = ema_surprise + sigma_threshold * std_dev
                    decision = "IGNORE"
                    if s_low <= surprise < s_high:
                        decision = "CONSOLIDATE"
                        optimizer.step()
                    elif surprise >= s_high:
                        decision = "REJECT"
                    epoch_summary[decision.lower()].append(1)
                    epoch_summary['surprise_values'].append((global_step, surprise.item()))
                    epoch_summary['decisions'].append((global_step, decision))
                
                elif pilr_mode == 'lr_scheduler':
                    lr_modulation = gaussian_modulation(surprise, ema_surprise, sigma_threshold * std_dev)
                    effective_lr = base_lr * lr_modulation
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = effective_lr
                    epoch_summary['lr_mod'].append(lr_modulation.item())
                    optimizer.step()
                else:
                    raise ValueError(f"Unknown PILR mode: {pilr_mode}")

            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            optimizer.zero_grad()

            if use_pilr and pilr_mode == 'lr_scheduler':
                for param_group in optimizer.param_groups:
                    param_group['lr'] = base_lr

            step_metrics['train_loss'].append((global_step, loss.item()))
            step_metrics['train_acc'].append((global_step, accuracy))
            step_metrics['train_pi'].append((global_step, pi_metrics['pi_score']))
            step_metrics['train_surprise'].append((global_step, pi_metrics['surprise']))
            step_metrics['train_tau'].append((global_step, pi_metrics['tau']))
            
            epoch_summary['loss'].append(loss.item())
            epoch_summary['acc'].append(accuracy)
            epoch_summary['pi'].append(pi_metrics['pi_score'])
            epoch_summary['surprise'].append(pi_metrics['surprise'])
            epoch_summary['tau'].append(pi_metrics['tau'])

            global_step += 1
        
    avg_metrics = {key: sum(vals) / len(vals) if vals else 0 for key, vals in epoch_summary.items() if key not in ['surprise_values', 'decisions']}
    
    for key, avg_val in avg_metrics.items():
        metric_name = f'train_{key}'
        if metric_name in epoch_metrics:
            epoch_metrics[metric_name].append((global_step - 1, avg_val))

    summary_str = f"Train Epoch {epoch} Summary: Avg loss: {avg_metrics['loss']:.4f}, Avg Accuracy: {avg_metrics['acc']:.2f}%, Avg PI: {avg_metrics['pi']:.4f}, Avg Surprise: {avg_metrics['surprise']:.4f}, Avg Tau: {avg_metrics['tau']:.4f}"
    if use_pilr and pilr_mode == 'lr_scheduler':
        summary_str += f", Avg LR-Mod: {avg_metrics['lr_mod']:.4f}"
    print(summary_str)
    
    if use_pilr and pilr_mode == 'gate':
        consolidate_count = len(epoch_summary['consolidate'])
        ignore_count = len(epoch_summary['ignore'])
        reject_count = len(epoch_summary['reject'])
        total_decisions = consolidate_count + ignore_count + reject_count
        if total_decisions > 0:
            print(f"Decision Stats: Consolidate: {100*consolidate_count/total_decisions:.1f}%, Ignore: {100*ignore_count/total_decisions:.1f}%, Reject: {100*reject_count/total_decisions:.1f}%")
        return global_step, epoch_summary['surprise_values'], epoch_summary['decisions']
    
    return global_step

def validate(model: nn.Module, device: torch.device, val_loader: DataLoader, loss_fn: nn.Module, pi_monitor: SigmaPI, dataset_name: str = "Validation") -> Tuple[float, float, float, float, float]:
    model.eval()
    total_loss, correct = 0.0, 0
    all_pi_scores: List[float] = []
    all_surprises: List[float] = []
    all_taus: List[float] = []

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

        with torch.no_grad():
            total_loss += loss_epsilon.item()
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = total_loss / len(val_loader)
    
    dataset = val_loader.dataset
    if not isinstance(dataset, Sized):
        raise TypeError("Dataset is not Sized, cannot calculate accuracy.")
    
    accuracy = 100. * correct / len(dataset)
    avg_pi = sum(all_pi_scores) / len(all_pi_scores) if all_pi_scores else 0.0
    avg_surprise = sum(all_surprises) / len(all_surprises) if all_surprises else 0.0
    avg_tau = sum(all_taus) / len(all_taus) if all_taus else 0.0

    print(f"{dataset_name} set: Avg loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, Avg PI: {avg_pi:.4f}, Avg Surprise: {avg_surprise:.4f}, Avg Tau: {avg_tau:.4f}")
    return avg_loss, accuracy, avg_pi, avg_surprise, avg_tau
