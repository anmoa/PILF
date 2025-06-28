import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader
import os
from sigma_pi import SigmaPI
from utils.plotting import plot_metrics
from utils.strategies import UpdateStrategy
from typing import Dict, List, Tuple, Any, Optional, Sized

def train(model: nn.Module, device: torch.device, train_loader: DataLoader, optimizer: optim.Optimizer, epoch: int, loss_fn: nn.Module, pi_monitor: SigmaPI, 
          update_strategy: UpdateStrategy, step_metrics: Dict[str, List[Tuple[int, float]]], epoch_metrics: Dict[str, List[Tuple[int, float]]], 
          global_step: int, accumulation_steps: int) -> Tuple[int, Dict[str, List[Any]]]:
    model.train()
    optimizer.zero_grad()

    for key in ['train_loss', 'train_acc', 'train_pi_score', 'train_surprise', 'train_tau', 'train_gating_tau']:
        step_metrics.setdefault(key, [])

    epoch_summary: Dict[str, List[Any]] = {
        'loss': [], 'acc': [], 'pi': [], 'surprise': [], 'tau': [], 'gating_tau': [],
        'surprise_values': [], 'decisions': [], 
        'lr_mod': [], 'sigma': [],
        'gating_lr_mod': [], 'gating_sigma': [],
        'expert_lr_mod': [], 'expert_sigma': [],
        'consolidate': [], 'ignore': [], 'reject': [], # Add decision tracking
        'all_top_indices': [], 'all_log_probs': [],
    }

    all_top_indices = None
    all_log_probs = None

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        output = model(data)
        
        # Unpack model output
        if isinstance(output, tuple) and len(output) == 3: # GPIL-MoE case
            logits, all_top_indices, all_log_probs = output
            all_gating_logits = all_log_probs # For compatibility with PISA update
        elif isinstance(output, tuple): # Other MoE cases
            logits, all_gating_logits = output
        else: # Base ViT case
            logits = output
            all_gating_logits = None

        loss = loss_fn(logits, target)
        loss_normalized = loss / accumulation_steps
        loss_normalized.backward()

        pred = logits.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        accuracy = 100. * correct / len(data)

        if (batch_idx + 1) % accumulation_steps == 0:
            pi_metrics = pi_monitor.calculate(model, loss, logits)
            
            # For GPIL-MoE, we also need to pass the top_indices for sparse gradient updates
            if isinstance(model, nn.DataParallel):
                is_gpil_moe = hasattr(model.module, 'zero_inactive_expert_grads')
            else:
                is_gpil_moe = hasattr(model, 'zero_inactive_expert_grads')

            if is_gpil_moe:
                pilr_metrics = update_strategy.step(model, loss, pi_metrics, all_gating_logits)
                if isinstance(model, nn.DataParallel):
                    model.module.zero_inactive_expert_grads(all_top_indices)
                else:
                    model.zero_inactive_expert_grads(all_top_indices)
            else:
                pilr_metrics = update_strategy.step(model, loss, pi_metrics, all_gating_logits)

            # In dual PISA mode, the strategy calculates its own global PI metrics.
            # We should use those for logging instead of the ones from the global monitor.
            if 'pi_score' in pilr_metrics:
                pi_metrics['pi_score'] = pilr_metrics['pi_score']
            if 'surprise' in pilr_metrics:
                pi_metrics['surprise'] = pilr_metrics['surprise']
            
            optimizer.zero_grad()

            # Log metrics
            step_metrics['train_loss'].append((global_step, loss.item()))
            step_metrics['train_acc'].append((global_step, accuracy))
            for key in ['pi_score', 'surprise', 'tau', 'gating_tau']:
                if key in pi_metrics:
                    step_metrics[f'train_{key}'].append((global_step, pi_metrics[key]))
            
            epoch_summary['loss'].append(loss.item())
            epoch_summary['acc'].append(accuracy)
            for key in ['pi_score', 'surprise', 'tau', 'gating_tau']:
                if key in pi_metrics:
                    epoch_summary[key.replace('_score', '')].append(pi_metrics[key])

            if pilr_metrics:
                for key, value in pilr_metrics.items():
                    if key in epoch_summary:
                        epoch_summary[key].append(value)
                # For backward compatibility with single-PISA surprise logging
                if 'surprise_value' in pilr_metrics and 'surprise_values' in epoch_summary:
                     epoch_summary['surprise_values'].append((global_step, pilr_metrics['surprise_value']))
                
                # Log decision for plotting
                if 'decision' in pilr_metrics:
                    epoch_summary['surprise_values'].append((global_step, pi_metrics['surprise']))
                    epoch_summary['decisions'].append((global_step, pilr_metrics['decision']))
                    epoch_summary[pilr_metrics['decision'].lower()].append(1)


            global_step += 1
        
    if all_top_indices is not None:
        epoch_summary['all_top_indices'].append(all_top_indices)
    if all_log_probs is not None:
        epoch_summary['all_log_probs'].append(all_log_probs)

    avg_metrics = {key: sum(vals) / len(vals) if vals else 0 for key, vals in epoch_summary.items() if isinstance(vals, list) and key not in ['surprise_values', 'decisions', 'all_top_indices', 'all_log_probs']}
    
    for key, avg_val in avg_metrics.items():
        metric_name = f'train_{key}'
        if metric_name in epoch_metrics:
            epoch_metrics[metric_name].append((global_step - 1, avg_val))

    summary_str = f"Train Epoch {epoch} Summary: Avg loss: {avg_metrics.get('loss', 0):.4f}, Avg Accuracy: {avg_metrics.get('acc', 0):.2f}%, Avg PI: {avg_metrics.get('pi', 0):.4f}, Avg Surprise: {avg_metrics.get('surprise', 0):.4f}, Avg Tau: {avg_metrics.get('tau', 0):.4f}"
    if avg_metrics.get('gating_tau'):
        summary_str += f", Avg Gating Tau: {avg_metrics['gating_tau']:.4f}"
    if avg_metrics.get('lr_mod'):
        summary_str += f", Avg LR-Mod: {avg_metrics['lr_mod']:.4f}"
    if avg_metrics.get('gating_lr_mod'):
        summary_str += f", Avg Gating LR-Mod: {avg_metrics['gating_lr_mod']:.4f}"
    if avg_metrics.get('expert_lr_mod'):
        summary_str += f", Avg Expert LR-Mod: {avg_metrics['expert_lr_mod']:.4f}"
    print(summary_str)
    
    # Add decision stats to summary string
    if hasattr(update_strategy, 'crisis_threshold') and update_strategy.crisis_threshold is not None:
        consolidate_count = len(epoch_summary['consolidate'])
        ignore_count = len(epoch_summary['ignore'])
        reject_count = len(epoch_summary['reject'])
        total_decisions = consolidate_count + ignore_count + reject_count
        if total_decisions > 0:
            print(f"Decision Stats: Consolidate: {100*consolidate_count/total_decisions:.1f}%, Ignore: {100*ignore_count/total_decisions:.1f}%, Reject: {100*reject_count/total_decisions:.1f}%")

    # Return all metrics collected during the epoch
    return global_step, epoch_summary

def validate(model: nn.Module, device: torch.device, val_loader: DataLoader, loss_fn: nn.Module, pi_monitor: SigmaPI, dataset_name: str = "Validation") -> Tuple[float, float, float, float, float, float]:
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
    if not isinstance(dataset, Sized):
        raise TypeError("Dataset is not Sized, cannot calculate accuracy.")
    
    accuracy = 100. * correct / len(dataset)
    avg_pi = sum(all_pi_scores) / len(all_pi_scores) if all_pi_scores else 0.0
    avg_surprise = sum(all_surprises) / len(all_surprises) if all_surprises else 0.0
    avg_tau = sum(all_taus) / len(all_taus) if all_taus else 0.0
    avg_gating_tau = sum(all_gating_taus) / len(all_gating_taus) if all_gating_taus else 0.0

    summary_str = f"{dataset_name} set: Avg loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, Avg PI: {avg_pi:.4f}, Avg Surprise: {avg_surprise:.4f}, Avg Tau: {avg_tau:.4f}"
    if avg_gating_tau > 0:
        summary_str += f", Avg Gating Tau: {avg_gating_tau:.4f}"
    print(summary_str)
    return avg_loss, accuracy, avg_pi, avg_surprise, avg_tau, avg_gating_tau
