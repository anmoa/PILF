import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from typing import Dict, List, Tuple, Optional, Any, Callable

def plot_gpil_routing_confidence(axes: np.ndarray, all_log_probs: List[List[torch.Tensor]], num_layers: int):
    """Plots histograms of routing log probabilities for each GPIL-MoE layer."""
    for i in range(num_layers):
        ax = axes[i]
        # Collect log_probs for layer i from all batches
        layer_log_probs = [batch_log_probs[i].detach().cpu().numpy().flatten() for batch_log_probs in all_log_probs if len(batch_log_probs) > i]
        if not layer_log_probs: continue
        
        flat_log_probs = np.concatenate(layer_log_probs)
        ax.hist(flat_log_probs, bins=50, alpha=0.7, density=True)
        ax.set_title(f'Layer {i+1} Routing Confidence')
        ax.set_xlabel('Log Probability')
        ax.set_ylabel('Density')
        ax.grid(True, linestyle='--', alpha=0.6)

def plot_gpil_expert_activation(ax: plt.Axes, all_top_indices: List[List[torch.Tensor]], num_layers: int, num_experts: int):
    """Plots a heatmap of expert activation frequencies across layers."""
    activation_counts = np.zeros((num_layers, num_experts))
    
    for batch_indices in all_top_indices:
        for layer_idx, indices_tensor in enumerate(batch_indices):
            if layer_idx < num_layers:
                unique_experts, counts = torch.unique(indices_tensor.cpu(), return_counts=True)
                activation_counts[layer_idx, unique_experts.numpy()] += counts.numpy()

    if np.sum(activation_counts) == 0: return # No data to plot

    im = ax.imshow(activation_counts, cmap='viridis', aspect='auto')
    ax.set_xticks(np.arange(num_experts))
    ax.set_yticks(np.arange(num_layers))
    ax.set_xticklabels(np.arange(1, num_experts + 1))
    ax.set_yticklabels(np.arange(1, num_layers + 1))
    ax.set_xlabel('Expert ID')
    ax.set_ylabel('Layer Index')
    ax.set_title('Expert Activation Frequency')
    
    # Add text annotations
    for i in range(num_layers):
        for j in range(num_experts):
            ax.text(j, i, f'{int(activation_counts[i, j])}', ha='center', va='center', color='w' if activation_counts[i, j] < activation_counts.max() / 2 else 'black')

    plt.colorbar(im, ax=ax, label='Activation Count')


def _plot_subplot(ax: plt.Axes, data_map: Dict[str, Any], x_label: str, y_label: str, title: str):
    """Helper function to plot a single subplot."""
    for label, values in data_map.items():
        if values:
            # Check if data is tuple (steps, values) or just values
            if isinstance(values[0], tuple):
                steps, y_values = zip(*values)
                ax.plot(steps, y_values, marker='o' if len(values) < 50 else '', linestyle='-', label=label)
            else:
                ax.plot(range(len(values)), values, label=label)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

def plot_metrics(
    step_metrics: Dict[str, List[Tuple[int, float]]],
    epoch_metrics: Dict[str, List[Tuple[int, float]]],
    output_dir: str,
    file_prefix: str,
    **kwargs: Optional[Any]
) -> None:
    
    plot_configs = {
        'loss': {'title': 'Loss', 'y_label': 'Loss'},
        'acc': {'title': 'Accuracy', 'y_label': 'Accuracy (%)'},
        'pi': {'title': 'Predictive Integrity (PI)', 'y_label': 'PI Score'},
        'surprise': {'title': 'Surprise (Gradient Norm)', 'y_label': 'Surprise'},
        'tau': {'title': 'Tau (Entropy - Experts)', 'y_label': 'Tau (Experts)'},
        'gating_tau': {'title': 'Tau (Entropy - Gating)', 'y_label': 'Tau (Gating)'},
        'lr_mod': {'title': 'Learning Rate Modifier', 'y_label': 'LR Modifier', 'x_label': 'Training Steps'},
        'sigma': {'title': 'Adaptive Sigma', 'y_label': 'Sigma Value', 'x_label': 'Training Steps'},
        'pilr_decisions': {'title': 'PILR Decision vs. Surprise', 'y_label': 'Surprise', 'x_label': 'Global Steps'},
        'gpil_activation': {'title': 'GPIL Expert Activation'},
        'gpil_confidence': {'title': 'GPIL Routing Confidence'},
    }

    # Prepare data for plotting
    plots_data: Dict[str, Dict[str, Any]] = {key: {} for key in plot_configs}

    for metric, config in plot_configs.items():
        # Step metrics
        train_key = f'train_{metric}'
        if train_key in step_metrics and step_metrics[train_key]:
            plots_data[metric][f'Train {metric} (Step)'] = step_metrics[train_key]
        # Epoch metrics
        for key, data in epoch_metrics.items():
            if metric in key and data:
                label = key.replace('_', ' ').title()
                plots_data[metric][f'{label} (Epoch Avg)'] = data

    # Handle special kwargs plots
    plots_data['lr_mod']['Overall'] = kwargs.get('lr_mod_values')
    plots_data['lr_mod']['Gating'] = kwargs.get('gating_lr_mod_values')
    plots_data['lr_mod']['Experts'] = kwargs.get('expert_lr_mod_values')
    
    plots_data['sigma']['Overall'] = kwargs.get('sigma_values')
    plots_data['sigma']['Gating'] = kwargs.get('gating_sigma_values')
    plots_data['sigma']['Experts'] = kwargs.get('expert_sigma_values')

    active_plots = [key for key, data in plots_data.items() if any(data.values())]
    
    # Check for decision data from epoch_summary
    if 'surprise_values' in epoch_metrics and 'decisions' in epoch_metrics:
        if epoch_metrics['surprise_values'] and epoch_metrics['decisions']:
            kwargs['pilr_surprise_values'] = epoch_metrics['surprise_values']
            kwargs['pilr_decisions'] = epoch_metrics['decisions']

    if kwargs.get('pilr_decisions') and kwargs.get('pilr_surprise_values'):
        if 'pilr_decisions' not in active_plots: active_plots.append('pilr_decisions')
    
    all_top_indices = kwargs.get('all_top_indices')
    all_log_probs = kwargs.get('all_log_probs')
    
    gpil_plots = 0
    num_layers = 0
    if all_top_indices and all_log_probs:
        active_plots.append('gpil_activation')
        active_plots.append('gpil_confidence')
        # Determine number of layers from the data, ensuring it's not None
        if isinstance(all_top_indices, list) and all_top_indices:
            num_layers = len(all_top_indices[0])
            gpil_plots = 1 + num_layers # 1 for activation heatmap, num_layers for confidence histograms

    if not active_plots:
        print("No data to plot.")
        return

    num_standard_plots = len(active_plots) - (2 if gpil_plots > 0 else 0)
    total_subplots = num_standard_plots + gpil_plots
    
    # Adjust layout based on total plots
    if total_subplots <= 2:
        rows, cols = 1, total_subplots
    elif total_subplots <= 4:
        rows, cols = 2, 2
    elif total_subplots <= 6:
        rows, cols = 2, 3
    elif total_subplots <= 9:
        rows, cols = 3, 3
    else:
        rows = (total_subplots + 2) // 3
        cols = 3

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)
    axes = axes.flatten()
    fig.suptitle(file_prefix, fontsize=16)
    
    plot_idx = 0
    for plot_key in active_plots:
        if plot_key not in ['gpil_activation', 'gpil_confidence']:
            ax = axes[plot_idx]
            config = plot_configs[plot_key]
            
            if plot_key == 'pilr_decisions':
                surprise_values = kwargs.get('pilr_surprise_values')
                decisions = kwargs.get('pilr_decisions')
                if surprise_values and decisions:
                    plot_pilr_decisions_internal(ax, surprise_values, decisions)
                    ax.set_title(config['title'])
            else:
                _plot_subplot(
                    ax=ax,
                    data_map=plots_data[plot_key],
                    x_label=config.get('x_label', 'Global Steps'),
                    y_label=config['y_label'],
                    title=config['title']
                )
            plot_idx += 1

    # Handle GPIL plots
    if gpil_plots > 0 and all_top_indices and all_log_probs:
        # Plot activation heatmap
        ax_activation = axes[plot_idx]
        # Ensure all_log_probs is valid before indexing
        if isinstance(all_log_probs, list) and all_log_probs and isinstance(all_log_probs[0], list) and all_log_probs[0]:
            num_experts = all_log_probs[0][0].shape[-1]
            plot_gpil_expert_activation(ax_activation, all_top_indices, num_layers, num_experts)
            plot_idx += 1
            
            # Plot confidence histograms
            confidence_axes = axes[plot_idx : plot_idx + num_layers]
            plot_gpil_routing_confidence(confidence_axes, all_log_probs, num_layers)
            plot_idx += num_layers

    # Hide unused subplots
    for j in range(plot_idx, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{file_prefix}-Metrics.png"
    plt.savefig(os.path.join(output_dir, file_name))
    plt.close()
    print(f"\nPlots saved to: {os.path.abspath(os.path.join(output_dir, file_name))}")


def plot_pilr_decisions_internal(
    ax: plt.Axes,
    pilr_surprise_values: List[Tuple[int, float]],
    pilr_decisions: List[Tuple[int, str]]
) -> None:
    
    steps, surprise_vals = zip(*pilr_surprise_values)
    _, decisions = zip(*pilr_decisions)

    color_map = {'CONSOLIDATE': 'green', 'IGNORE': 'blue', 'REJECT': 'red'}
    marker_map = {'CONSOLIDATE': 'o', 'IGNORE': 'x', 'REJECT': '^'}
    label_map = {'CONSOLIDATE': 'Consolidate', 'IGNORE': 'Ignore', 'REJECT': 'Reject'}

    for decision_type in color_map:
        filtered_steps = [steps[i] for i, d in enumerate(decisions) if d == decision_type]
        filtered_surprise = [surprise_vals[i] for i, d in enumerate(decisions) if d == decision_type]
        
        if filtered_steps:
            ax.scatter(
                filtered_steps,
                filtered_surprise,
                color=color_map[decision_type],
                marker=marker_map[decision_type],
                label=label_map[decision_type],
                alpha=0.6
            )
    ax.set_xlabel('Global Steps')
    ax.set_ylabel('Surprise (Gradient Norm)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
