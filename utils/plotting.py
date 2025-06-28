import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Optional, Any, Callable

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
    if kwargs.get('pilr_decisions') and kwargs.get('pilr_surprise_values'):
        if 'pilr_decisions' not in active_plots: active_plots.append('pilr_decisions')

    if not active_plots:
        print("No data to plot.")
        return

    num_plots = len(active_plots)
    rows = (num_plots + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(10 * 2, 5 * rows), squeeze=False)
    axes = axes.flatten()

    fig.suptitle(file_prefix, fontsize=16)

    for i, plot_key in enumerate(active_plots):
        ax = axes[i]
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

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
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
