import matplotlib

matplotlib.use('Agg')
from typing import Dict, List, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np

from utils.types import StepResult


def plot_expert_scatter(
    train_steps: List[StepResult],
    num_layers: int,
    num_experts: int,
    plot_type: str
) -> plt.Figure:
    fig, axes = plt.subplots(num_layers, 1, figsize=(15, 6 * num_layers), squeeze=False)
    fig.suptitle(f"Expert {plot_type.capitalize()} Activation Timeline", fontsize=16)

    task_names = sorted(list(set(s['task_name'] for s in train_steps)))
    task_colors = {task: plt.cm.get_cmap('tab10', len(task_names))(i) for i, task in enumerate(task_names)}

    for layer_idx in range(num_layers):
        ax = axes[layer_idx, 0]
        ax.set_title(f"Layer {layer_idx}")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Expert ID")
        ax.set_yticks(np.arange(num_experts))
        ax.set_ylim(-0.5, num_experts - 0.5)
        ax.grid(True, linestyle='--', alpha=0.6)

        for step_result in train_steps:
            global_step = step_result['global_step']
            task_name = step_result['task_name']
            
            expert_indices_dict = cast(Optional[Dict[int, List[int]]], step_result.get(f'{plot_type}_expert_indices'))
            if expert_indices_dict is not None and layer_idx in expert_indices_dict:
                expert_ids = expert_indices_dict[layer_idx]
                for expert_id in expert_ids:
                    ax.scatter(global_step, expert_id, color=task_colors[task_name], s=15, alpha=0.5)
    
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=task,
                          markerfacecolor=task_colors[task], markersize=10)
               for task in task_colors]
    fig.legend(handles=handles, title="Task", loc='upper right')

    plt.tight_layout(rect=(0, 0, 0.9, 1))
    return fig

def plot_expert_heatmap(
    train_steps: List[StepResult],
    num_layers: int,
    num_experts: int,
    plot_type: str
) -> plt.Figure:
    task_names = sorted(list(set(s['task_name'] for s in train_steps)))
    
    # Initialize activation counts matrix
    activation_counts = np.zeros((num_layers, num_experts, len(task_names)))

    # Populate the matrix
    for step in train_steps:
        task_idx = task_names.index(step['task_name'])
        expert_indices_dict = cast(Optional[Dict[int, List[int]]], step.get(f'{plot_type}_expert_indices'))
        if expert_indices_dict:
            for layer_idx, expert_ids in expert_indices_dict.items():
                if layer_idx < num_layers:
                    for expert_id in expert_ids:
                        if expert_id < num_experts:
                            activation_counts[layer_idx, expert_id, task_idx] += 1

    fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 8), squeeze=False)
    fig.suptitle(f'Total Expert {plot_type.capitalize()} Activations per Task', fontsize=16)

    for layer_idx in range(num_layers):
        ax = axes[0, layer_idx]
        im = ax.imshow(activation_counts[layer_idx], cmap='viridis', aspect='auto')

        ax.set_title(f'Layer {layer_idx}')
        ax.set_xticks(np.arange(len(task_names)))
        ax.set_xticklabels(task_names, rotation=45, ha="right")
        ax.set_yticks(np.arange(num_experts))
        ax.set_ylabel('Expert ID')

        # Add text annotations
        for i in range(num_experts):
            for j in range(len(task_names)):
                ax.text(j, i, int(activation_counts[layer_idx, i, j]),
                        ha="center", va="center", color="w")

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='Activation Count')
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    return fig

# Future plotting functions (placeholders)
def plot_core_metrics(train_steps: List[StepResult], val_logs: Dict[str, List[Tuple[int, float]]]) -> plt.Figure:
    # Placeholder for core metrics plotting
    fig, ax = plt.subplots()
    ax.set_title("Core Metrics Placeholder")
    return fig

def plot_lr_scatter(train_steps: List[StepResult]) -> plt.Figure:
    # Placeholder for LR scatter plotting
    fig, ax = plt.subplots()
    ax.set_title("LR Scatter Placeholder")
    return fig
