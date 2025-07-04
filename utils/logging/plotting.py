from typing import Dict, List, Literal, Optional, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from utils.logging.types import StepResult, ValidationResult

matplotlib.use("Agg")

def plot_expert_scatter(
    ax: plt.Axes,
    train_steps: List[StepResult],
    num_layers: int,
    num_experts: int,
    expert_indices_key: str,
    title: str,
):
    ax.set_title(title)
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Layer * Num_Experts + Expert_ID")
    ax.grid(True, linestyle="--", alpha=0.6)
    
    task_names = sorted(list(set(s["task_name"] for s in train_steps)))
    task_colors = {
        task: plt.cm.get_cmap("tab10", len(task_names))(i)
        for i, task in enumerate(task_names)
    }

    y_ticks = []
    y_tick_labels = []

    for layer_idx in range(num_layers):
        for expert_id in range(num_experts):
            y_pos = layer_idx * num_experts + expert_id
            y_ticks.append(y_pos)
            y_tick_labels.append(f"L{layer_idx} E{expert_id}")

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels, fontsize=8)
    ax.set_ylim(-0.5, num_layers * num_experts - 0.5)

    for step_result in train_steps:
        global_step = step_result["global_step"]
        task_name = step_result["task_name"]
        
        expert_indices_dict = cast(
            Optional[Dict[int, List[int]]], step_result.get(expert_indices_key)
        )
        if expert_indices_dict:
            for layer_idx, expert_ids in expert_indices_dict.items():
                for expert_id in expert_ids:
                    y_pos = layer_idx * num_experts + expert_id
                    ax.scatter(
                        global_step,
                        y_pos,
                        color=task_colors[task_name],
                        s=15,
                        alpha=0.5,
                    )

def plot_expert_heatmap(
    ax: plt.Axes,
    train_steps: List[StepResult],
    num_layers: int,
    num_experts: int,
    expert_indices_key: str,
    title: str,
):
    task_names = sorted(list(set(s["task_name"] for s in train_steps)))
    activation_counts = np.zeros((num_layers * num_experts, len(task_names)))

    for step in train_steps:
        task_idx = task_names.index(step["task_name"])
        expert_indices_dict = cast(
            Optional[Dict[int, List[int]]], step.get(expert_indices_key)
        )
        if expert_indices_dict:
            for layer_idx, expert_ids in expert_indices_dict.items():
                for expert_id in expert_ids:
                    y_pos = layer_idx * num_experts + expert_id
                    if y_pos < activation_counts.shape[0]:
                        activation_counts[y_pos, task_idx] += 1
    
    im = ax.imshow(activation_counts, cmap="viridis", aspect="auto")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(task_names)))
    ax.set_xticklabels(task_names, rotation=45, ha="right")
    ax.set_ylabel("Layer * Num_Experts + Expert_ID")
    
    y_ticks = np.arange(num_layers * num_experts)
    y_tick_labels = [f"L{l}E{e}" for l in range(num_layers) for e in range(num_experts)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels, fontsize=6)
    
    return im

def plot_expert_dashboard(
    train_steps: List[StepResult], num_layers: int, num_experts: int
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(20, 18), constrained_layout=True)
    fig.suptitle("Expert Activation Dashboard", fontsize=16)

    plot_expert_scatter(
        axes[0, 0],
        train_steps,
        num_layers,
        num_experts,
        "top_k_expert_indices",
        "Top-K Activation Scatter",
    )
    plot_expert_scatter(
        axes[0, 1],
        train_steps,
        num_layers,
        num_experts,
        "surprise_min_k_expert_indices",
        "Surprise Min-K Activation Scatter",
    )
    
    im1 = plot_expert_heatmap(
        axes[1, 0],
        train_steps,
        num_layers,
        num_experts,
        "top_k_expert_indices",
        "Top-K Activation Heatmap",
    )
    
    im2 = plot_expert_heatmap(
        axes[1, 1],
        train_steps,
        num_layers,
        num_experts,
        "surprise_min_k_expert_indices",
        "Surprise Min-K Activation Heatmap",
    )

    fig.colorbar(im1, ax=axes[1, 0], shrink=0.8)
    fig.colorbar(im2, ax=axes[1, 1], shrink=0.8)

    return fig

def plot_core_metrics(
    train_steps: List[StepResult],
    val_logs: Dict[str, List[ValidationResult]],
    run_name: str,
    num_layers: int,
    num_experts: int,
) -> plt.Figure:
    fig, axes = plt.subplots(3, 2, figsize=(20, 15), constrained_layout=True)
    fig.suptitle(f"Core Metrics for {run_name}", fontsize=16)

    MetricKey = Literal["loss", "accuracy", "pi_score", "surprise", "tau"]
    metrics_to_plot: Dict[MetricKey, str] = {
        "loss": "Loss",
        "accuracy": "Accuracy",
        "pi_score": "PI Score",
        "surprise": "Surprise",
        "tau": "Tau",
    }

    axes_flat = axes.flatten()
    task_names = sorted(list(set(s["task_name"] for s in train_steps)))
    task_colors = {
        task: plt.cm.get_cmap("tab10", len(task_names))(i)
        for i, task in enumerate(task_names)
    }

    for i, (metric, title) in enumerate(metrics_to_plot.items()):
        ax = axes_flat[i]
        for task_name in task_names:
            task_specific_steps = [s for s in train_steps if s["task_name"] == task_name]
            train_data = [
                (s["global_step"], s[metric])
                for s in task_specific_steps
                if s.get(metric) is not None
            ]
            if train_data:
                steps, values = zip(*train_data, strict=False)
                ax.plot(
                    list(steps),
                    list(values),
                    label=f"Train {task_name}",
                    alpha=0.7,
                    color=task_colors[task_name],
                )

        for task, val_data in val_logs.items():
            val_points = [
                (v["global_step"], v[metric])
                for v in val_data
                if v.get(metric) is not None
            ]
            if val_points:
                steps, values = zip(*val_points, strict=False)
                ax.plot(
                    list(steps),
                    list(values),
                    marker="o",
                    linestyle="--",
                    label=f"Val {task}",
                )

        ax.set_title(title)
        ax.set_xlabel("Global Step")
        ax.set_ylabel(title)
        ax.grid(True, linestyle="--", alpha=0.6)

    # Legend subplot
    legend_ax = axes_flat[5]
    legend_ax.axis("off")
    handles, labels = [], []
    for ax in axes_flat[:5]:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=False):
            if label not in labels:
                labels.append(label)
                handles.append(handle)
    
    if handles:
        legend_ax.legend(handles, labels, loc="center", fontsize="large")

    return fig
