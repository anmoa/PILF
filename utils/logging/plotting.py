from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from utils.logging.types import StepResult, ValidationResult

matplotlib.use("Agg")

def plot_core_metrics(
    train_steps: List[StepResult],
    val_logs: Dict[str, List[ValidationResult]],
    run_name: str,
) -> plt.Figure:
    fig, axes = plt.subplots(3, 2, figsize=(20, 15), constrained_layout=True)
    fig.suptitle(f"Core Metrics for {run_name}", fontsize=16)

    metrics_to_plot = {
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
                (s["global_step"], s.get(metric))
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
                (v["global_step"], v.get(metric))
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


def plot_expert_dashboard(
    train_steps: List[StepResult], num_layers: int, num_experts: int
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(20, 18), constrained_layout=True)
    fig.suptitle("Expert Activation Dashboard", fontsize=16)

    plot_expert_scatter(axes[0, 0], train_steps, num_layers, num_experts, "top_k_expert_indices", "Top-K Activation Scatter")
    plot_expert_heatmap(axes[1, 0], train_steps, num_layers, num_experts, "top_k_expert_indices", "Top-K Activation Heatmap")

    if any("surprise_min_k_expert_indices" in step for step in train_steps):
        plot_expert_scatter(axes[0, 1], train_steps, num_layers, num_experts, "surprise_min_k_expert_indices", "Surprise Min-K Activation Scatter")
        plot_expert_heatmap(axes[1, 1], train_steps, num_layers, num_experts, "surprise_min_k_expert_indices", "Surprise Min-K Activation Heatmap")
    else:
        for ax in [axes[0, 1], axes[1, 1]]:
            ax.text(0.5, 0.5, "N/A for non-SMK mode", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

    return fig


def plot_expert_scatter(ax, train_steps, num_layers, num_experts, key, title):
    ax.set_title(title)
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Expert ID")
    ax.grid(True, linestyle="--", alpha=0.6)
    
    task_names = sorted(list(set(s["task_name"] for s in train_steps)))
    task_colors = {
        task: plt.cm.get_cmap("tab10", len(task_names))(i)
        for i, task in enumerate(task_names)
    }

    ax.set_ylim(-0.5, num_layers * num_experts - 0.5)
    ax.set_yticks(np.arange(0, num_layers * num_experts, num_experts))
    ax.set_yticklabels([f"Layer {i}" for i in range(num_layers)])

    points = []
    for step in train_steps:
        if key in step:
            for layer_idx, expert_ids in step[key].items():
                for expert_id in expert_ids:
                    points.append((step["global_step"], layer_idx * num_experts + expert_id, task_colors[step["task_name"]]))

    if points:
        if len(points) > 5000:
            rng = np.random.default_rng()
            sampled_indices = rng.choice(len(points), size=5000, replace=False)
            points = [points[i] for i in sampled_indices]

        steps, y_pos, colors = zip(*points, strict=False)
        ax.scatter(steps, y_pos, c=colors, s=10, alpha=0.5, edgecolors='none')


def plot_expert_heatmap(ax, train_steps, num_layers, num_experts, key, title):
    ax.set_title(title)
    task_names = sorted(list(set(s["task_name"] for s in train_steps)))
    
    counts = np.zeros((num_layers * num_experts, len(task_names)))
    for step in train_steps:
        if key in step:
            task_idx = task_names.index(step["task_name"])
            for layer_idx, expert_ids in step[key].items():
                for expert_id in expert_ids:
                    counts[layer_idx * num_experts + expert_id, task_idx] += 1
    
    normalized_counts = counts / (counts.sum(axis=0, keepdims=True) + 1e-9)
    im = ax.imshow(normalized_counts, cmap="viridis", aspect="auto")
    
    ax.set_xticks(np.arange(len(task_names)))
    ax.set_xticklabels(task_names, rotation=45, ha="right")
    ax.set_ylabel("Expert")
    
    y_tick_labels = [f"L{l}E{e}" for l in range(num_layers) for e in range(num_experts)]
    ax.set_yticks(np.arange(len(y_tick_labels)))
    ax.set_yticklabels(y_tick_labels, rotation=0, fontsize=6)
    plt.colorbar(im, ax=ax)
