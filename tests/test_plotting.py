import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from utils.logging.plotting import plot_core_metrics, plot_expert_dashboard
from utils.logging.types import StepResult


def generate_mock_data(num_steps=100, num_layers=4, num_experts=8, seed=42):
    rng = np.random.default_rng(seed)
    train_steps = []
    tasks = ["CIFAR10", "MNIST", "FashionMNIST"]
    for i in range(num_steps):
        task = tasks[i % len(tasks)]
        step: StepResult = {
            "global_step": i,
            "loss": np.exp(-i / num_steps) + 0.1 * rng.random(),
            "accuracy": 100 * (1 - np.exp(-i / (num_steps / 2))),
            "pi_score": 0.5 + 0.5 * np.tanh((i - num_steps / 2) / (num_steps / 4)),
            "surprise": 0.1 + 0.05 * rng.standard_normal(),
            "tau": 0.3 + 0.1 * rng.random(),
            "task_name": task,
            "top_k_expert_indices": {
                l: list(rng.choice(num_experts, 2, replace=False)) for l in range(num_layers)
            },
            "surprise_min_k_expert_indices": {
                l: list(rng.choice(num_experts, 2, replace=False)) for l in range(num_layers)
            },
        }
        train_steps.append(step)

    val_logs = {}
    for task in tasks:
        val_logs[task] = []
        for i in range(10, num_steps, 20):
            val_logs[task].append(
                {
                    "global_step": i,
                    "epoch": i // 10,
                    "loss": np.exp(-i / num_steps) + 0.05 * rng.random(),
                    "accuracy": 100 * (1 - np.exp(-i / (num_steps / 2.2))),
                    "pi_score": 0.5 + 0.5 * np.tanh((i - num_steps / 2) / (num_steps / 4.5)),
                }
            )
    return train_steps, val_logs

def test_plots():
    NUM_STEPS = 200
    NUM_LAYERS = 4
    NUM_EXPERTS = 8
    RUN_NAME = "comprehensive_test_run"
    
    train_steps, val_logs = generate_mock_data(NUM_STEPS, NUM_LAYERS, NUM_EXPERTS)
    
    output_dir = project_root / "output" / "test_plots"
    output_dir.mkdir(exist_ok=True)

    # Test Core Metrics Plot
    try:
        fig_core = plot_core_metrics(train_steps, val_logs, RUN_NAME, NUM_LAYERS, NUM_EXPERTS)
        assert isinstance(fig_core, plt.Figure)
        save_path_core = output_dir / f"{RUN_NAME}_core_metrics.png"
        fig_core.savefig(save_path_core)
        print(f"Core metrics plot saved to {save_path_core}")
    except Exception as e:
        pytest.fail(f"plot_core_metrics raised an exception: {e}")
    finally:
        plt.close("all")

    # Test Expert Dashboard Plot
    try:
        fig_expert = plot_expert_dashboard(train_steps, NUM_LAYERS, NUM_EXPERTS)
        assert isinstance(fig_expert, plt.Figure)
        save_path_expert = output_dir / f"{RUN_NAME}_expert_dashboard.png"
        fig_expert.savefig(save_path_expert)
        print(f"Expert dashboard plot saved to {save_path_expert}")
    except Exception as e:
        pytest.fail(f"plot_expert_dashboard raised an exception: {e}")
    finally:
        plt.close("all")
