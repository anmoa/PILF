# PILF: Predictive Integrity Learning Framework

## 1. 核心目标

我们的核心目标是构建一个先进的、受认知科学启发的训练与分析框架。该框架旨在：

1. **实现并验证 PILR-S**: 将 PILR-S（Predictive Integrity Learning Rate Scheduler）从一个二元的门控机制，进化为一个由模型自身“惊奇度”驱动的、连续的学习率调度器。
2. **研究并复现 Grokking**: 在受控的实验环境中（如 MNIST），通过长时程训练，系统性地研究并复现“顿悟”（Grokking）现象，即模型从记忆式学习到泛化式理解的相变过程。
3. **探索专家特化与持续学习**: 利用 MoE（Mixture of Experts）架构和 Top-K 硬路由机制，研究模型如何在持续学习任务中，通过门控网络将不同任务智能地分配给特化专家，从而有效缓解灾难性遗忘。

## 2. 项目结构

```plain
PILF/
├── configs/# 存放所有实验的配置文件
│   ├── base_vit.py
│   ├── gbp_moe_vit.py
│   └── ...
├── models/# 模型定义 (ViT, MoE-ViT)
├── utils/# 训练、验证、绘图等辅助工具
├── run_experiment.py # (核心) 可参数化的单任务训练脚本
├── run_rehearsal_experiment.py # CIFAR/SVHN 持续学习脚本
└── run_mnist_rehearsal.py # MNIST/FashionMNIST 持续学习脚本
```

## 3. 工程准则

- **向后兼容**: 所有代码修改都应保持向后兼容性，允许旧的实验配置和脚本继续运行。
- **配置驱动**: 实验的超参数（如数据集、模型类型、PILR-S 模式）应通过配置文件或命令行参数指定，而不是硬编码在脚本中。
- **代码风格**: 代码必须简洁、自解释，移除所有不必要的注释。
- **环境管理**: **必须**使用 `uv` 管理独立的虚拟环境，并通过 `uv add` 命令将所有依赖项添加到 `pyproject.toml`。
- **纯函数原则**: 所有数据处理函数都应为纯函数。

## 5. 实验脚本参数速记

为了方便快速启动不同类型的实验，以下是主要脚本的命令行参数说明：

| 脚本 | 主要目的 | 示例命令 |
| :--- | :--- | :--- |
| `run_experiment.py` | 单任务训练或预训练 | `python run_experiment.py --config [config_path] --train_dataset MNIST --val_dataset MNIST --ood_dataset FashionMNIST` |
| `run_rehearsal_experiment.py` | 在 CIFAR/SVHN 上进行持续学习复习 | `python run_rehearsal_experiment.py --config [config_path] --checkpoint_path [ckpt_path]` |
| `run_mnist_rehearsal.py` | 在 MNIST/FashionMNIST 上进行持续学习复习 | `python run_mnist_rehearsal.py --config [config_path]` |
