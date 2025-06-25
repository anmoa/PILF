# PILF: Predictive Integrity Learning Framework

## 1. 核心目标

我们的核心目标是构建一个先进的、受认知科学启发的训练与分析框架。该框架旨在：

1. **实现并验证 PILR-S**: 将“门控反向传播”（Gated Backpropagation）从一个二元的门控机制，进化为一个由模型自身“惊奇度”驱动的、连续的学习率调度器（Predictive Integrity Learning Rate Scheduler）。
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
- **配置驱动**: 实验的超参数（如数据集、模型类型、GBP 模式）应通过配置文件或命令行参数指定，而不是硬编码在脚本中。
- **代码风格**: 代码必须简洁、自解释，移除所有不必要的注释。

## 4. 当前开发路线图

我们当前遵循一个清晰的三阶段计划，以实现我们的核心目标。

1. **Phase 0: 基础重构 (已完成)**
    - **SigmaPI Torch 化**: 已将 `main.py` 中的计算逻辑完全重构为 `torch` 原生操作，以兼容 `torch.compile`。
    - **实验脚本参数化**: 已重构 `run_experiment.py`，使其能够通过命令行参数灵活地加载不同的数据集。
    - **数据集支持**: 已在 `get_dataloaders` 中添加了对 `MNIST` 和 `FashionMNIST` 的支持。

2. **Phase 1: 算法重铸 (已完成)**
    - **实现 GBP-LR**: 已重构 `test/utils/training.py`，实现了由 `Surprise` 驱动的动态学习率调度器。
    - **保留双模式**: `train` 函数现在支持 `gbp_mode` 参数，可以在经典的“门控”模式和新的“LR 调度器”模式之间切换，确保了向后兼容。

3. **Phase 2: 实验执行 (当前阶段)**
    - **目标**: 使用我们新构建的 GBP-LR 框架，在 **FashionMNIST 数据集上进行长时程训练，并使用 MNIST 作为复习集**，以寻找和分析“Grokking”现象。
    - **下一步**: 启动使用 `test/configs/tiny_gbp_mnist.py` 配置的训练任务。

## 5. 实验脚本参数速记

为了方便快速启动不同类型的实验，以下是主要脚本的命令行参数说明：

| 脚本 | 主要目的 | 示例命令 |
| :--- | :--- | :--- |
| `run_experiment.py` | 单任务训练或预训练 | `python run_experiment.py --config [config_path] --train_dataset MNIST --val_dataset MNIST --ood_dataset FashionMNIST` |
| `run_rehearsal_experiment.py` | 在 CIFAR/SVHN 上进行持续学习复习 | `python run_rehearsal_experiment.py --config [config_path] --checkpoint_path [ckpt_path]` |
| `run_mnist_rehearsal.py` | 在 MNIST/FashionMNIST 上进行持续学习复习 | `python run_mnist_rehearsal.py --config [config_path]` |
