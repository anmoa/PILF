# PILF: Predictive Integrity Learning Framework

我们的核心目标是构建一个先进的、受认知科学启发的模型训练与分析框架。更多信息可查看 [PILF.md](./PILF.md)

## 2. 框架核心组件

当前框架中可组合的模块，这定义了我们实验的“基因库”。

- **模型架构**
  - `Dense` (基线 ViT): 标准的 Vision Transformer。
  - `MoE` (线性门控): `MoELayer` 使用 `nn.Linear` 进行门控。
  - `GPIL-MoE` (高斯路由): `GaussianMoELayer` 使用高斯分布进行路由。
  - `GPILD-MoE` (动态高斯): 计划中，将实现动态 `top_k`。
  - `G2PILD-MoE`(生成对抗高斯): 计划中，将实现生成式自我巩固。
- **学习率调度**
  - `固定 LR`: 默认行为，不使用任何调度器。
  - `PILR-S` (固定 Sigma): 通过在配置中设置 `beta=0` 实现。
  - `PISA` (自适应 Sigma): `PisaAdaptor` 动态调整 `sigma`，支持单/双模式。
- **前向路由策略**
  - `Linear Top-K`: 简单的线性层 + Top-K。
  - `Gaussian Top-K`: 基于输入与专家高斯分布的匹配度进行路由。
  - `Gaussian Top-K with PISA`: 对 OOD 输入动态抑制高置信度但低匹配度的专家。
- **反向更新策略**
  - `Standard` (全量更新): 对所有参数执行梯度下降。
  - `Selective` (稀疏更新): 仅更新被激活的 Top-K 专家的权重。
  - `Surprise-Min-K`: 在**所有专家**中，仅更新 Surprise 最低的 Min-K 个。
  - `Surprise-Act-Min-K`: 在**所有激活专家**中，仅更新 Surprise 最低的 Min-K 个。

## 3. 工程准则

- **向后兼容**: 所有代码修改都应保持向后兼容性，允许旧的实验配置和脚本继续运行。
- **配置驱动**: 实验的超参数（如数据集、模型类型、PILR-S 模式）应通过配置文件或命令行参数指定，而不是硬编码在脚本中。
- **代码风格**: 代码必须简洁、自解释，移除所有不必要的注释。如果必须要保留一定的注释，则使用英文注释。如果有中文注释，在进行编辑的时候要清理或翻译。
- **环境管理**: **必须**使用 `uv` 管理独立的虚拟环境，并通过 `uv add` 命令将所有依赖项添加到 `pyproject.toml`。
- **纯函数原则**: 所有数据处理函数都应为纯函数。

## 5. 实验脚本参数速记

为了方便快速启动不同类型的实验，所有实验都通过唯一的 `train.py` 脚本启动，该脚本由一个**调度文件**和一个**模型配置文件**共同驱动：

| 脚本 | 主要目的 | 示例命令 |
| :--- | :--- | :--- |
| `train.py` | 运行所有类型的实验 | `python train.py --schedule <schedule_path> --model-config <model_config_path>` |
| `train.py` | 运行马拉松复习实验 | `python train.py --schedule schedules/marathon_v1.py --model-config configs/large_pilr_mnist.py` |
