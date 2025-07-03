# PILF: Predictive Integrity Learning Framework

## 1. 框架核心组件

当前框架中可组合的模块，这定义了我们实验的“基因库”。

- **模型架构**
  - `ViT`: 基线架构。标准的 Vision Transformer。
  - `LinearMoE`: 线性门控 MoE。门控机制为 `nn.Linear`。
  - `GaussMoE`: 高斯路由 MoE。核心特征是使用高斯分布进行专家路由。
  - `GaussMoE-D`: 动态 `top_k` 的高斯路由 MoE。
  - `MemoryGaussianMoE`: 记忆高斯路由 MoE。通过在反向传播中利用历史路由分布来增强模型的知识隔离和梯度正交化能力。
- **定制优化器**
  - `LGO`: 局部门控优化器 (Local Gating Optimizer)。通过局部损失（置信度与负载均衡）独立优化门控网络。
- **学习率引擎**
  - `固定 LR`: 默认行为，不使用任何调度器。
  - `PILR-S`: 预测完整性学习率调度器 (静态)。基于`Surprise`调节学习率。
  - `PILR-D`: 预测完整性学习率调度器 (动态)。基于`Surprise`调节学习率变化比率的动态学习率引擎 (原 `PISA`)。可以通过设置 `expert_initial_var` 为一个大值（例如1000.0）来“软禁用”专家网络的PILR调制。
- **前向路由策略**
  - `Linear Top-K`: 简单的线性层 + Top-K。
  - `Gaussian Top-K`: 基于输入与专家高斯分布的匹配度进行路由。
  - `Gaussian Top-K with PILR-A`: 对 OOD 输入动态抑制高置信度但低匹配度的专家。
- **反向更新策略**
  - `Standard`: 全量更新。对所有参数执行梯度下降。
  - `Selective`: 稀疏更新。仅更新被激活的 Top-K 专家的权重。
  - `SMK`: 意外最小 K 更新 (Surprise Min-K)。在**所有**专家中，仅更新“意外”最低的 `min_k` 个。
  - `SAMK`: 激活内意外最小 K 更新 (Surprise Activated Min-K)。在**被激活**的专家中，再筛选出“意外”最低的 `min_k` 个进行更新。

## 2. 工程准则

- **向后兼容**: 所有代码修改都应保持向后兼容性，允许旧的实验配置和脚本继续运行。
- **配置驱动**: 实验的超参数（如数据集、模型类型、PILR-S 模式）应通过配置文件或命令行参数指定，而不是硬编码在脚本中。
- **代码风格**: 代码必须简洁、自解释，移除所有不必要的注释。如果必须要保留一定的注释，则使用英文注释。如果有中文注释，在进行编辑的时候要清理或翻译。
- **环境管理**: **必须**使用 `uv` 管理独立的虚拟环境，并通过 `uv add` 命令将所有依赖项添加到 `pyproject.toml`。
- **纯函数原则**: 所有数据处理函数都应为纯函数。
- **类型安全**: 所有新代码都必须有完整的类型标注，并能通过 `ruff check --select I --fix` 的严格检查。
- **安全检查**: 在进行了10次文件编辑后，**必须**同时运行 `ruff check . --fix` 和 `mypy .` 来进行一次全面的静态类型检查，以确保代码库的健壮性。
- **事项记录**：在安全检查通过后，更新`process.md`。
- **宣布任务完成**：确保 `process.md` 中已经没有待办任务后，宣布任务完成。

## 3. 实验脚本参数速记

为了方便快速启动不同类型的实验，所有实验都通过唯一的 `train.py` 脚本启动，该脚本由一个**调度文件**和一个**模型配置文件**共同驱动：

| 脚本 | 主要目的 | 示例命令 |
| :--- | :--- | :--- |
| `train.py` | 运行所有类型的实验 | `python train.py --schedule <schedule_path> --model-config <model_config_path>` |
| `train.py` | 从检查点恢复训练 | `python train.py --schedule <schedule_path> --model-config <model_config_path> --resume-from <checkpoint_path>` |
| `train.py` | 运行马拉松复习实验 | `python train.py --schedule schedules/marathon_v3 --model-config configs/large_gauss_moe_smk_pilr_d` |
