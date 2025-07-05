# 变量与类名参考手册

本手册旨在统一整个 PILF 项目中的命名约定，以提高代码的可读性和可维护性。

## 核心原则

- **清晰性优于简洁性**: 变量名应清晰地表达其含义。
- **驼峰式命名法 (CamelCase)**: 用于类名，例如 `PILFTrainLoop`。
- **蛇形命名法 (snake_case)**: 用于变量、函数和文件名，例如 `expert_loss`。
- **常量全大写**: 用于不应改变的常量，例如 `BASE_CONFIG`。

## 标准命名约定

### 类名

| 名称 | 描述 |
| --- | --- |
| `VisionTransformer` | 基础的 Transformer 模型。 |
| `MoELayer` | 包含专家和线性门控的基础 MoE 层。 |
| `GaussianMoELayer` | 使用高斯分布进行路由的 MoE 层。 |
| `MemoryGaussianMoELayer` | 具备记忆和元学习能力的 `PILF-2` 核心 MoE 层。 |
| `GatingTransformer` | `MemoryGaussianMoELayer` 内部的元学习门控网络。 |
| `PICalculator` | 用于计算预测完整性 (PI) 相关指标的工具类。 |
| `RoutingExperienceBuffer` | 用于存储和采样历史路由经验的缓存区。 |
| `Trainer` | 封装了模型、优化器和训练相关状态的核心训练协调器。 |
| `BaseTrainLoop` | 训练循环的基类。 |
| `PILFTrainLoop` | `PILF-2` 模型的专用训练循环。 |
| `StrategyComponent` | 所有策略（如学习率、反向传播）的基类。 |
| `SurpriseMinKStrategy` | 基于 Surprise 过滤梯度的反向传播策略。 |
| `PILRStrategy` | 预测完整性引导的学习率调节策略。 |
| `StepResult` | 用于在日志中记录单个训练步骤指标的 `TypedDict`。 |

### 变量与参数名

| 名称 | 描述 |
| --- | --- |
| `expert_loss` | 主任务的损失，由专家网络输出计算得出。 |
| `gating_loss` | 门控网络的损失，在元学习阶段计算。 |
| `smk_min_k` | `SurpriseMinKStrategy` 中保留的最小专家数量。 |
| `top_k` | 推理时为每个 token 选择的专家数量。 |
| `pi_metrics` | 一个包含 `pi_score`, `surprise`, `tau` 等指标的字典。 |
| `surprise_min_k_expert_indices` | 由 `SurpriseMinKStrategy` 计算出的、梯度最小的专家索引。 |
| `gating_optimizer` | 专门用于更新 `GatingTransformer` 参数的优化器。 |
| `main_optimizer` | 用于更新除门控网络外所有其他模型参数的优化器。 |
