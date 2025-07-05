# 日志与可视化规范

本文档定义了 PILF-2 框架中用于训练后分析的日志记录和可视化标准。

## 1. 数据采集

### 1.1. 数据容器

- **容器**: `Trainer.epoch_results`
- **类型**: `List[StepResult]`
- **生命周期**: 在训练开始时初始化，在整个训练过程中持续追加，不得清空。

### 1.2. `StepResult` 数据字典

所有训练循环 (`BaseTrainLoop`, `PILFTrainLoop`) 在每个训练步骤 (step) 都必须生成一个 `StepResult` 字典，并确保其包含以下键：

| 键 (Key)                        | 类型                   | 来源          | 适用模式   | 描述                           |
| :------------------------------ | :--------------------- | :------------ | :--------- | :----------------------------- |
| `global_step`                   | `int`                  | Trainer       | 所有       | 全局唯一 ID                    |
| `loss`                          | `float`                | Train Loop    | 所有       | 主任务损失                     |
| `accuracy`                      | `float`                | Train Loop    | 所有       | 当前批次准确率                 |
| `task_name`                     | `str`                  | Train Loop    | 所有       | 当前训练任务名                 |
| `pi_score`                      | `float`                | PICalculator  | 所有       | PI 分数                        |
| `surprise`                      | `float`                | PICalculator  | 所有       | 梯度范数                       |
| `tau`                           | `float`                | PICalculator  | 所有       | 输出熵                         |
| `gating_loss`                   | `float`                | PILFTrainLoop | MGM-MoE    | 门控网络损失                   |
| `top_k_expert_indices`          | `Dict[int, List[int]]` | Train Loop    | 所有 MoE   | `{层索引: [激活专家ID列表]}`   |
| `surprise_min_k_expert_indices` | `Dict[int, List[int]]` | SMK Strategy  | `smk` 模式 | `{层索引: [被更新专家ID列表]}` |

## 2. 可视化产出物

训练结束后，`main.py` 必须根据收集到的 `trainer.epoch_results` 生成以下可视化图表。

### 2.1. 全局核心指标图

- **文件名**: `core_metrics_final.png`
- **函数**: `plot_core_metrics()`
- **适用范围**: **所有**训练运行。
- **布局**: 3x2 子图网格。
- **子图内容**:
  1. Loss (vs. global_step)
  2. Accuracy (vs. global_step)
  3. PI Score (vs. global_step)
  4. Surprise (vs. global_step)
  5. Tau (vs. global_step)
  6. 图例 (Legend)

### 2.2. 专家激活仪表盘

- **文件名**: `expert_dashboard_final.png`
- **函数**: `plot_expert_dashboard()`
- **适用范围**: **所有 MoE** 模型 (`moe`, `gaussian_moe`, `memory_gaussian_moe`)。
- **布局**: 2x2 子图网格。
- **子图内容**:
  - **左上**: Top-K 路由决策散点图 (`top_k_expert_indices`)。
  - **左下**: Top-K 专家激活热力图 (`top_k_expert_indices`)。
  - **右上**:
    - **如果** `surprise_min_k_expert_indices` **存在**: Min-K 更新决策散点图。
    - **否则**: 显示 "N/A for non-SMK mode"。
  - **右下**:
    - **如果** `surprise_min_k_expert_indices` **存在**: Min-K 专家更新热力图。
    - **否则**: 显示 "N/A for non-SMK mode"。
