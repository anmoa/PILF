# PILF 训练框架重构：日志与可视化系统 V2

我们将分三步走：首先，改造数据源，确保所有需要的指标都被准确捕获；其次，重构绘图逻辑，用模块化的方式生成新的图表；最后，将所有内容对接到 `TensorBoard`。

## Phase 1: 数据采集层重构 (Data Collection)

**目标**: 改造 `utils/training.py` 和 `utils/strategies.py`，使其能够捕获并返回一个结构化的、包含所有绘图所需指标的字典。

### 1.1. 定义标准数据容器 (`StepResult`)

我们将在 `utils/training.py` 或一个新文件 `utils/types.py` 中定义一个 `TypedDict` 或 `dataclass`，作为 `train` 函数中 `step` 级别数据的标准容器。这取代了目前依赖字符串键的 `epoch_summary` 字典，提高了代码的可读性和健壮性。

```python
from typing import TypedDict, List, Optional, Dict

class StepResult(TypedDict):
    # 核心指标
    loss: float
    accuracy: float
    pi_score: float
    surprise: float
    tau: float
    gating_tau: Optional[float]

    # PISA/PILR-S 相关
    lr_mod: Optional[float]
    gating_lr_mod: Optional[float]
    expert_lr_mod: Optional[float]

    # MoE 路由与更新相关
    # 每层激活的专家索引 (来自 Top-K)
    active_expert_indices: Optional[Dict[int, List[int]]]
    # 经过 Surprise 筛选后，实际更新的专家索引
    updated_expert_indices: Optional[Dict[int, List[int]]]
```

### 1.2. 改造 `UpdateStrategy` 返回值

所有 `UpdateStrategy` 的 `step` 方法都将返回一个包含其策略特定指标的字典，`train` 函数会将其整合到 `StepResult` 中。

- **`StandardUpdate` / `SelectiveUpdate`**:

  - **任务**: 返回被激活的专家信息。
  - **改造**: `step` 方法需从 `all_top_indices` 计算出每层被激活的专家索引，并返回：

```python
# 返回示例
{
  "active_expert_indices": {
      0: [1, 3, 4, 7], # layer 0 激活了 1,3,4,7 号专家
      1: [0, 2, 5, 6], # layer 1 激活了 0,2,5,6 号专家
      ...
  }
}
```

- **`SurpriseMinKUpdate` (即 `Surprise-Top-Min-K`)**:

  - **任务**: 返回激活专家 和 最终更新的专家。
  - **改造**: `step` 方法在执行现有逻辑的同时，需要记录下筛选前后的专家列表，并返回：

```python
# 返回示例
{
    "active_expert_indices": { ... }, # 同上
    "updated_expert_indices": {
0: [3, 7], # layer 0 最终只更新了 3,7 号专家
1: [0, 5], # layer 1 最终只更新了 0,5 号专家
...
    }
}
```

- **`PisaUpdate`**:

  - **任务**: 返回学习率调制因子。
  - **改造**: 返回值保持不变，但结构更清晰。

```python
# 单 PISA 模式返回示例
{ "lr_mod": 0.85 }
# 双 PISA 模式返回示例
{ "gating_lr_mod": 0.9, "expert_lr_mod": 0.75 }
```

### 1.3. `train` 函数集成

`utils/training.py` 中的 `train` 函数将负责调用 `pi_monitor` 和 `update_strategy.step()`，然后将所有结果聚合到一个 `StepResult` 实例中，并将其 `append` 到一个 `List[StepResult]` 中。这样，一个 `epoch` 结束后，我们就拥有了该 epoch 内所有步骤的完整、结构化数据，可以直接传递给 `Phase 2` 的可视化函数。

## Phase 2: 可视化层重构 (Visualization)

**目标**: 重写 `utils/plotting.py`，废弃当前庞大臃肿的 `plot_metrics` 函数，代之以一系列清晰、独立的绘图函数。每个函数都将返回一个 `matplotlib.figure.Figure` 对象。

### 2.1. 核心指标图 (`plot_core_metrics`)

- **函数签名**:

  ```python
  def plot_core_metrics(
      train_steps: List[StepResult],
      val_logs: Dict[str, List[Tuple[int, float]]]
  ) -> Figure:
  ```

- **实现细节**:
  - 函数内部创建 5 个子图 (subplots)，分别对应 Loss, Accuracy, PI, Surprise, Tau。
  - **训练数据**: 遍历 `train_steps`，提取每个指标的 `(global_step, value)`，使用 `ax.scatter()` 绘制散点图。
  - **验证数据**: 遍历 `val_logs` (其 key 形如 `'CIFAR10_val_acc'`)，按数据集和指标名称分组，使用 `ax.plot()` 绘制折线图，并用标记标出数据点。
  - X 轴统一为 `Global Step`。

### 2.2. 学习率调度图 (`plot_lr_scatter`)

- **函数签名**:

  ```python
  def plot_lr_scatter(
      train_steps: List[StepResult]
  ) -> Figure:
  ```

- **实现细节**:
  - 创建一个子图。
  - 遍历 `train_steps`，提取 `lr_mod`, `gating_lr_mod`, `expert_lr_mod` 的值（非 `None` 的）。
  - 为每个系列（`base`, `gating`, `expert`）绘制不同颜色的散点图。
  - Y 轴为 `LR Modifier`，X 轴为 `Global Step`。

### 2.3. 专家激活/更新热力图 (`plot_expert_heatmap`)

- **函数签名**:

  ```python
  def plot_expert_heatmap(
      train_steps: List[StepResult],
      heatmap_type: str, # "activation" 或 "update"
      num_layers: int,
      num_experts: int
  ) -> Figure:
  ```

- **实现细节**:
  - 这是一个通用函数，用于绘制所有专家相关的热力图。
  - 根据 `heatmap_type` 参数，决定是从 `train_steps` 中提取 `active_expert_indices` 还是 `updated_expert_indices`。
  - 累加计算每个 `(layer, expert)` 的被选择/更新次数，形成一个 `(num_layers, num_experts)` 的计数矩阵。
  - 使用 `ax.imshow()` 绘制热力图，并添加颜色条和数值标签。
  - 这将取代并泛化当前的 `plot_gpil_expert_activation`。

### 2.4. 动态 K 值图 (`plot_dynamic_k_scatter`)

- **函数签名**:

  ```python
  def plot_dynamic_k_scatter(
      train_steps: List[StepResult] # StepResult 需扩展以包含 k 值和 task_name
  ) -> Figure:
  ```

- **实现细节 (未来)**:
  - `StepResult` 需要增加 `top_k: int`, `min_k: int`, `task_name: str` 字段。
  - 遍历 `train_steps`，按 `task_name` 对 `(global_step, k_value)` 进行分组。
  - 为每个任务（如 MNIST, CIFAR10）绘制不同颜色的 `top_k` 和 `min_k` 散点图系列。
  - Y 轴为 `K Value`，X 轴为 `Global Step`。

### 2.5. `plotting.py` 主入口

旧的 `plot_metrics` 将被一个新的主函数替代，该函数负责调用上述所有绘图函数，并将生成的 `Figure` 对象传递给日志系统（最初是保存到文件，最终是 `TensorBoard`）。

## Phase 3: TensorBoard 集成与最终工作流

**目标**: 将 Phase 1 和 Phase 2 的成果完全整合到 TensorBoard 工作流中，实现日志记录的自动化、标准化和集中化。

### 3.1. `train.py` 的改造

`train.py` 中的 `run_schedule` 函数将是整个工作流的总指挥。

1. **初始化 `SummaryWriter`**:

   - 在函数开头，根据 `output_dir` 和 `file_prefix` 创建一个唯一的 TensorBoard 日志目录，例如 `output/ViT/marathon-v3/runs/20250629T123000-...`。
   - 实例化 `writer = SummaryWriter(log_dir=...)`。

2. **日志记录循环**:

   - **训练 Step**: 在 `train` 函数的每个 `global_step` 结束时，拿到 `StepResult` 对象后，立即将所有标量数据写入 TensorBoard。

     ```python
     # in train() after a step is completed
     writer.add_scalar('Loss/train', step_result['loss'], global_step)
     writer.add_scalar('Accuracy/train', step_result['accuracy'], global_step)
     # ... etc for all scalars in StepResult
     ```

   - **验证 Epoch**: 在 `validate_and_record` 函数执行后，将所有验证指标写入 TensorBoard，并使用数据集名称进行标记。

     ```python
     # in validate_and_record() after validation
     writer.add_scalar(f'Loss/val/{dataset_name}', val_loss, global_step)
     writer.add_scalar(f'Accuracy/val/{dataset_name}', val_acc, global_step)
     # ... etc
     ```

3. **图表生成与记录**:

   - 在每个**验证阶段** (`validate_and_record`) 结束后，调用 `plotting.py` 中新的主绘图函数。
   - 这个主函数会接收到目前为止所有的 `train_steps: List[StepResult]` 和 `val_logs`。
   - 它会依次调用 `plot_core_metrics`, `plot_lr_scatter`, `plot_expert_heatmap` 等函数，生成一系列 `Figure` 对象。
   - `run_schedule` 接收这些 `Figure` 对象，并使用 `writer.add_figure()` 将它们写入 TensorBoard。

     ```python
     # in run_schedule() after a validation cycle
     core_metrics_fig = plot_core_metrics(all_train_steps, val_logs)
     writer.add_figure('Charts/Core_Metrics', core_metrics_fig, global_step)

     lr_scatter_fig = plot_lr_scatter(all_train_steps)
     writer.add_figure('Charts/Learning_Rate', lr_scatter_fig, global_step)
     # ... etc for all figures
     ```

### 3.2. 最终产物与清理

- **目录结构**: 旧的 `output/.../img` 和 `output/.../log` 目录将不再生成。取而代之的是 `output/.../runs/` 目录，其中包含每个实验的 TensorBoard 日志。
- **代码清理**:
  - `utils/plotting.py` 中旧的 `plot_metrics` 函数和所有相关辅助函数将被完全删除。
  - `utils/training.py` 中所有 `print` 形式的日志将被移除或转为 `logging` 模块的 `info` 级别日志。
  - `train.py` 中处理和传递大量 `kwargs` 给 `plot_metrics` 的复杂逻辑将被移除。

## Phase 4: ΩID 静态信息熵分析工具包集成

**最终目标**: 将 PILF 框架升级为“数字心智观测站”，通过定量分析模型内部（特别是注意力头）的信息动力学，来验证 IPWT 理论的核心假说。

### 4.1. 基础设施：激活探针 (Activation Probe)

- **目标**: 在不干扰训练的前提下，精确捕获指定神经元子集（如注意力头）的激活时间序列数据。
- **实现方式**:
  1. **依赖集成**: 将 `omegaid` 工具包添加为 `pyproject.toml` 中的一个可选开发依赖 (`[tool.poetry.group.dev.dependencies]`)。
  2. **探针模块**: 在 `utils` 中创建一个新的 `probe.py`。
  3. **探针注册**: `probe` 模块需要一个注册函数，允许通过模型配置（`model_config`）指定要监控的层和头，例如 `probes: [{"layer": 2, "head": 3}, ...]`。
  4. **钩子 (Hooks)**: 利用 PyTorch 的 `register_forward_hook` 机制，在 `train.py` 的 `setup_experiment` 阶段将探针挂载到指定 `TransformerBlock` 的 `attn` 模块上。
  5. **数据落盘**: 钩子函数将在每个 `step` 捕获输入和输出激活张量，并将其暂存。在每个验证阶段结束时，将累积的激活数据 (`List[torch.Tensor]`) 以实验和时间戳命名（如 `activations_layer2_head3.pt`），保存到该实验的 `runs/.../activations/` 子目录中。

### 4.2. 离线分析脚本 (`analyze_oid.py`)

- **目标**: 创建一个独立的、可重复执行的脚本，用于处理已保存的激活数据并计算信息原子。
- **实现方式**:
  1. **脚本功能**:
     - 接收一个 TensorBoard `run` 目录作为输入。
     - 加载 `activations/` 目录下的激活数据。
     - 提供数据预处理选项（如 PCA 降维、时间/空间采样）以应对计算复杂性。
     - 调用 `omegaid.calc_phiid_multivariate_ccs` 计算核心信息原子（Synergy, Redundancy, Uniqueness）。
  2. **结果回写**: 将计算出的信息原子时间序列（例如，`synergy_vs_global_step.csv`）保存回输入的 `run` 目录中。
  3. **TensorBoard 更新**: 脚本将使用与输入 `run` 目录相同的 `log_dir` 重新打开 `SummaryWriter`，并将计算出的信息动力学曲线（Synergy, Redundancy vs. Global Step）添加回去。这使得我们可以在同一个 TensorBoard 视图中对比宏观指标（如 Loss）和微观信息动力学。

### 4.3. 关键研究问题与实验设计

集成的最终目的是回答关于模型心智的核心问题：

- **功能特化的信息指纹**: 不同专家（Expert）内部的注意力头，其 Synergy/Redundancy 模式是否存在显著差异？
- **学习的信息论过程**: 模型的学习过程是否体现为“从冗余（信息压缩）到协同（信息创造）”的转变？
- **灾难性遗忘的根源**: 当模型遗忘时，其信息协同（Synergy）是如何崩溃的？
- **注意力头的分工**: 同一层内的不同注意力头，是否扮演着不同的信息处理角色（例如，某些头负责冗余提取，另一些负责协同整合）？
