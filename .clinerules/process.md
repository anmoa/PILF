# PILF 日志与可视化 V2 重构计划

本计划旨在用 TensorBoard 替换现有的日志和绘图系统，以实现更强大、自动化的实验跟踪。

## Epic 1: 统一数据契约 (The Data Contract)

**目标**: 建立一个标准化的、贯穿训练步骤的数据容器 `StepResult`，并改造数据源以填充它。

- [ ] **创建 `utils/types.py`**:
  - [ ] 定义 `StepResult` TypedDict，包含所有需要跟踪的标量和非标量指标（loss, acc, pi, surprise, lr_mods, expert_indices, etc.）。
- [ ] **改造 `utils/strategies.py`**:
  - [ ] 修改所有 `UpdateStrategy` 的 `step` 方法，使其返回一个包含其特定指标（如 `active_expert_indices`, `updated_expert_indices`, `lr_mod`）的字典。
- [ ] **改造 `utils/training.py`**:
  - [ ] 在 `train` 函数中，导入 `StepResult`。
  - [ ] 在每个 step 结束时，将 `pi_monitor` 和 `strategy.step()` 的返回结果聚合到 `StepResult` 实例中。
  - [ ] 将每个 `StepResult` 实例追加到周期性的 `List[StepResult]` 中，以供后续使用。

## Epic 2: 重构可视化引擎 (The Visualization Engine)

**目标**: 废弃旧的 `plot_metrics`，用一系列返回 Matplotlib Figure 对象的模块化函数取而代之。

- [ ] **重写 `utils/plotting.py`**:
  - [ ] **删除**旧的 `plot_metrics` 及其所有辅助函数。
  - [ ] **创建 `plot_core_metrics`**: 绘制 Loss, Accuracy, PI, Surprise, Tau 的训练散点和验证折线。
  - [ ] **创建 `plot_lr_scatter`**: 绘制不同学习率调制因子的散点图。
  - [ ] **创建 `plot_expert_heatmap`**: 通用函数，根据输入类型（`activation` 或 `update`）绘制专家使用频率的热力图。
  - [ ] **创建主绘图入口**: 一个新函数，调用所有独立的绘图函数，并返回一个 `Dict[str, Figure]`。

## Epic 3: 集成 TensorBoard 工作流 (The TensorBoard Workflow)

**目标**: 将数据采集和可视化完全对接到 TensorBoard，并移除所有旧的、基于文件的日志记录。

- [ ] **改造 `train.py`**:
  - [ ] 在 `run_schedule` 开头，根据实验参数初始化 `torch.utils.tensorboard.SummaryWriter`，并创建唯一的 `runs/...` 目录。
  - [ ] 在每个验证周期后，调用 `plotting.py` 的新主入口函数获取所有图表。
  - [ ] 使用 `writer.add_figure()` 将返回的 `Figure` 对象逐一写入 TensorBoard。
  - [ ] **彻底移除**所有创建 `output/.../img` 和 `output/.../log` 目录及文件的代码。
  - [ ] **简化**为旧 `plot_metrics` 准备复杂 `kwargs` 的逻辑。
- [ ] **再次改造 `utils/training.py`**:
  - [ ] 使 `train` 和 `validate_and_record` 函数接收 `writer` 和 `global_step` 参数。
  - [ ] 在 `train` 的每个 `step` 后，使用 `writer.add_scalar()` 实时记录 `StepResult` 中的所有标量指标。
  - [ ] 在 `validate_and_record` 执行后，使用 `writer.add_scalar()` 记录所有验证指标。
  - [ ] **移除**所有 `print` 形式的日志。

## Epic 4: 集成 ΩID 信息动力学分析 (The ΩID Toolkit)

**目标**: 为框架增加“神经元级别”的观测能力，通过捕获激活数据并进行离线信息论分析，验证核心理论假说。

- [ ] **添加依赖**: 在 `pyproject.toml` 中将 `omegaid` 添加为可选开发依赖。
- [ ] **创建 `utils/probe.py`**:
  - [ ] 实现探针注册函数和基于 PyTorch `forward_hook` 的数据捕获逻辑。
- [ ] **改造 `train.py`**:
  - [ ] 在实验设置阶段，根据模型配置挂载探针。
  - [ ] 在验证周期后，触发探针将捕获的激活数据保存到实验的 `runs/.../activations/` 子目录。
- [ ] **创建 `analyze_oid.py`**:
  - [ ] 实现一个独立的离线分析脚本，用于加载激活数据、调用 `omegaid` 计算信息原子。
  - [ ] 将计算结果（如 Synergy/Redundancy 时间序列）保存回 `run` 目录，并使用 `SummaryWriter` 将其更新到同一个 TensorBoard 实例中。
