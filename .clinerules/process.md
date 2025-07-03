# 待办事项清单

## Epic 0: 简化和整理代码 (已完成)

- [x] **将专家网络从 PILR 调制中移除**
- [x] **预测完整性 (PI) 计算系统重构**：实现局域 PI 计算和全局 PI 汇总，并共享`alpha`和`gamma`超参数。
- [x] **将 `mlp_dim` 转换为 `mlp_ratio` 派生参数**
- [x] **移除未使用的 `ood_inhibition_c` 参数**
- [x] **门控损失函数设计与实现**：引入 `TopKMinKLoss` 类，并修改 `Trainer` 类以计算和应用门控损失。
- [x] **将 `tqdm` 的 `surprise` 修改为 `router_surprise`**。
- [x] **将 `router_surprise` 加入记录系统**。
- [x] **`epoch log` 不再打印 `global avg surprise` 而是打印 `router avg surprise`，只在 `tensorboard` 中记录**。

## Epic 2: 集成 ΩID 信息动力学分析 (The ΩID Toolkit)

**目标**: 为框架增加“神经元级别”的观测能力，通过捕获激活数据并进行离线信息论分析，验证核心理论假说。

- [x] **添加依赖**: 在 `pyproject.toml` 的 `[project.optional-dependencies]` 中添加 `dev` 依赖，并包含 `omegaid`。为了获得最佳性能，应指明需要 CUDA 支持，例如 `"omegaid[cuda-12x]"`。
- [ ] **创建 `utils/probe.py`**:

  - [ ] **探针配置**: 在模型配置文件中定义一个 `probes` 列表，每个探针指定 `name`, `source_module`, `target_module`。
  - [ ] **数据捕获**: 实现一个 `ActivationProbe` 类，使用 `forward_hook` 捕获指定模块的输出。
    - 探针应能捕获指定模块的激活，并**累计每个专家的激活次数**。
    - 探针应处理源 (`source_modules`) 和目标 (`target_modules`) 模块列表，允许更灵活的 M-to-N 信息分析。
  - [ ] **数据存储与格式**: 在每个符合条件的验证周期结束时，将捕获的激活数据及元数据打包成字典，使用 `torch.save` 保存。

    - **路径与命名**: 文件保存在 `runs/{run_id}/probes/{probe_name}/epoch_{epoch_num}.pt.gz`。
    - **数据结构**: 保存的字典应包含：

      ```python
      {
          'metadata': {
              'probe_name': str,
              'epoch': int,
              'source_modules': list[str],
              'target_modules': list[str],
              'dataset_name': str, # e.g., 'CIFAR10_val'
              'expert_activation_counts': dict[str, dict[int, int]], # {module_name: {expert_idx: count}}
          },
          'source_activations': dict[str, torch.Tensor], # key is module name
          'target_activations': dict[str, torch.Tensor], # key is module name
          'labels': torch.Tensor, # Optional but useful for conditional analysis
      }
      ```

    - **压缩**: 文件应使用 `gzip` 进行压缩，以显著减小磁盘占用。

  - [ ] **存储策略与优化**: 为避免磁盘空间爆炸，探针应支持灵活的存储策略：
    - **采样频率**: `save_every_n_epochs: int`，仅每 N 个周期保存一次数据。
    - **数据子集**: `validation_subset_size: Optional[int]`，仅从验证集中随机抽取固定大小的子集进行捕获。

- [ ] **改造 `train.py`**:
  - [ ] **探针初始化与注册**: 在实验设置阶段，根据模型配置中的 `probes` 列表，实例化并注册 `ActivationProbe`。
  - [ ] **触发保存**: 在 `validate_one_epoch` 函数之后，根据探针配置的 `save_every_n_epochs` 决定是否调用探针的保存方法。
- [ ] **创建 `analyze_oid.py`**:
  - [ ] **脚本接口**: 脚本接收 `--run-dir` 和可选的 `--probe-name`。增加 `--top-n-experts` 参数（默认为 4），用于指定分析最活跃的 N 个专家。
  - [ ] **数据加载与预处理**:
    - 脚本遍历指定探针目录，使用 `gzip` 解压并用 `torch.load` 加载每个 `epoch_*.pt.gz` 文件。
    - 从元数据中读取 `expert_activation_counts`。
  - [ ] **核心计算 (动态专家对)**:
    - **自动选择专家**: 根据 `expert_activation_counts`，识别出 `--top-n-experts` 个最活跃的专家。
    - **两两组合**: 对这 N 个专家进行两两组合，形成分析对 `(expert_i, expert_j)`。
    - **信息分解**: 对每一对组合，提取其激活张量作为 `s1` 和 `s2`，并调用 `omegaid` 进行信息分解。
  - [ ] **结果可视化**:
    - 对每个 epoch，为所有分析的专家对生成信息原子图表。
    - 可以生成一个矩阵图（heatmap），行和列都是专家索引，单元格的颜色表示特定信息原子（如协同 `Syn` 或冗余 `Red`）的强度。
    - 将图表保存到 `run-dir/oid_plots/{probe_name}/` 目录中。

## Epic 3: 动态 Schedules

**目标：** 允许模型根据 PI 分数自行安排学习规划，直到将所有任务的 PI 都最大化为止。模型将回顾过去学习周期中的 ΔACC（准确率变化）和 ΔPI（预测完整性变化），以选择效率最高的学习路径，并确保在完成当前任务的 Grokking 后，能够自主转向复习其他任务。

### 3.1. 任务状态跟踪与评估

- [ ] **修改 `utils/types.py`**:
  - 引入 `TaskMetrics` TypedDict，用于存储每个任务的 `avg_pi`, `avg_accuracy`, `delta_pi`, `delta_accuracy` 等历史指标。
- [ ] **修改 `utils/training.py` 的 `Trainer` 类**:
  - 在 `__init__` 中添加 `self.task_history: Dict[str, List[TaskMetrics]]` 用于存储每个任务的历史指标。
  - 在 `validate` 方法中，计算每个验证数据集的 `delta_pi` 和 `delta_accuracy`，并更新 `self.task_history`。
  - 确保 `validate` 方法返回的 `ValidationResult` 包含 `delta_pi` 和 `delta_accuracy`。

### 3.2. 动态调度器实现

- [ ] **创建 `utils/dynamic_scheduler.py`**:
  - 引入 `DynamicScheduler` 类。
  - `__init__` 接收 `config` 和 `task_history`。
  - `select_next_task(current_task: str) -> str`: 根据 `task_history` 中的 ΔACC 和 ΔPI，以及 Grokking 状态，选择下一个要训练的任务。
    - **Grokking 检测逻辑**: 定义 Grokking 的判断标准（例如，PI 达到阈值且准确率稳定在高位）。
    - **任务优先级逻辑**: 根据 ΔACC 和 ΔPI 评估任务的学习效率，优先选择需要改进或效率最高的任务。
    - **复习机制**: 如果当前任务已 Grokking，则从其他未 Grokking 的任务中选择一个进行复习。
  - `get_all_tasks() -> List[str]`: 从 `config.schedule['tasks']` 中提取所有训练任务名称。

### 3.3. `train.py` 集成动态调度

- [ ] **修改 `train.py` 的 `run_schedule` 函数**:
  - 实例化 `DynamicScheduler`。
  - 移除原有的静态 `config.schedule['tasks']` 循环。
  - 在主训练循环中，每次迭代开始时，调用 `dynamic_scheduler.select_next_task()` 来获取当前要训练的任务。
  - 调整训练和验证逻辑，以适应动态任务选择。
  - 确保 `Trainer` 能够访问 `DynamicScheduler` 或其相关信息，以便在 `validate` 阶段更新任务历史。

### 3.4. 配置更新

- [ ] **修改 `utils/config.py`**:
  - 允许在模型配置或调度配置中定义动态调度相关的参数（例如，Grokking 阈值，任务选择策略参数）。

## Epic 4: 实现 GenGaussMoE (已完成)

- [x] **定义 `GenGaussianMoELayer`**: 继承 `GaussianMoELayer`，引入叙事生成器模块。
- [x] **实现“叙事生成器”模块**: 包含任务特征提取器、Transformer 调度与分析模块、VAE 核心。
- [x] **集成“叙事生成器”到训练流程**: `Trainer` 已修改以传递 `pi_score`。
- [x] **更新 `models/__init__.py`**: 添加 `GenGaussianMoEVisionTransformer` 到 `model_map`。
- [x] **创建 `configs/large_gen_gauss_moe_smk_pilr_d.py`**: 新增模型配置文件。
- [x] **通过静态检查**: `ruff check` 和 `mypy` 已通过。
- [x] **解决维度硬编码问题**: `narrative_generator_dim` 和 `vae_latent_dim` 的计算确保确定性，并根据 `GenGaussianMoELayer` 的 `in_features`（即 `embed_dim`）动态计算，同时确保 `narrative_generator_dim + 1` 可被 `nhead` 整除，且整个特征提取与记忆回放系统的参数量控制在主模型参数量的 1/10 以内。
- [x] **优化训练速度**: 解决训练速度慢的问题。
