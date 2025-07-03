# 待办事项清单

## Epic 0: 简化和整理代码

- [x] **将专家网络从 PILR 调制中移除**：专家网络将使用其默认学习率，并由 SMK 机制管理。通过将 `expert_initial_var` 设置为大值（例如1000.0）来实现“软禁用”PILR调制。
- [x] **预测完整性 (PI) 计算系统重构**：
  - **目标**：实现局域PI计算和全局PI汇总，并共享`alpha`和`gamma`超参数，以更好地反映分层BP和FP的动态。
  - [x] **子任务1：修改`utils/pi_calculator.类`**
    - [x] **引入`LocalPiCalculator`类**：
      - 职责：计算单个模块的局域PI组成部分（`epsilon`, `tau`, `surprise`）。
      - 输入：模块的梯度、损失和logits。
      - `alpha`和`gamma`：由外部传入，不再是其成员变量。
    - [x] **重命名`PiCalculator`为`GlobalPiCalculator`**：
      - 职责：汇总所有局域PI组成部分，计算最终的全局PI。
      - `__init__`：只接收`alpha`和`gamma`。
      - `calculate`：接收包含所有局域PI组成部分的列表（或字典），进行汇总计算。
  - [x] **子任务2：修改`utils/training.py`的`Trainer`类**
    - [x] **实例化PI计算器**：在`__init__`中实例化`GlobalPiCalculator`。
    - [x] **调整`train_one_epoch`**：
      - 在门控网络和专家网络各自的反向传播之后，分别计算其局域的`surprise`、`epsilon`和`tau`。
      - 收集这些局域PI组成部分。
      - 在每个step结束时，调用`GlobalPiCalculator`的`calculate`方法，计算全局PI。
      - 将全局PI传递给`PILRStrategy`。
    - [x] **调整`validate`方法**：在验证阶段也进行局域PI的计算和汇总。
  - [x] **子任务3：修改`utils/strategies/learning_rate_strategies.py`的`PILRStrategy`类**
    - [x] **简化`_apply_dual`**：`PILRAdaptor`不再需要独立的`alpha`参数。
    - [x] `PILRStrategy`从`Trainer`接收全局PI，并根据需要提取或计算适用于门控和专家的`surprise`值，传递给各自的`PILRAdaptor`。
  - [x] **子任务4：更新`train.py`**
    - [x] 调整`GlobalPiCalculator`的实例化方式。
- [x] **将 `mlp_dim` 转换为 `mlp_ratio` 派生参数**：
  - 在模型配置中引入 `mlp_ratio`，并从 `model_config['embed_dim'] * model_config['mlp_ratio']` 派生 `mlp_dim`。
- [x] **移除未使用的 `ood_inhibition_c` 参数**：
  - 从 `models/gaussian_moe.py` 中的 `GaussianMoELayer` 的 `__init__` 方法中移除 `ood_inhibition_c` 参数及其相关代码。

## Epic 1: 门控网络学习机制优化

**目标：** 优化门控网络的学习机制，使其能够从 `top_k` 和 `min_k` 专家分布的差异中学习，从而更有效地指导专家选择。

## 1.1. 门控损失函数设计与实现

- [x] **修改 `utils/gating_loss.py`**:
  - [x] **引入 `TopKMinKLoss` 类**:
    - 继承 `nn.Module`。
    - 接收 `log_probs` (所有专家的对数概率)、`top_k_indices` (当前批次 `top_k` 专家的索引) 和 `min_k_expert_indices` (由 `SurpriseMinKStrategy` 确定的 `min_k` 专家的索引) 作为输入。
    - 计算 `top_k` 专家分布与 `min_k` 专家分布之间的差异。这可以通过多种方式实现，例如：
      - 计算两个分布的 KL 散度。
      - 计算两个分布的交叉熵。
      - 惩罚 `top_k` 专家中不在 `min_k` 专家集合中的专家，或者奖励 `min_k` 专家中被 `top_k` 选中的专家。
    - 返回一个标量损失值。
- [x] **修改 `utils/training.py` 的 `Trainer` 类**:
  - [x] **实例化 `TopKMinKLoss`**: 在 `__init__` 方法中实例化新的损失函数。
  - [x] **调整 `_compute_gating_loss` 方法**:
    - 接收 `all_routing_info` (包含 `log_probs` 和 `top_indices`) 和 `min_k_expert_indices_per_layer`。
    - 调用 `TopKMinKLoss` 计算门控损失。
    - 确保梯度正确反向传播到门控参数。

## Epic 1.5: 惊奇度指标优化

**目标：** 优化惊奇度指标的命名和记录方式。

- [ ] **将 `tqdm` 的 `surprise` 修改为 `router_surprise`**。
- [ ] **将 `router_surprise` 加入记录系统**。
- [ ] **`epoch log` 不再打印 `global avg surprise` 而是打印 `router avg surprise`，只在 `tensorboard` 中记录**。

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
