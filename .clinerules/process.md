# PILF 日志与可视化 V2 重构计划

本计划旨在用 TensorBoard 替换现有的日志和绘图系统，以实现更强大、自动化的实验跟踪。

## 已完成事项 (V2 日志与可视化)

- [x] **V2 日志与可视化系统**: 已完成 TensorBoard 集成和配置文件系统重构 V4。
- [x] **V5 框架重构与解耦**: 已完成核心组件分离、Trainer 类重构和 LGO 解耦路由训练。
- [x] **`surprise` 计算与 PI 兼容性修复**: 调整了梯度裁剪顺序，并创建了本地 `PiCalculator` 以适应分层 BP。
- [x] **日志与绘图功能增强**: 实现了 `TensorBoardLogger` 和 `calculate_gating_selection_accuracy` 的模块化，并在训练结束时生成专家激活和门控路由热力图及散点图。

## Epic 5: 集成 ΩID 信息动力学分析 (The ΩID Toolkit)

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
    - 可以生成一个矩阵图（heatmap），行和列都是专家索引，单元格的颜色表示特定信息原子（如协同 `S` 或冗余 `R`）的强度。
    - 将图表保存到 `run-dir/oid_plots/{probe_name}/` 目录中。

