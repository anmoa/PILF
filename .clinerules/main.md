# PILF: Predictive Integrity Learning Framework

我们的核心目标是构建一个先进的、受认知科学启发的模型训练与分析框架。更多信息可查看 [PILF.md](./PILF.md)

## 2. 项目结构

```plain
PILF/
├── configs/      # 存放所有模型的架构配置文件
├── schedules/    # 存放所有实验的调度配置文件
├── models/       # 模型定义的包
│   ├── __init__.py
│   ├── base_vit.py
│   ├── moe_vit.py
│   └── gpil_moe.py
├── utils/        # 训练、验证、绘图等辅助工具
│   ├── __init__.py
│   ├── plotting.py
│   ├── strategies.py
│   └── training.py
└── train.py      # 由调度+架构驱动的运行器
```

## 3. 工程准则

- **向后兼容**: 所有代码修改都应保持向后兼容性，允许旧的实验配置和脚本继续运行。
- **配置驱动**: 实验的超参数（如数据集、模型类型、PILR-S 模式）应通过配置文件或命令行参数指定，而不是硬编码在脚本中。
- **代码风格**: 代码必须简洁、自解释，移除所有不必要的注释。
- **环境管理**: **必须**使用 `uv` 管理独立的虚拟环境，并通过 `uv add` 命令将所有依赖项添加到 `pyproject.toml`。
- **纯函数原则**: 所有数据处理函数都应为纯函数。

## 5. 实验脚本参数速记

为了方便快速启动不同类型的实验，所有实验都通过唯一的 `train.py` 脚本启动，该脚本由一个**调度文件**和一个**模型配置文件**共同驱动：

| 脚本 | 主要目的 | 示例命令 |
| :--- | :--- | :--- |
| `train.py` | 运行所有类型的实验 | `python train.py --schedule <schedule_path> --model-config <model_config_path>` |
| `train.py` | 运行马拉松复习实验 | `python train.py --schedule schedules/marathon_v1.py --model-config configs/large_pilr_mnist.py` |
