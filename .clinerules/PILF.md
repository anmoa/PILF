# 预测完整性学习框架（Predictive Integrity Learning Framework, PILF）

> 不仅要训练你的模型，更要理解它的心智。

核心理念: 一个旨在将固定的超参数（如学习率、模型容量）转变为由数据内在“惊奇度”(`Surprise`)实时驱动的动态策略的认知学习框架。其本质是一种自适应超参数调度算法，它让模型根据学习内容的价值，自主决定“学多少”和“用多大容量学”。该框架源于 IPWT (Integrated Predictive Workspace Theory) 理论，相关论文信息请见 <https://github.com/dmf-archive/IPWT。>

---

## 1. 设计哲学：从“固定规则”到“动态策略”

传统训练范式依赖于手动设定的、在整个训练过程中通常固定或按预定计划衰减的超参数（如学习率）。这种“一刀切”的方法忽略了不同数据批次所包含的学习价值的巨大差异。

PILF 的设计哲学是：**用动态的、数据驱动的策略取代静态的、人为设定的规则**。

它不再盲目地使用固定的学习率或固定的模型容量，而是通过实时评估每一批次数据带来的 `Surprise`，动态地、按比例地调整其学习行为：

1. **动态学习率 (Dynamic Learning Rate)**: 当 `Surprise` 适中时，意味着遇到了有价值的“可学习区”信息，系统会分配较高的学习率；当 `Surprise` 过低（冗余信息）或过高（异常信息）时，则分配接近于零的学习率，从而自然地实现了“忽略”和“拒绝”的效果。**这直接取代了手动设定的学习率调度器**。
2. **动态容量 (Dynamic Capacity)**: 在 MoE 架构中，`Surprise` 不仅调节学习率，还决定了需要激活的“专家”数量 `k`。简单的任务 (`Surprise` 低) 只需少数专家，而复杂的任务 (`Surprise` 高) 则会动态调动更多专家参与。**这取代了固定的 Top-K 路由**。

---

## 2. 核心实现：PILF 的演进阶段

### 阶段一：PILR-S (预测完整性驱动的学习率调度器)

PILR-S 是 PILF 思想在**任何标准神经网络**上的直接应用。它只关注一个问题：**如何根据 `Surprise` 动态调整学习率？** 这是通过核心计算工具包 [SigmaPI](https://github.com/dmf-archive/SigmaPI) 实现的，该项目是一个必要的依赖项。PILF 的测试框架和实验详情见第 3 节。

它取代了传统的 `optimizer.step()` 是否执行的“门控”逻辑，演变为一个平滑的、连续的学习率调制器。

```mermaid
sequenceDiagram
    participant Trainer
    participant Model
    participant SigmaPI_Monitor
    participant LRScheduler as PILR-S
    participant Optimizer

    Trainer->>Model: 前向传播 (Feedforward)
    Model-->>Trainer: 返回 logits

    Trainer->>SigmaPI_Monitor: calculate(model, logits)
    SigmaPI_Monitor-->>Trainer: 返回 pi_metrics (含 Surprise)

    Trainer->>LRScheduler: update(Surprise)
    activate LRScheduler
    LRScheduler->>LRScheduler: lr_modifier = gaussian(Surprise, EMA, std)
    LRScheduler-->>Trainer: 返回 lr_modifier
    deactivate LRScheduler

    Trainer->>Trainer: 计算 loss & loss.backward()
    
    Trainer->>Optimizer: 设置 effective_lr = base_lr * lr_modifier
    Trainer->>Optimizer: step()
    Trainer->>Optimizer: 恢复 base_lr
```

**机制详解:**

1. **`Surprise` 计算**: 目前我们使用反向传播梯度范数来计算，但未来，完全可以考虑将Forward Forward累积梯度作为surprise的来源。这个过程无需等待昂贵的反向传播，实现了对学习价值的快速评估。
2. **动态调制**: PILR-S 模块接收 `Surprise`，并根据其与 `Surprise` 的指数移动平均（EMA）和标准差（std）的关系，通过一个高斯函数 `exp(-0.5 * ((surprise - mu) / sigma)^2)` 计算出一个平滑的调制因子 `lr_modifier` (范围在0到1之间)。
3. **权重更新**: 在计算出 `lr_modifier` 后，才执行标准的 `loss.backward()`。随后，`optimizer` 使用 `effective_lr = base_lr * lr_modifier` 来执行权重更新。`optimizer.step()` **总是被执行**，但其更新的幅度已被 `Surprise` 预先动态缩放。

### 阶段二：PIL-MoE (预测完整性驱动的 MoE - 静态 Top-K)

PILF 是在 MoE 架构上的完全实现，它将动态调度思想扩展到了模型容量的分配上。

```mermaid
graph TD
    Input --> InitialSurprise["Initial Surprise Assessment"]
    
    subgraph DynamicPolicy [Surprise-Driven Dynamic Policy]
        direction LR
        InitialSurprise -- "g(Surprise)" --> k_Value["k = g(S)"]
        InitialSurprise -- "f(Surprise)" --> lr_mod_Value["lr_mod = f(S)"]
    end

    k_Value --> HierarchicalGatingNetwork["Hierarchical Gating (route to k experts)"]
    HierarchicalGatingNetwork --> MicroExpertPool[...]
    
    MicroExpertPool --> Aggregator
    Aggregator --> Logits

    Logits --> LossCalculation
    LossCalculation -- Gradients --> SelectiveUpdate

    subgraph SelectiveUpdate [Selective Update Module]
        direction LR
        lr_mod_Value --> SetLR["Set effective_lr"]
        SetLR --> OptimizerStep["Optimizer.step()"]
    end
    
    OptimizerStep -- Updates only active experts & gating --> FinalModel
```

**训练循环详解:**

1. **双重动态决策**: 模型接收数据，计算一个初始 `Surprise`。基于此 `Surprise`，PILF 并行做出两个决策：
    - **容量决策**: `k = g(Surprise)`，决定激活多少专家。
    - **学习率决策**: `lr_modifier = f(Surprise)`，决定学习的强度。
2. **动态路由与计算**: 门控网络根据 `k` 值将任务路由到最合适的专家。
3. **动态权重更新**: 计算损失和梯度后，优化器使用由 `lr_modifier` 调制后的有效学习率，**仅对被激活的专家和门控网络**进行更新。

---

## 3. 模型动物园与实验

我们的测试套件现在围绕一个轻量级（约1M参数）的 Vision Transformer 架构构建，以便于快速进行认知学习原理的实验。我们在 CIFAR-10 数据集上比较了三种主要变体，并使用 SVHN 作为分布外（OOD）验证集。

目标是观察不同学习策略在资源受限下的表现，从而更清晰地展示 PILR-S（Predictive Integrity Learning Rate Scheduler）等机制的优势。

| **基线 ViT** | **4x1 MoE-ViT** | **16x4 MoE-ViT** | **带有 3σ 学习的 16x4 PILR-S-MoE-ViT** |
| :---: | :---: | :---: | :---: |
| ~0.81M | ~1.21M | ~1.23M | ~1.23M |
| <img src="output/ViT/legacy_img/20250626-BASE_ViT-Params_0.81M.png" style="max-width:200px;"> | <img src="output/ViT/legacy_img/20250626-MOE_4x1_ViT-Params_1.21M.png" style="max-width:200px;"> | <img src="output/ViT/legacy_img/20250626-MOE_16x4_ViT-Params_1.23M.png" style="max-width:200px;"> | <img src="output/ViT/legacy_img/20250626-GBP_MOE_ViT-Params_1.23M.png" style="max-width:200px;"> |

### MNIST 间隔复习实验

我们还在 MNIST 和 FashionMNIST 数据集上进行了间隔复习实验，以进一步探索持续学习的能力。

| **8x2 全程 (FashionMNIST -> MNIST)** | **8x2 预训练 + 8x2 PILR-S 间隔复习 (FashionMNIST -> MNIST)** | **8x2 PILR-S 全程 (FashionMNIST -> MNIST) (1.2σ)** |
| :---: | :---: | :---: |
| ~0.26M | ~0.26M | ~0.26M |
| <img src="output/ViT/20250627-tiny-moe-mnist-mnist-rehearsal.png" style="max-width:200px;"> | <img src="output/ViT/20250627-tiny-gbp-mnist-mnist-rehearsal.png" style="max-width:200px;"> | <img src="output/ViT/legacy_img/20250627-gbp-moe-vit-1-2-sigma-rehearsal.png" style="max-width:200px;"> |

---

## 4. 安装与使用

本项目依赖 `sigma-pi` 包进行核心计算。要复现实验并使用完整的测试框架，您必须首先克隆本仓库。

```bash
git clone https://github.com/dmf-archive/PILF.git
cd PILF
```

**注意:** 本包不会自动安装 PyTorch。请在继续之前，为您的系统（CPU 或 CUDA）手动安装合适的版本。对于支持 CUDA 的系统，建议使用 `uv` 或 `pip` 安装：

```bash
# CUDA 12.1 示例
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

设置好 PyTorch 后，安装测试框架的依赖项：

```bash
pip install -e .[dev]
```

测试框架是模块化和配置驱动的。

### 4.1. 配置您的实验

在 `test/configs/` 目录中创建或修改一个配置文件。例如, `test/configs/base_vit.py`:

```python
# test/configs/base_vit.py

# 模型参数
model_config = {
    'model_type': 'base',
    'embed_dim': 128,
    'depth': 6,
    # ... 其他模型参数
}

# 训练参数
train_config = {
    'epochs': 20,
    'batch_size': 256,
    # ... 其他训练参数
}
```

### 4.2. 运行实验

从根目录使用 `test/run_experiment.py` 脚本启动实验：

```bash
python test/run_experiment.py --config test/configs/base_vit.py
```

要运行其他变体，只需指向它们各自的配置文件：

```bash
# 运行 MoE-ViT 实验
python test/run_experiment.py --config test/configs/moe_vit.py

# 运行 PILR-S-MoE-ViT 实验
python test/run_experiment.py --config test/configs/gbp_moe_vit.py
```

---

## 5. 理论贡献

- **变超参数为策略**: 将学习率和模型容量从开发者设定的“静态超参数”转变为模型根据数据价值自主调节的“动态策略”。
- **统一“学习”与“遗忘”**: 通过将学习率与 `Surprise` 挂钩，PILF 提供了一个统一的框架来处理学习、忽略（低`Surprise`导致低`lr`）和拒绝（高`Surprise`导致低`lr`），从而内在地缓解了灾难性遗忘。
- **计算资源按需分配**: (PILF) 实现了真正的按需计算，简单的任务消耗极少资源，复杂的任务则动态调用更多资源，极大提升了效率。

---

## 6. 演进路径

PILF 的演进分为四个主要阶段，每个阶段都建立在前一阶段的基础上，逐步实现更高级的自适应能力：

- **阶段零：GBP (门控反向传播)**
  - **目标:** 在传统训练中，通过门控机制选择性地更新权重，以缓解灾难性遗忘。
  - **核心机制:** `optimizer.step()` 的执行由一个二元门控信号控制，该信号基于`Surprise`指标。
  - **优势:** 作为 PILF 的前身，验证了基于 PI 进行选择性学习的有效性，为后续的动态学习率和容量调度奠定了基础。

- **阶段一：PILR-S (Predictive Integrity Learning Rate Scheduler)**
  - **目标:** 在任何标准神经网络上，用 `Surprise` 驱动的动态学习率取代固定的学习率调度器。
  - **核心机制:** `effective_lr = base_lr * f(Surprise)`。学习率的调整完全由数据批次的“惊奇度”决定。
  - **优势:** 无需修改模型架构，可作为现有训练流程的直接替代品，快速验证动态策略的有效性。

- **阶段二：PIL-MoE (Predictive Integrity-driven Mixture of Experts - 静态 Top-K)** (当前阶段)
  - **目标:** 将 PILR-S 的动态学习率机制引入 MoE 架构，并结合静态 Top-K 硬路由，同时只更新激活的专家权重。
  - **核心机制:** `effective_lr = base_lr * f(Surprise)` 应用于 MoE 架构。门控网络根据静态 Top-K 值将任务路由到专家，且仅更新被激活的专家权重。
  - **优势:** 在 MoE 架构中引入了数据驱动的学习率，同时通过选择性更新提高了训练效率，并为后续的动态容量分配奠定基础。

- **阶段三：PILD-MoE (Predictive Integrity-driven Dynamic Mixture of Experts)**
  - **目标:** 实现一个完全自适应的认知系统，其中 `Surprise` 不仅调节学习率，还动态缩放激活的专家数量 `k`。
  - **核心机制:** `k = g(Surprise)` 和 `effective_lr = base_lr * f(Surprise)` 并行运作。模型根据数据复杂性动态调整激活的专家数量和学习强度。
  - **优势:** 实现了计算效率和模型容量扩展性的最大化，真正实现了计算资源按需分配。