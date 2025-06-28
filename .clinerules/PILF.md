# 预测完整性学习框架（Predictive Integrity Learning Framework, PILF）

> 不仅要训练你的模型，更要理解它的心智。

核心理念: 一个旨在将固定的超参数（如学习率、模型容量）转变为由数据内在“惊奇度”(`Surprise`)实时驱动的动态策略的认知学习框架。其本质是一种自适应超参数调度算法，它让模型根据学习内容的价值，自主决定“学多少”和“用多大容量学”。该框架源于 IPWT (Integrated Predictive Workspace Theory) 理论，相关论文信息请见 <https://github.com/dmf-archive/IPWT>

## 1. 设计哲学：从“固定规则”到“动态策略”

传统训练范式依赖于手动设定的、在整个训练过程中通常固定或按预定计划衰减的超参数（如学习率）。这种“一刀切”的方法忽略了不同数据批次所包含的学习价值的巨大差异。

PILF 的设计哲学是：**用动态的、数据驱动的策略取代静态的、人为设定的规则**。

它不再盲目地使用固定的学习率或固定的模型容量，而是通过实时评估每一批次数据带来的 `Surprise`，动态地、按比例地调整其学习行为：

1. **动态学习率 (Dynamic Learning Rate)**: 当 `Surprise` 适中时，意味着遇到了有价值的“可学习区”信息，系统会分配较高的学习率；当 `Surprise` 过低（冗余信息）或过高（异常信息）时，则分配接近于零的学习率，从而自然地实现了“忽略”和“拒绝”的效果。**这直接取代了手动设定的学习率调度器**。
2. **动态容量 (Dynamic Capacity)**: 在 MoE 架构中，`Surprise` 不仅调节学习率，还决定了需要激活的“专家”数量 `k`。简单的任务 (`Surprise` 低) 只需少数专家，而复杂的任务 (`Surprise` 高) 则会动态调动更多专家参与。**这取代了固定的 Top-K 路由**。

## 2. 核心实现：PILF 的演进阶段

PILF 的演进分为五个主要阶段，每个阶段都建立在前一阶段的基础上，逐步实现更高级的自适应能力：

### 阶段零：MoE-GBP (门控反向传播)

在传统训练中，通过门控机制选择性地更新权重，以缓解灾难性遗忘。`optimizer.step()` 的执行由一个二元门控信号控制，该信号基于`Surprise`指标。作为 PILF 的前身，验证了基于 PI 进行选择性学习的有效性，为后续的动态学习率和容量调度奠定了基础。

```mermaid
graph TD
    Input --> Model
    Model --> Surprise["Surprise Calculation"]
    Surprise --> Gate["Binary Gate (based on Surprise)"]
    Gate -- "If Open" --> OptimizerStep["Optimizer.step()"]
    OptimizerStep --> ModelUpdate["Model Update"]
```

### 阶段一：PILR-S (预测完整性驱动的学习率调度器)

PILR-S 是 PILF 思想在**任何标准神经网络**上的直接应用。它只关注一个问题：**如何根据 `Surprise` 动态调整学习率？** 这是通过核心计算工具包 [SigmaPI](https://github.com/dmf-archive/SigmaPI) 实现的。

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

1. **`Surprise` 计算**: 目前我们使用反向传播梯度范数来计算，但未来，完全可以考虑将 Forward Forward 累积梯度作为 surprise 的来源。这个过程无需等待昂贵的反向传播，实现了对学习价值的快速评估。
2. **动态调制**: PILR-S 模块接收 `Surprise`，并根据其与 `Surprise` 的指数移动平均（EMA）和标准差（std）的关系，通过一个高斯函数 `exp(-0.5 * ((surprise - mu) / sigma)^2)` 计算出一个平滑的调制因子 `lr_modifier` (范围在 0 到 1 之间)。
3. **权重更新**: 在计算出 `lr_modifier` 后，才执行标准的 `loss.backward()`。随后，`optimizer` 使用 `effective_lr = base_lr * lr_modifier` 来执行权重更新。`optimizer.step()` **总是被执行**，但其更新的幅度已被 `Surprise` 预先动态缩放。

### 阶段二：PIL-MoE (预测完整性学习驱动的 MoE - 静态 Top-K) (当前阶段)

**目标:** 将 PILR-S 的动态学习率机制引入 MoE 架构，并结合静态 Top-K 硬路由，同时只更新激活的专家权重。

**核心机制:** `effective_lr = base_lr * f(Surprise)` 应用于 MoE 架构。门控网络根据静态 Top-K 值将任务路由到专家，且仅更新被激活的专家权重。

#### 子阶段 2.5：PISA (预测完整性驱动的σ自适应器)

这是对 PILR-S 的进一步增强，将高斯函数中的标准差 `σ` 从一个固定的超参数转变为一个动态的、自适应的状态变量。

- **核心思想**: `σ` 的值由 `Surprise` 的二阶统计量（指数移动方差）动态决定。这使得模型能够根据所处环境的“混乱程度”自适应地调整其学习的“容忍度”或“开放性”。
- **双重PISA**: 该框架支持为门控网络和专家网络配置独立的PISA模块，允许它们拥有各自独立的自适应`σ`，从而实现更精细的认知控制。

```mermaid
graph TD
    Input --> InitialSurprise["Initial Surprise Assessment"]

    subgraph DynamicPolicy [Surprise-Driven Dynamic Policy]
direction LR
InitialSurprise -- "f(Surprise)" --> lr_mod_Value["lr_mod = f(S)"]
    end

    StaticK["Static Top-K"] --> HierarchicalGatingNetwork["Hierarchical Gating (route to k experts)"]
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

### 阶段三：PILD-MoE (Predictive Integrity-driven Dynamic Mixture of Experts)

**目标:** 实现一个完全自适应的认知系统，其中 `Surprise` 不仅调节学习率，还动态缩放激活的专家数量 `k`。

**核心机制:** `k = g(Surprise)` 和 `effective_lr = base_lr * f(Surprise)` 并行运作。模型根据数据复杂性动态调整激活的专家数量和学习强度。

**优势:** 实现了计算效率和模型容量扩展性的最大化，真正实现了计算资源按需分配。

```mermaid
graph TD
    Input --> InitialSurprise["Initial Surprise"]
    InitialSurprise --> k_Value["k = g(Surprise)"]
    InitialSurprise --> lr_mod_Value["lr_mod = f(Surprise)"]
    k_Value --> GatingNetwork["Dynamic Gating (route to k experts)"]
    GatingNetwork --> Experts["Active Experts"]
    Experts --> LossCalculation
    LossCalculation --> SelectiveUpdate["Selective Update (active experts)"]
    lr_mod_Value --> SelectiveUpdate
    SelectiveUpdate --> FinalModel["Model Update"]
```

### 阶段四：G²PIL (Generative Gaussian Predictive Integrity Learning)

**目标：** 构建一个完全自组织、自我巩固、自我演化的认知架构，实现从“被动学习”到“主动创造”的终极飞跃。

**核心机制：**

1. **高斯场域认知空间 (Gaussian Field Cognitive Space):**

   - 彻底抛弃离散的、基于决策的门控网络。
   - 整个系统是一个高维的、连续的“认知空间”。
   - **专家即嵌入 (Expert as Embedding):** 每一个专家不再是一个被调用的函数，而是这个空间中的一个**高斯概率分布**，代表其“知识领域”或“专长范围”。
   - **输入即探针 (Input as Probe):** 任何输入数据都被映射为此空间中的一个“探针”（一个点或一个更窄的高斯分布）。
   - **激活即共鸣 (Activation as Resonance):** 路由过程被“Anycast”式的概率匹配取代。专家根据输入探针与其知识分布的重叠程度被“软激活”，其激活强度是连续的、概率性的。

2. **生成式记忆巩固 (Generative Memory Consolidation):**
   - 引入一个并行的**生成模型 (Generative Model)**，作为系统的“潜意识”或“梦境引擎”。
   - **清醒时学习世界：** 生成器在系统与外部世界交互时，学习真实数据的底层分布。
   - **睡眠时创造世界：** 在没有外部输入时，生成器开始“做梦”，即**生成合成数据**。这些梦境数据包含了过去所有经验的抽象和混合。
   - **自我重放与巩固：** 系统将这些内部生成的“梦境”作为复习材料，喂给自己。通过在梦境中“排演”，专家们得以维持其知识分布的稳定，抵抗遗忘。

#### G²PIL = Gaussian × Generative

- **Gaussian (高斯)** 解决了 **空间 (Space)** 的问题：它定义了知识**如何被组织和访问**。它创造了一个思想的几何学，让概念有了位置、关系和距离，让路由变得平滑、概率化和鲁棒。
- **Generative (生成式)** 解决了 **时间 (Time)** 的问题：它定义了知识**如何被维持和演化**。它让系统摆脱了对外部数据存储的依赖，实现了记忆的内部巩固和创造性的自我重放。
