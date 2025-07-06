# PILF-2 重构计划 V3: 动态高斯路由

## 1. 核心目标与设计哲学

- **最终目标**: 实现一个强大的、能够抵抗灾难性遗忘的持续学习模型。
- **核心机制**: 通过**动态高斯路由**实现专家的自动分化和知识分组。高斯路由基于距离的几何归纳偏置是实现这一目标的基础，是必需品而非可选项。
- **架构原则**:
  - **路由**: `Per-Token` 执行，但其上下文来自 `Per-Subject` 的检索，且路由机制为动态高斯。
  - **优化**: `Per-Batch` 执行，以保证效率。
  - **策展**: `Per-Subject` 执行，以保证经验的质量。

---

## 2. 实施步骤

### 步骤一：[模型层] 重构 `GatingTransformer` 和 `MoE Layer`

- **文件**: `models/gating_transformer.py` & `models/moe_layers.py`
- **任务**:
    1. **修改 `GatingTransformer`**:
        - 修改 `out_proj` 线性层，使其输出维度足以参数化所有专家的 `μ` 和 `log_sigma` (即 `num_experts * 2 * embed_dim`)。
        - `forward` 方法的输出需被 `reshape`，以便清晰地分离出每个专家的 `μ` 和 `log_sigma`。
    2. **重构 `MemoryGaussianMoELayer.forward`**:
        a.  **上下文检索 (Per-Subject)**: 保留 `q_subject = x.mean(dim=1)`，其**唯一目的**是用于从经验缓存区中检索与整个 `Subject` 相关的历史上下文 `retrieved_history`。
        b.  **构建 Per-Token 输入**: 将 `retrieved_history` 与 `x_token` 拼接，为 `GatingTransformer` 构建输入序列。
        c.  **动态参数生成 (Per-Token)**: 调用 `gating_transformer`，为每个 `token` 动态生成一组高斯分布参数 `(dynamic_mus, dynamic_log_sigmas)`。
        d.  **动态高斯路由 (Per-Token)**: 使用生成的参数创建动态高斯分布，并计算原始 `x_token` 在该分布下的对数概率 `log_probs` 作为最终的路由分数。

### 步骤二：[流程与策略层] 维持混合粒度优化

- **文件**: `utils/train_loops/pilf_train_loop.py`
- **任务**:
  - **优化**: 维持 `Per-Batch` 的梯度计算 (`total_loss.backward()`)、`SMK` 过滤和权重更新 (`optimizer.step()`)。
  - **策展**: 维持 `Per-Subject` 的策展逻辑，基于 `per_subject_loss` 和其他主体级指标进行决策，并将 `(q_subject, batch_min_k_indices, priority)` 存入经验缓存区。

---

## 3. 最终架构流程图 (V3 - 动态高斯版)

```mermaid
graph TD
    subgraph "动态高斯路由方案"
        style Final fill:#f2e6ff,stroke:#4c0099,stroke-width:2px

        Start[Batch In] --> x_token["Token-Level x"];
        x_token -- "mean over tokens" --> q_subject["Subject-Level q"];
        
        subgraph "路由路径 (Token/Subject)"
            q_subject --> Retrieve["检索 Subject 上下文 History"];
            
            x_token --> BuildSequence["构建 [History, x_token] 序列"];
            Retrieve --> BuildSequence;
            
            BuildSequence --> Gating[GatingTransformer];
            Gating --> DynamicParams["动态生成 [μ, log_σ]"];
            
            DynamicParams --> DynamicGaussians["创建动态高斯分布"];
            x_token --> Match["匹配 x_token"];
            DynamicGaussians --> Match;
            
            Match --> LogProbs["计算对数概率 (路由分数)"];
            LogProbs --> ExpertForward["专家前向传播"];
        end

        subgraph "优化与策展路径 (Batch/Subject)"
             ExpertForward --> PerSubjectLoss["聚合为 Per-Subject Loss"];
             PerSubjectLoss --> TotalLoss["求和为 Total Loss"];
             TotalLoss --> Backward["Batch Backward()"];
             Backward --> SMK["Batch SMK Filter"];
             SMK --> OptimizerStep["Optimizer.step()"];

             PerSubjectLoss --> Curation["Per-Subject 策展"];
             SMK --> Curation;
             q_subject --> Curation;
             Curation --> Buffer["更新Buffer"];
        end

        OptimizerStep --> End[Batch Out];
    end
