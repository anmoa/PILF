# 模型动物园与实验

我们的测试套件现在围绕一个轻量级（约 1M 参数）的 Vision Transformer 架构构建，以便于快速进行认知学习原理的实验。我们在 CIFAR-10 数据集上比较了三种主要变体，并使用 SVHN 作为分布外（OOD）验证集。

目标是观察不同学习策略在资源受限下的表现，从而更清晰地展示 PILR-S（Predictive Integrity Learning Rate Scheduler）等机制的优势。

|   **基线 ViT**    |   **4x1 MoE-ViT**    |   **16x4 MoE-ViT**    | **带有 3σ 学习的 16x4 PILR-S-MoE-ViT**       |
| :-------: | :----------: | :: | - |
|      ~0.81M       |~1.21M|~1.23M | ~1.23M       |
| <img src="output/ViT/img/legacy_img/20250626-BASE_ViT-Params_0.81M.png" style="max-width:200px;"> | <img src="output/ViT/img/legacy_img/20250626-MOE_4x1_ViT-Params_1.21M.png" style="max-width:200px;"> | <img src="output/ViT/img/legacy_img/20250626-MOE_16x4_ViT-Params_1.23M.png" style="max-width:200px;"> | <img src="output/ViT/img/legacy_img/20250626-GBP_MOE_ViT-Params_1.23M.png" style="max-width:200px;"> |

### MNIST 间隔复习实验

我们还在 MNIST 和 FashionMNIST 数据集上进行了间隔复习实验，以进一步探索持续学习的能力。

|      **8x2 全程 (FashionMNIST -> MNIST)**       |  **8x2 预训练 + 8x2 PILR-S 间隔复习 (FashionMNIST -> MNIST)**   |**8x2 PILR-S 全程 (FashionMNIST -> MNIST) (1.2σ)** |
| :-----: | :-----: | :-------: |
|     ~0.26M      |     ~0.26M      |      ~0.26M       |
| <img src="output/ViT/img/20250627-tiny-moe-mnist-mnist-rehearsal.png" style="max-width:200px;"> | <img src="output/ViT/img/20250627-tiny-gbp-mnist-mnist-rehearsal.png" style="max-width:200px;"> | <img src="output/ViT/img/20250627-tiny-gbp-2-mnist-mnist-rehearsal.png" style="max-width:200px;"> |

### 马拉松复习实验

一个更严苛的实验设置，涉及在四个数据集（CIFAR-10、MNIST、FashionMNIST 和 SVHN）之间进行循环复习，遵循 `5 * (5+1+1+1)` 的周期计划。该实验旨在测试模型在长时间训练中处理灾难性遗忘和适应不同数据分布的能力。

| **16x4 MoE 马拉松** | **16x4 PILR-S MoE 马拉松** |
| :--:| :--:|
| ~0.46M | ~0.46M |
| *待定* | *待定* |
