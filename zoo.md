# Model Zoo and Experiments

Our test suite is now built around a lightweight Vision Transformer architecture to facilitate rapid experimentation with cognitive learning principles.

The goal is to observe the performance of different learning strategies under resource constraints, thereby more clearly demonstrating the advantages of mechanisms like PILR-S (Predictive Integrity Learning Rate Scheduler).

### MNIST Spaced Rehearsal Experiments

We also conducted spaced rehearsal experiments on the MNIST and FashionMNIST datasets to further explore continuous learning capabilities.

| **8x2 Full (FashionMNIST -> MNIST)** | **8x2 Pre-trained + 8x2 PILR-S Spaced Rehearsal (FashionMNIST -> MNIST)** | **8x2 PILR-S Full (FashionMNIST -> MNIST) (1.2σ)** |
| :--:| :--:| :--:|
| ~0.26M | ~0.26M | ~0.26M |
| <img src="output/ViT/img/tiny-gbp/20250627-tiny-moe-mnist-mnist-rehearsal.png" style="max-width:200px;"> | <img src="output/ViT/img/tiny-gbp/20250627-tiny-gbp-mnist-mnist-rehearsal.png" style="max-width:200px;"> | <img src="output/ViT/img/tiny-gbp/20250627-tiny-gbp-2-mnist-mnist-rehearsal.png" style="max-width:200px;"> |

### Marathon Rehearsal Experiments v1

A more demanding experimental setup involving cyclical rehearsal across four datasets: CIFAR-10, MNIST, FashionMNIST, and SVHN, in a `5 * (5+1+1+1)` epoch schedule. This tests the model's ability to handle catastrophic forgetting and adapt to different data distributions over a prolonged training period. Unfortunately, the results were not prominent due to improper schedule settings.

| **16x4 MoE Marathon** | **16x4 PILR-S MoE Marathon** | **16x4 PISA MoE Marathon** |
| :--:| :--:| :--:|
| ~0.46M | ~0.46M | ~0.46M |
| <img src="output/ViT/img/marathon-v1/20250628T053559_large-moe-mnist-marathon-rehearsal-Metrics.png" style="max-width:200px;"> | <img src="output/ViT/img/marathon-v1/20250628T044505_large-pilr-mnist-marathon-rehearsal-Metrics.png" style="max-width:200px;"> | <img src="output/ViT/img/marathon-v1/20250628T070228-marathon_v1-large_pisa_mnist-Metrics.png" style="max-width:200px;"> |

### Marathon Rehearsal Experiments v2

A more challenging setup with a `5 * (4+2+3+5)` schedule, increasing the training focus on more complex datasets (CIFAR-10, SVHN).

| **16x4 MoE** | **16x4 PILR-S MoE** | **16x4 PISA MoE** | **16x4 PISA-2 MoE** |
| :--:| :--:| :--:| :--:|
| ~0.46M | ~0.46M | ~0.46M | ~0.46M |
| <img src="output/ViT/marathon-v2/img/20250628T075240-marathon_v2-large_moe_mnist-Metrics.png" style="max-width:200px;"> | <img src="output/ViT/marathon-v2/img/20250628T095638-marathon_v2-large_pisa_mnist-Metrics.png" style="max-width:200px;"> | <img src="output/ViT/marathon-v2/img/20250628T090017-marathon_v2-large_pisa_2_mnist-Metrics.png" style="max-width:200px;"> | *TBD* |

### Marathon Rehearsal Experiments v3

A marathon rehearsal experiment without SVHN, following a `4 * (5+2+3)` epoch schedule.

| **16x4 MoE ** | **16x4 PISA MoE** |
| :--:| :--:|
| ~0.46M | ~0.46M |
| <img src="output/ViT/marathon-v3/img/20250628T105444-marathon_v3-large_moe_mnist-Metrics.png" style="max-width:200px;"> | <img src="output/ViT/marathon-v3/img/20250628T112443-marathon_v3-large_pisa_mnist-Metrics.png" style="max-width:200px;"> |
