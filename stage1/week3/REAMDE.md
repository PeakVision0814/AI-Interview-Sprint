### 🚀 Week 3 (Day 15 - Day 21) 详细学习计划：MNIST 图像分类项目

本周的核心是将 W2 的“八股文”知识，应用到一个完整的端到端 (End-to-End) 项目中。我们将引入新的模型 (CNN)，并补全评估 (Evaluation) 环节。

#### W3D1 (Day 15): CNN 理论与模型构建

* **今日目标**：理解为什么 CNN 适合图像，并使用 PyTorch 定义你的第一个 CNN 模型。
* **学习任务**：
    1.  **理论补课 (面试导向)**：
        * **为什么是 CNN？** 学习 CNN 的两大核心特性：**局部感受野 (Local Receptive Fields)** 和 **参数共享 (Parameter Sharing)**。
        * **面试问题**：“为什么全连接网络 (MLP) 不适合处理高分辨率图像？” (提示：参数量爆炸)
    2.  **PyTorch 新模块**：
        * `nn.Conv2d`：学习它的核心参数 (in_channels, out_channels, kernel_size, stride, padding)。
        * `nn.MaxPool2d`：学习它的作用（下采样）和参数 (kernel_size)。
        * `nn.Flatten` 或 `view()`：学习如何在卷积层和全连接层之间“压平”数据。
* **今日产出**：
    * 编写一个 `model.py` 文件。
    * 在其中定义一个 `SimpleCNN` 类 (继承 `nn.Module`)，它至少包含： `Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool -> Flatten -> Linear -> Linear`。
    * **单元测试**：实例化你的模型，创建一个 `torch.randn(1, 1, 28, 28)` 的假数据，并将其喂给模型，确保 `forward` 函数能跑通并返回正确的形状 (e.g., `[1, 10]`)。

#### W3D2 (Day 16): 数据集准备 (Data Pipeline)

* **今日目标**：使用 `torchvision` 加载标准数据集，并构建训练和测试的 `DataLoader`。
* **学习任务**：
    1.  **Torchvision**：学习如何使用 `torchvision.datasets.MNIST` (或 `FashionMNIST`)。
    2.  **Transforms**：学习使用 `torchvision.transforms`。
        * `transforms.ToTensor()`：理解它如何将 PIL 图像 (HWC, 0-255) 转换为 PyTorch 张量 (CHW, 0.0-1.0)。
        * `transforms.Normalize()`：(可选，但推荐) 了解为什么以及如何进行归一化。
        * `transforms.Compose()`：学习如何将多个变换组合起来。
* **今日产出**：
    * 编写一个 `dataset.py` 文件 (或在主脚本中完成)。
    * 加载 MNIST 训练集和测试集。
    * 创建 `train_loader` 和 `test_loader` (设置合理的 `batch_size`, 并确保 `train_loader` 开启 `shuffle=True`)。
    * **单元测试**：从 `train_loader` 中取一个 `batch` 的数据，并打印其 `images.shape` 和 `labels.shape`，确保它们符合预期。

#### W3D3 (Day 17): 编写训练循环 (Training Loop)

* **今日目标**：将 W3D1 (模型) 和 W3D2 (数据) 组装起来，默写 W2 的“训练八股文”。
* **学习任务**：
    1.  **组装**：实例化你的 `SimpleCNN` 模型、损失函数 (`nn.CrossEntropyLoss`) 和优化器 (`torch.optim.Adam`)。
    2.  **编写循环**：
        * `for epoch in epochs:`
        * `for images, labels in train_loader:`
        * **复习“5步法”**：
            1.  前向传播 (`outputs = model(images)`)
            2.  计算损失 (`loss = criterion(outputs, labels)`)
            3.  **梯度清零** (`optimizer.zero_grad()`)
            4.  **反向传播** (`loss.backward()`)
            5.  **参数更新** (`optimizer.step()`)
* **今日产出**：
    * 编写 `train.py` 脚本的主体。
    * 让模型在训练集上跑起来 (例如，先跑 1-2 个 epoch)。
    * **关键指标**：在训练循环中打印 `loss`，确保它在**显著下降**。

#### W3D4 (Day 18): 编写评估循环 (Evaluation Loop)

* **今日目标**：编写一个**独立**的评估函数，用于在验证集/测试集上计算模型的**准确率 (Accuracy)**。
* **学习任务**：
    1.  **评估模式**：
        * `model.eval()`：理解它为什么是必须的 (关闭 Dropout 和 BatchNorm)。
        * `with torch.no_grad():`：理解它为什么是必须的 (关闭梯度计算，节省显存和算力)。
    2.  **计算准确率**：
        * 从模型的 `outputs` (logits) 中获取预测结果：`preds = torch.argmax(outputs, dim=1)`。
        * 比较预测和真实标签：`correct += (preds == labels).sum().item()`。
        * 计算总准确率：`total_accuracy = correct / len(test_dataset)`。
* **今日产出**：
    * 在 `train.py` 中定义一个 `evaluate(model, data_loader, criterion)` 函数。
    * 在每个 epoch 训练结束后，调用此函数，打印出当前 epoch 在**测试集**上的 `test_loss` 和 `test_accuracy`。

#### W3D5 (Day 19): 整合、调优与理论

* **今日目标**：整合所有代码，进行一次完整的训练，并探讨 `batch_size` 的影响。
* **学习任务**：
    1.  **模型保存**：在评估循环中，加入 W2 学过的 `torch.save`。在每个 epoch 结束时，如果 `test_accuracy` 创下新高，则保存当前模型 (`.pth` 文件)。
    2.  **完整训练**：运行一个完整的训练 (例如 10-20 个 epochs)，观察 `train_loss` 和 `test_accuracy` 的变化曲线。
    3.  **理论补课 (面试导向)**：
        * **`batch_size` 的影响**：
            * **大 Batch Size** (如 1024)：优点 (训练快，梯度稳定)；缺点 (可能陷入“尖锐”的最小值，泛化能力下降)。
            * **小 Batch Size** (如 32)：优点 (自带噪声，泛化能力强)；缺点 (训练慢，梯度震荡)。
* **今日产出**：
    * 一个干净的、可运行的 `train.py`。
    * 一个训练好的模型文件，例如 `best_model.pth`。
    * 能口头回答 `batch_size` 对训练速度和模型泛化性的影响。

#### W3D6 (Day 20): 推理 (Inference)

* **今日目标**：学习如何使用已保存的模型进行预测。
* **学习任务**：
    1.  **加载模型**：复习 `model.load_state_dict()`。
    2.  **准备单张图片**：从测试集中取 *一张* 图片，并确保对其进行了与训练时**完全相同**的 `transform` (特别是 `ToTensor` 和 `Normalize`)。
    3.  **处理 Batch 维度**：模型期望的输入是 `(B, C, H, W)`，但你只有 `(C, H, W)`。学习使用 `image.unsqueeze(0)` 来增加一个 `batch` 维度。
    4.  **预测**：将处理后的图片喂给模型，并得到预测结果。
* **今日产出**：
    * 编写一个**新的** `predict.py` 脚本。
    * 该脚本加载 `best_model.pth`，加载一张测试图片，并打印出“预测标签”和“真实标签”。

#### W3D7 (Day 21): 项目复盘与准备 P2

* **今日目标**：复盘 W3 项目，查漏补缺，为 P2 (Transformer) 做准备。
* **学习任务**：
    1.  **代码重构**：将你的代码整理成更清晰的结构 (例如，`model.py`, `train.py`, `predict.py`)。
    2.  **面试复盘**：
        * “请你描述一下你这个 MNIST 项目的完整流程。” (考察端到端能力)
        * “`model.train()` 和 `model.eval()` 有什么区别？” (W2 知识)
        * “`loss.backward()` 和 `optimizer.step()` 分别做了什么？” (W2 知识)
        * “为什么在评估时要使用 `with torch.no_grad()`？” (W3D4 知识)
    3.  **预习 P2**：(如果时间允许) 开始阅读 "The Illustrated Transformer"，对 "Attention 机制"有一个初步印象。

---

我们今天将从 **W3D1 (Day 15)** 开始。

请问你准备好开始学习 CNN 的核心理论 (`nn.Conv2d`) 并构建你的第一个 CNN 模型了吗？