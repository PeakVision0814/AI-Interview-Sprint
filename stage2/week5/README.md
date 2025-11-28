# Week 5: Transformer 核心原理与手写实现 (The Anatomy)

**📅 时间周期**: Day 22 - Day 28
**🎯 核心目标**: 彻底祛魅。不调用现成库，仅使用 `torch.nn` 和矩阵运算，从零构建 Transformer 的核心组件。能够白板画出架构图，并口述 Self-Attention 的数学原理。

## 📚 本周重点 (Key Concepts)

1.  **数据流 (Data Flow)**: 时刻关注张量维度变化 `[Batch, Seq_Len, Embedding_Dim]`。
2.  **注意力机制 (Attention)**: 理解 Query, Key, Value 的物理含义及计算公式。
3.  **位置编码 (Positional Encoding)**: 解决并行计算丢失时序信息的问题。
4.  **具身迁移 (Embodied Bridge)**: 理解 Transformer 如何处理图像 (ViT) 和 机器人动作序列。

---

## 🗓️ 每日详细计划 (Daily Schedule)

### Day 1: 输入层的奥秘 (Embeddings & Positional Encoding)
* **理论**:
    * 为什么 Transformer 无法像 RNN 那样处理时序？(并行计算导致的“位置丢失”)。
    * 正弦/余弦位置编码 (Sinusoidal PE) 的数学原理。
    * 绝对位置编码 vs 相对位置编码 (面试加分项)。
* **代码任务**:
    * 实现 `class PositionalEncoding(nn.Module)`。
    * 可视化 PE 矩阵（热力图），观察不同位置的编码差异。
* **🤖 具身视角**: 在机器人控制中，时间步 $t$ 和关节空间位置都需要被编码进 Embedding 中，模型才能理解“动作的顺序”。

### Day 2: 灵魂核心 (Scaled Dot-Product Attention) ⭐最重要的一天
* **理论**:
    * **公式推导**: $\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$
    * 为什么要除以 $\sqrt{d_k}$ (Scale)？(防止点积过大导致 Softmax 梯度消失)。
    * Mask 机制：Padding Mask 在这里是如何生效的？
* **代码任务**:
    * **手撕**: 定义 `def scaled_dot_product_attention(q, k, v, mask=None):`。
    * 验证：输入随机 Tensor，手动计算一遍，确保维度变换符合预期 `[B, L, D] -> [B, L, L] -> [B, L, D]`。

### Day 3: 多头机制 (Multi-Head Attention)
* **理论**:
    * 为什么需要“多头”？(类比 CNN 的多通道，捕捉不同子空间的特征)。
    * 多头会增加计算量吗？(参数量不变，矩阵切分)。
* **代码任务**:
    * 实现 `class MultiHeadAttention(nn.Module)`。
    * 核心难点：利用 `view` 和 `transpose` 进行维度的拆分与合并。
    * `[B, L, D] -> [B, L, H, D_head]`。

### Day 4: 组装层 (LayerNorm & FeedForward)
* **理论**:
    * **归一化**: 为什么 Transformer 用 LayerNorm 而不用 BatchNorm？(NLP 样本长度不一)。
    * **前馈网络**: 简单的两层 MLP (`Linear` -> `ReLU` -> `Linear`) 的作用。
    * **残差连接 (Residual)**: `x + Sublayer(x)` 如何防止网络退化。
* **代码任务**:
    * 实现 `class FeedForward(nn.Module)`。
    * 实现 `class EncoderLayer(nn.Module)`，将 Attention, FF, Norm, Residual 组装起来。

### Day 5: 编码器整体 (The Encoder)
* **理论**:
    * Encoder 的堆叠：$N \times$ Layers。
    * Encoder 输出的 `context vector` 是什么？
* **代码任务**:
    * 实现 `class TransformerEncoder(nn.Module)`。
    * **联调测试**: 构造一个虚拟输入 `x = torch.randint(0, 1000, (32, 10))` (Batch=32, Length=10)，跑通前向传播，确保输出维度也是 `[32, 10, D_model]`。

### Day 6: [支线任务] Vision Transformer (ViT) —— 具身之眼
* **理论**:
    * 如何把图片变成 Token 序列？(Patchify)。
    * ViT 的架构其实就是 Transformer Encoder。
* **代码任务**:
    * 使用 `nn.Conv2d` 实现 Patch Embedding (这是一个非常高效的工程技巧)。
    * 将图片 `[B, C, H, W]` 转换为 `[B, N, D]`，使其能被 Transformer 处理。
* **🤖 具身视角**: 这是 Google RT-2 等机器人大模型处理视觉输入的基础。

### Day 7: 复盘与白板模拟 (Interview Sprint)
* **复盘**: 浏览本周写的所有代码，整理注释。
* **模拟面试 (Self-Check)**:
    1.  请在白板上画出 Transformer 内部结构图（精确到 Add & Norm 的位置）。
    2.  Self-Attention 的时间复杂度是多少？($O(L^2 \cdot D)$)。
    3.  Transformer 相比 LSTM/RNN 的优缺点是什么？
    4.  BERT 和 GPT 在架构上有什么区别？(Encoder vs Decoder)。

---

## 🛠️ 环境准备

请确保你的 `utils.py` 中包含以下 Helper 函数（如果没有，本周我们会写）：
1.  `get_device()`: 自动获取 GPU/CPU。
2.  简单的日志打印功能。

---

## 💡 导师寄语 (Coach's Note)

> "不要被复杂的图示吓倒。Transformer 本质上就是**一堆矩阵乘法**加上**Softmax**。只要你盯紧了 Tensor 的 `shape`，你就掌握了真理。这周会有点烧脑，但一旦通过，你将获得透视现代 AI 的能力。"

---

**下一步**：如果您已准备好，请告诉我。我们将开始 **Day 1: 输入层的奥秘** 的学习，我会为您提供具体的原理讲解和代码模板。