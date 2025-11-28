# 📅 Stage 2: Transformer 与 NLP/具身实战 (21天计划)

**核心战略**：

1.  **祛魅**：Transformer 不是黑魔法，它只是一堆矩阵乘法。
2.  **维度优先**：时刻关注张量维度变化 `[Batch, Seq_Len, Dim]`，这是理解算法和 Debug 的唯一真理。
3.  **具身迁移**：学习 NLP 的同时，我会告诉你这个组件在 Robot 身上变成了什么（例如：Token 变成了 机器人关节指令 或 图像Patch）。

-----

## Week 5: 深入 Transformer 心脏 (理论 + PyTorch 原生实现)

**目标**：不依赖任何库，仅用 `torch.nn` 实现 Attention，能白板画图。

  * **Day 1: 输入的奥秘 - Embedding 与 位置编码 (Positional Encoding)**

      * **理论**：为什么 Transformer 需要位置编码？(因为 Self-Attention 是并行的，没有时序信息)。
      * **代码**：实现正弦/余弦位置编码公式。
      * **具身视角**：在机器人中，位置编码可能代表时间步（t=1, t=2），也可能代表空间位置（图像的 Patch 坐标）。
      * **面试题**：为什么要用正弦函数？为什么不用可学习的 Embedding？

  * **Day 2: 灵魂核心 - Scaled Dot-Product Attention (点积注意力)**

      * **理论**：$Q$ (Query), $K$ (Key), $V$ (Value) 的物理含义。
      * **公式**：$$\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$
      * **代码**：**这是本周最重要的一天。** 手写 `def scaled_dot_product_attention(...)`。
      * **维度流**：`[B, L, D] @ [B, D, L] -> [B, L, L]` (Attention Map)。

  * **Day 3: 多头机制 - Multi-Head Attention**

      * **理论**：为什么要“多头”？（类比 CNN 的多通道，关注不同的特征子空间）。
      * **代码**：实现 `class MultiHeadAttention(nn.Module)`。重点在于如何把 Tensor 切分再拼接。
      * **面试题**：多头之后计算量增加了吗？（标准答案：没有，因为每个头的维度降了）。

  * **Day 4: 组装 - Encoder, Decoder 与 Mask 机制**

      * **理论**：Encoder (双向可见) vs Decoder (只能看过去)。什么是 `Padding Mask` 和 `Look-ahead Mask`？
      * **代码**：实现 `Add & Norm` (残差连接 + LayerNorm)。
      * **面试题**：为什么 Transformer 用 LayerNorm 而不用 BatchNorm？（NLP 数据长度不一，BN 效果差）。

  * **Day 5: 完整前向传播 (Forward Pass) 跑通**

      * **实战**：把 Day 1-4 的组件拼起来，输入一个随机 Tensor `[32, 10, 512]`，确保输出维度不变（或符合预期）。

  * **Day 6: [支线任务] Vision Transformer (ViT) —— 具身智能的桥梁**

      * **概念**：如何把一张图片变成 NLP 的句子？
      * **核心**：Patch Embedding。把 $224 \times 224$ 的图切成 $16 \times 16$ 的小方块，每个方块就是一个 "Word"。
      * **价值**：这是你作为具身机器人负责人的核心竞争力。理解了 ViT，你就理解了为什么机器人现在能“看懂”世界。

  * **Day 7: 总结与白板模拟**

      * **任务**：在一张白纸上，默写出 Transformer 整体架构图，并口述数据流转过程。

-----

## Week 6: 拥抱 Hugging Face (工具链与生态)

**目标**：从“造轮子”转为“用轮子”。熟练使用工业界标准库 `transformers`。

  * **Day 8-9: Tokenizer (分词器) 详解**

      * **概念**：WordPiece, BPE 算法。`[CLS]`, `[SEP]` 标记的作用。
      * **实战**：使用 `BertTokenizer` 处理中文句子，查看输出的 `input_ids` 和 `attention_mask`。

  * **Day 10-11: Model 与 Pre-trained Weights**

      * **实战**：加载 `bert-base-chinese`。
      * **任务**：输入一句话，提取其 Embedding 向量。计算两个句子的余弦相似度（语义相似度）。
      * **具身视角**：在机器人任务中，这相当于提取“指令特征”，用于指导动作生成。

  * **Day 12-14: Datasets 库与数据预处理**

      * **实战**：加载一个公开的情感分析数据集（如 `ChnSentiCorp`）。
      * **任务**：编写 `collate_fn`，将长短不一的句子 Padding 到相同长度，制作为 `DataLoader`。

-----

## Week 7: [简历项目] BERT 微调与模型落地

**目标**：产出简历上的第一个 NLP 项目 —— **"基于 BERT 的垂直领域文本分类/意图识别"**。

  * **Day 15-16: 搭建微调 (Fine-tuning) 框架**

      * **架构**：BERT Encoder + `nn.Linear` (分类头)。
      * **代码**：复用 Stage 1 的训练循环，只需要把 Model 换成 BERT。
      * **面试点**：什么是 "Freeze" (冻结参数)？什么是全量微调？

  * **Day 17-18: 训练与调优**

      * **任务**：在 GPU 上训练模型。观察 Loss 下降曲线。
      * **调参**：调整 Learning Rate (通常 Transformer 需要很小的 LR，如 2e-5)。

  * **Day 19: 推理 (Inference) 与 部署**

      * **任务**：编写 `predict(text)` 函数，输入自然语言，输出类别。
      * **交互**：用 `input()` 做一个简单的命令行 Demo。

  * **Day 20: 具身智能场景模拟**

      * **思考**：假设这是一个“机器人指令分类器”。
      * **Case**：输入 "把桌上的红色杯子递给我" -\> 分类为 "Pick\_and\_Place"。
      * **输出**：将这个简单的分类器包装成你项目中的一个“意图识别模块”。

  * **Day 21: 复盘与模拟面试**

      * **任务**：整理项目代码到 GitHub（或本地归档）。
      * **模拟**：回答关于 BERT 结构、Attention 机制、微调策略的面试题。