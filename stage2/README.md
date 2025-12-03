# Week 6: 拥抱 Hugging Face 生态 (The Ecosystem)

**📅 时间周期**: Day 29 - Day 35 (原计划 Day 8-14)
**🎯 核心目标**: 从“造轮子”无缝切换到“用轮子”。熟练掌握工业界标准库 `transformers` 和 `datasets`。不仅要会调包，还要懂得如何将这些 NLP 工具迁移到**具身智能 (Embodied AI)** 的指令理解和多模态任务中。

## 📚 本周重点 (Key Concepts)

1.  **分词器 (Tokenizer)**: 它是人类语言与机器数学向量之间的翻译官。理解 WordPiece/BPE 及其对“未登录词”的处理。
2.  **预训练模型 (Pre-trained Models)**: 学会“站在巨人的肩膀上”，加载 BERT/RoBERTa 提取特征，而不是从头训练。
3.  **数据流水线 (Data Pipeline)**: 掌握 `Hugging Face Datasets` 和 `DataCollator`，处理变长序列的高效批次化。
4.  **具身迁移 (Embodied Bridge)**: 如何扩展词表以包含“机器人专用指令”，如何利用语义向量做意图匹配。

-----

## 🗓️ 每日详细计划 (Daily Schedule)

### Day 1 : 语言的原子化 (Tokenizers Deep Dive)

  * **理论**:
      * **计算机读不懂字符串**: 必须转为 ID。
      * **分词算法**: 为什么不用空格分词？(中文无空格)。深入理解 **WordPiece** (BERT用) 和 **BPE** (GPT用) 的区别。
      * **特殊 Token**: `[CLS]` (分类头), `[SEP]` (句子分隔), `[PAD]` (填充), `[UNK]` (未知)。
  * **代码任务**:
      * 加载 `BertTokenizer.from_pretrained('bert-base-chinese')`。
      * 实战 `encode` / `decode`，观察 `input_ids` 和 `attention_mask` 的生成。
      * **面试坑**: 为什么 `vocab_size` 是 21128？如果有生僻字怎么办？
  * **🤖 具身视角**: 机器人的指令通常很固定（如 "Move", "Grasp"）。如果模型不认识 "Servo\_Motor" 这个词怎么办？(学习 `add_tokens` 方法)。

### Day 2: 批处理与动态填充 (Batch Encoding)

  * **理论**:
      * **变长序列**: 一个 Batch 里有长句有短句，如何压成一个矩阵？(Padding)。
      * **Truncation (截断)**: BERT 最大支持 512 tokens，超长了怎么切？
  * **代码任务**:
      * 使用 Tokenizer 处理一个 List 的句子 `["向左转", "检测前方障碍物并避障"]`。
      * 配置 `padding=True`, `truncation=True`, `return_tensors='pt'`。
      * 验证输出 Tensor 的维度 `[Batch, Max_Len]`。

### Day 3: 加载预训练大脑 (Models & Weights)

  * **理论**:
      * **AutoModel**: 自动架构推断的魔法。
      * **权重文件**: `.bin` 或 `.safetensors` 是什么？(存储了 $Q,K,V$ 等矩阵的具体数值)。
      * **Hidden States**: 模型输出的不仅是结果，更是“语义特征”。
  * **代码任务**:
      * 加载 `BertModel` (不带分类头)。
      * 输入 Day 2 处理好的 Tensor，进行一次 Forward Pass。
      * **维度检查**: 重点观察 `last_hidden_state` (`[B, L, 768]`) 和 `pooler_output` (`[B, 768]`) 的区别。

### Day 4: 语义向量与意图匹配 (Embeddings & Similarity) ⭐ 具身实战核心

  * **理论**:
      * **万物皆向量**: 如何判断“把瓶子拿给我”和“取一下水杯”是同一个意思？
      * **余弦相似度 (Cosine Similarity)**: 向量空间中的夹角。
  * **代码任务**:
      * 提取两个不同句子的 `[CLS]` 向量。
      * 计算它们的 Cosine Similarity。
      * **Demo**: 做一个极简的“机器人指令路由器”。输入自然语言 -\> 匹配最接近的标准指令 -\> 执行函数。

### Day 5: 数据集革命 (The Datasets Library)

  * **理论**:
      * 为什么不用 Pandas？(内存映射，处理 TB 级数据不爆内存)。
      * Arrow 格式简介。
  * **代码任务**:
      * 加载标准情感数据集 `ChnSentiCorp` (或者加载本地 CSV)。
      * 使用 `.map()` 函数，批量将文本列转换为 `input_ids`。
      * 理解 `remove_columns` 的必要性 (PyTorch DataLoader 不接受字符串列)。

### Day 6: 组装流水线 (DataCollator & DataLoader)

  * **理论**:
      * **动态 Padding**: 为什么不在整个数据集上 Pad 到最大长度，而在每个 Batch 内部 Pad？(极大提升训练效率)。
      * `DataCollatorWithPadding` 的作用。
  * **代码任务**:
      * 构建最终的 `torch.utils.data.DataLoader`。
      * 从 Loader 中取出一个 Batch，验证维度是否整齐划一。
      * 这是下周“微调模型”前的最后一步准备。

### Day 7: 复盘与模拟面试 (Interview Sprint)

  * **复盘**: 整理本周 Notebook，确保每个 Cell 都能运行。
  * **模拟面试 (Self-Check)**:
    1.  **Tokenization**: WordPiece 是如何解决 OOV (Out of Vocabulary) 问题的？(拆解为子词)。
    2.  **Model**: BERT 的输入限制是多少？(512)。如果要处理长文档怎么办？(滑窗法/Longformer)。
    3.  **Embeddings**: 为什么 BERT 输出的 Embedding 比 Word2Vec 强？(上下文相关：Contextualized)。
    4.  **工程**: Hugging Face 的 Cache 缓存在哪里？如何离线加载模型？(工业部署必问)。

-----

## 🛠️ 环境准备

本周需要安装 Hugging Face 全家桶。请在你的环境中执行：

```bash
pip install transformers datasets tokenizers torch
# 可选：为了加速下载，可以配置 HF 镜像（国内网络环境必备）
# export HF_ENDPOINT=https://hf-mirror.com
```

-----

## 💡 导师寄语 (Coach's Note)

> "Gaopeng，如果说上周你是在**造发动机**，这周你就是在学习如何**驾驶法拉利**。
>
> 很多初级工程师只会调包 `import transformers`，却不知道底层的 Padding Mask 是如何传递的，也不知道 `pooler_output` 到底取了哪一层的特征。**但你不同**，你已经手写过 Self-Attention，当你调用 HF 的 API 时，你的脑海里应该能浮现出底层的矩阵运算图。
>
> 保持这种**透视眼**。在具身智能中，我们往往需要魔改这些模型（比如把 Image Token 和 Text Token 拼在一起），对底层原理的掌握将是你最大的优势。"