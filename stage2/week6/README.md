# 📅 Stage 2 Week 6: 拥抱 Hugging Face (工程实战篇)

**📅 时间周期**: Day 29 - Day 35 (本周 7 天)
**🎯 核心目标**: 从“造轮子”切换到“用轮子”。熟练掌握工业界标准库 `transformers` 和 `datasets`。不仅要会调用 API，还要深挖其背后的 `Tokenizer` 逻辑和 `Batch` 处理机制，为下周的 BERT 微调项目打下坚实的工程基础。

## 📂 本周目录结构 (Directory Structure)

请在 `stage2/week6/` 下创建以下 Notebooks：

```text
.
├── stage2/
│   ├── week6/
│   │   ├── README.md                          <-- (本周学习笔记汇总)
│   │   ├── S2W6D1_Tokenizer_Basics.ipynb      <-- [Day 1] 分词器原理与实战
│   │   ├── S2W6D2_Advanced_Batch_Input.ipynb  <-- [Day 2] 批量处理与 Mask 详解
│   │   ├── S2W6D3_Model_Architecture.ipynb    <-- [Day 3] 加载预训练模型与权重
│   │   ├── S2W6D4_Semantic_Embedding.ipynb    <-- [Day 4] 语义向量与相似度 (具身核心)
│   │   ├── S2W6D5_HuggingFace_Datasets.ipynb  <-- [Day 5] 高效数据加载与预处理
│   │   ├── S2W6D6_Dynamic_Padding.ipynb       <-- [Day 6] 手写 collate_fn (工程核心)
│   │   └── S2W6D7_Pipeline_and_Review.ipynb   <-- [Day 7] 综合流水线与复盘
```

*(注：本周涉及的数据集如果是临时下载的，Hugging Face 默认会缓存在 `~/.cache/huggingface`，或者我们可以指定下载到根目录的 `data/` 文件夹中)*

-----

## 🗓️ 每日详细计划 (Daily Schedule)

### Day 1: 分词器的解剖 (Tokenizer Basics)

  * **文件名**: `S2W6D1_Tokenizer_Basics.ipynb`
  * **理论**:
      * **从文本到数字**: 计算机只认识 ID。理解 WordPiece (BERT) 和 BPE (GPT) 的区别。
      * **特殊 Token**: `[CLS]` (分类标记), `[SEP]` (句子分隔), `[UNK]` (未知词), `[PAD]` (填充) 的物理含义。
  * **代码任务**:
      * 加载 `BertTokenizer.from_pretrained('bert-base-chinese')`。
      * 探究 `vocab.txt`: 查看 ID `101` 对应什么？你的名字“黄”字对应什么 ID？
      * 手动执行 `tokenize` -\> `convert_tokens_to_ids` -\> `decode` 的完整闭环。
  * **🤖 具身视角**: 机器人指令中，动词（如 "Pick"）和名词（"Cup"）的 Token ID 是后续模型理解意图的原子单位。

### Day 2: 输入管道进阶 (Advanced Batch Input)

  * **文件名**: `S2W6D2_Advanced_Batch_Input.ipynb`
  * **理论**:
      * **Attention Mask**: 为什么 Transformer 需要 Mask？(告诉 Self-Attention 矩阵忽略 Padding 的部分，防止将 0 算入 Softmax)。
      * **Token Type IDs**: 在处理句子对（Sentence Pair）任务时，如何区分前半句和后半句？
  * **代码任务**:
      * 使用 `tokenizer.encode_plus()` 或 `tokenizer(text, padding=True, truncation=True)` 一键生成所有张量。
      * **对比实验**: 对同一个 Batch，开启 vs 关闭 `attention_mask`，观察 BERT 输出的差异。
  * **💥 面试坑点**: "Tokenizer 输出的 `offset_mapping` 是做什么的？" (在做命名实体识别 NER 时，用于将 Token 映射回原始字符串的字符位置)。

### Day 3: 加载预训练模型 (Model Loading & Architecture)

  * **文件名**: `S2W6D3_Model_Architecture.ipynb`
  * **理论**:
      * **Hugging Face 三剑客**: `Config` (配置), `Model` (架构), `PreTrainedModel` (权重)。
      * `AutoModel` 的魔法：它是如何根据字符串自动推断模型结构的？
  * **代码任务**:
      * 使用 `BertModel.from_pretrained('bert-base-chinese')`。
      * **打印模型结构**: 对比 `print(model)` 的输出和你上周手写的 `TransformerEncoder`，找到对应的层 (Embeddings, Encoder, Pooler)。
      * **前向传播**: 传入 Day 2 生成的 Input IDs，查看输出张量的 Shape `[Batch, Seq_Len, Hidden_Dim]`。

### Day 4: 语义向量与具身指令 (Semantic Embedding) ⭐ 重点

  * **文件名**: `S2W6D4_Semantic_Embedding.ipynb`
  * **理论**:
      * **Sentence Embedding**: 如何用一个向量代表整句话？(常用 `[CLS]` 位置的输出，或 Mean Pooling)。
      * **余弦相似度**: 衡量两个向量在空间中夹角的余弦值。
  * **代码任务**:
      * **具身场景模拟**:
          * 指令库: `["向左转", "向右转", "抓取物体", "停止"]`
          * 用户输入: "请把那个东西拿起来"
      * 计算用户输入与指令库中每个指令的相似度，找出匹配度最高的动作。
  * **🤖 具身视角**: 这是实现**零样本 (Zero-shot) 意图识别**的最快路径。不需要训练分类器，直接比对语义距离。

### Day 5: Datasets 库实战 (Hugging Face Datasets)

  * **文件名**: `S2W6D5_HuggingFace_Datasets.ipynb`
  * **理论**:
      * **Apache Arrow**: 为什么 Datasets 库读写几个 GB 的数据几乎不占内存？(Memory Mapping)。
      * `map` 函数：并行处理数据的神器。
  * **代码任务**:
      * 加载 `ChnSentiCorp` (中文情感分析数据集) 或类似 Demo 数据。
      * 编写预处理函数 `preprocess_function(examples)`。
      * 使用 `dataset.map(preprocess_function, batched=True)` 将文本批量转换为 Input IDs。

### Day 6: 动态 Padding 与 Collate\_fn (The Engineering Core)

  * **文件名**: `S2W6D6_Dynamic_Padding.ipynb`
  * **理论**:
      * **静态 Padding**: 整个数据集按最大长度补齐 (浪费显存)。
      * **动态 Padding**: 每个 Batch 内部按当前 Batch 的最长样本补齐 (高效)。
  * **代码任务**:
      * **手写**: 定义 `def collate_fn(batch):`。这是 PyTorch 面试中考察工程能力的**高频考点**。
      * 将其传入 `DataLoader(..., collate_fn=collate_fn)`。
      * 验证：打印不同 Batch 的 Input Shape，确认它们的长度是动态变化的。

### Day 7: 复盘与模拟面试 (Review & Interview)

  * **文件名**: `S2W6D7_Pipeline_and_Review.ipynb`
  * **复盘**:
      * 将 Day 1-6 的代码片段整合成一个微型的推理脚本：`Input Text -> Tokenizer -> Model -> Embedding -> Similarity`。
  * **模拟面试 (Self-Check)**:
    1.  `bert-base-chinese` 的词表大小是多少？(约 21128)。
    2.  为什么 BERT 的输入长度限制通常是 512？(位置编码是学习出来的/固定的，且 Attention 是 $O(L^2)$ 复杂度，太长会显存爆炸)。
    3.  `DataCollatorWithPadding` 和你自己写的 `collate_fn` 有什么区别？

-----

## 🛠️ 环境准备

请在终端运行以下命令，确保安装了必要的库 (如果是本地环境)：

```bash
pip install transformers datasets
```

同时，请检查 `src/utils.py`，如果里面有通用的绘图函数或日志函数，本周可以直接调用，保持代码复用。