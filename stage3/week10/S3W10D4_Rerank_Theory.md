# S3W10D4: RAG 的精排大脑 —— Cross-Encoder

## 1. 核心矛盾：速度 vs 精度

在面试中，当面试官问你：“为什么不能直接用向量检索的结果？”或者“为什么有了向量检索还需要 Re-ranker？”，你可以画出下面这张图来解释：

### 1.1 Bi-Encoder (我们目前用的向量检索)

* **架构**: 双塔模型 (Two Towers)。Query 和 Document 分别通过 BERT，互不干扰。
* **计算**: `Cosine_Similarity(Vector_Q, Vector_D)`。
* **优点**: **快！** 文档向量可以离线算好存起来。检索时只是简单的矩阵运算。
* **缺点**: **“眼神交流少”**。Query 和 Doc 直到最后一步才计算相似度，模型无法理解两者之间细腻的语法交互。它只能捕捉大概的语义（vibe）。
* *例子*: "I love Python" 和 "I hate Python" 在向量空间可能离得很近（共享大部分词），但语义完全相反。



### 1.2 Cross-Encoder (我们要引入的重排序)

* **架构**: 单塔模型。
* **输入**: 把 Query 和 Document 拼在一起喂给 BERT：
`[CLS] Query [SEP] Document [SEP]`
* **机制**: **全注意力交互 (Full Self-Attention)**。Query 里的每一个 token 都能看到 Document 里的每一个 token。模型可以深度“审题”。
* **优点**: **准！** 能够处理逻辑否定、指代消歧等复杂语义。
* **缺点**: **慢！** 无法预计算。每次新的 Query 来了，都要把所有候选文档重新跑一遍 BERT。

---

## 2. 解决方案：漏斗模式 (The Funnel)

既然 Cross-Encoder 慢，我们不能对库里 100 万个文档都跑 Cross-Encoder。
标准做法是 **“粗排 + 精排”** 的漏斗结构：

1. **Retrieve (粗排)**: 使用 Bi-Encoder (向量库) 快速从 100 万个文档中捞出 **Top-100**。
2. **Re-rank (精排)**: 使用 Cross-Encoder 对这 **Top-100** 进行深度打分。
3. **Cut-off**: 取精排后的 **Top-3** 喂给 LLM。

> **面试金句**: “Bi-Encoder 负责召回率 (Recall)，保证相关文档不被漏掉；Cross-Encoder 负责准确率 (Precision)，保证给 LLM 的都是精华。”

---

## 3. 调研模型：BAAI/bge-reranker

目前中文社区最强的开源 Re-ranker 之一是智源（BAAI）发布的 BGE 系列。

**Action**: 请访问 [Hugging Face - bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) (或者 v2-m3)。

**在阅读 Model Card 时，请重点关注以下几点（并在你的笔记中记录）：**

1. **Usage (用法)**:
    * 它的输入格式是什么？(通常是 `pairs = [['query', 'doc1'], ['query', 'doc2']]`)
    * 它的输出是什么？(通常是一个 Scalar Score，分数越高越相关)。


2. **Max Length (最大长度)**:
    * 因为是拼接 Query + Doc，总长度不能超过多少？(通常是 512)。这意味着如果你的 Chunk 很大，可能会被截断。


3. **多语言支持**:
    * `bge-reranker-v2-m3` 支持多语言，如果你的文档里夹杂英文，这个版本更好。



---

## 4. 你的任务清单 (To-Do)

今天不需要写完整代码，但请准备好环境，明天我们要实战。

1. **笔记**: 记录 Bi-Encoder vs Cross-Encoder 的区别。
2. **阅读**: 花 10 分钟快速浏览 BGE 的文档。
3. **思考**:
    * 如果向量检索返回的 Top-1 是：“小明喜欢吃苹果”，Query 是：“小明不喜欢吃什么？”，向量检索可能会认为这句话很相关（都有小明、吃）。
    * Cross-Encoder 会给这句话打高分还是低分？（答案应该是低分，因为它能理解“不喜欢”和“喜欢”的冲突）。