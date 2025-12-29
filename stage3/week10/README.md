# Stage 3 Week 10: Deep Learning Math & RAG Refinement
**Owner**: Huang Gaopeng  
**Role**: AI Algorithm Engineer Candidate  
**Status**: In Progress  
**Focus**: Backpropagation, Optimizers, RAG Re-ranking, Binary Trees

## 🎯 Weekly Objectives (本周目标)
1.  **Math "Internal Strength" (数学内功)**:
    * 能够手推全连接层 (Linear Layer) 的反向传播 (BP) 过程。
    * 深刻理解“梯度消失/爆炸”的数学成因及解决方案 (Sigmoid vs ReLU, BatchNorm)。
    * 理解 SGD, Momentum, Adam 的核心区别。
2.  **RAG Enhancement (精排优化)**:
    * 理解 Bi-Encoder (向量检索) 与 Cross-Encoder (重排序) 的区别。
    * 在现有 RAG 流程中引入 `Re-ranker` 模块，提高 Context 准确性。
3.  **Algorithm (数据结构)**:
    * 掌握二叉树的基础遍历 (DFS/BFS)。

---

## 📅 Daily Schedule (每日安排)

### Day 1: The Calculus of Backpropagation (BP 推导日)
* **Theme**: 痛苦但必须经历的手推公式。
* **Theory**:
    * 复习链式法则 (Chain Rule)。
    * **Core Task**: 在纸上推导一个简单的 2 层神经网络 (Input -> Linear -> Sigmoid -> Linear -> Loss) 的梯度更新公式。
    * Key Question: 当输入维度是 $(N, D_{in})$，输出是 $(N, D_{out})$ 时，梯度的维度是多少？
* **Action**:
    * 创建 `S3W10D1_Backprop_Derivation.md` 记录推导过程（拍照或手写 LaTeX）。
* **LeetCode**: LC 104. Maximum Depth of Binary Tree (Easy - 递归热身).

### Day 2: Gradient Problems & Activations (梯度与激活函数)
* **Theme**: 为什么深层网络难训练？
* **Theory**:
    * 分析 Sigmoid 导数图像（最大值 0.25），解释其导致的梯度消失 (Vanishing Gradient)。
    * 对比 ReLU 的导数特性。
* **Coding**:
    * 创建 `S3W10D2_Gradient_Viz.ipynb`。
    * 构建一个简单的深层网络（10层+），分别使用 Sigmoid 和 ReLU，打印每层的梯度均值，观察“梯度消失”现象。
* **LeetCode**: LC 226. Invert Binary Tree (翻转二叉树 - 经典题).

### Day 3: Optimizers - SGD vs Adam (优化器之战)
* **Theme**: 为什么 Adam 是“炼丹”首选？
* **Theory**:
    * **SGD**: 随机梯度下降的缺点（震荡、卡在鞍点）。
    * **Momentum**: 动量概念（惯性）。
    * **Adam**: 结合 Momentum + RMSprop (自适应学习率)。不需要背复杂公式，但要懂核心思想：一阶动量（方向），二阶动量（步长缩放）。
* **Coding**:
    * 在 `S3W10D2` 的 Notebook 中，对比使用 `torch.optim.SGD` 和 `torch.optim.Adam` 的 Loss 下降曲线。

### Day 4: RAG Logic - The Need for Re-ranking (重排序理论)
* **Theme**: 向量相似度 $\neq$ 语义相关度。
* **Theory**:
    * 回顾 Naive RAG 的痛点：向量库召回 Top-K，但第 1 名可能不是最相关的（只是字面相似）。
    * **Cross-Encoder**: 为什么它比 Bi-Encoder 准但更慢？(Input: `[CLS] Query [SEP] Doc [SEP]`)。
    * 调研模型：`BAAI/bge-reranker-base` 或 `bge-reranker-v2-m3`。
* **Action**:
    * 阅读 BGE Reranker 的 Hugging Face Model Card。

### Day 5: Implementing RAG Re-ranker (实战重排序)
* **Theme**: 给 RAG 加上“审题”模块。
* **Coding**:
    * 创建 `src/rag/reranker.py`。
    * 封装一个 `RerankClient` 类，使用 Hugging Face `CrossEncoder` 或 `AutoModelForSequenceClassification`。
    * **Integration**: 修改 `src/rag/engine.py` (或你的主流程)，在 `vector_db.search()` 之后，接入 `reranker.rank()`，从 Top-10 筛选出 Top-3。
* **Deliverable**: 运行一个 Query，对比加上 Rerank 前后的检索结果。

### Day 6: Binary Tree BFS & Review (广度优先搜索)
* **Theme**: 队列 (Queue) 的应用。
* **LeetCode**:
    * LC 102. Binary Tree Level Order Traversal (层序遍历 - 中等重点)。
    * 体会 Queue 在 BFS 中的作用。
* **Review**:
    * 整理本周的数学笔记。
    * 检查 `src/rag/reranker.py` 是否已提交到 Git。

### Day 7: Weekly Summary & Rest
* **Check**:
    * [ ] 能否口述 BP 的核心逻辑？
    * [ ] RAG 系统是否已经集成了 Re-ranking？
    * [ ] 二叉树题目是否熟练（递归 vs 迭代）？
* **Plan Next**: 准备 Week 11 的 Tool Use (Agent 工具调用)。

---

## 📚 Resources
* **Paper**: "Neural Machine Translation by Jointly Learning to Align and Translate" (Attention 基础回顾)
* **Blog**: Jay Alammar - Visualizing A Neural Machine Translation
* **Docs**: Hugging Face `sentence-transformers` Cross-Encoder documentation.