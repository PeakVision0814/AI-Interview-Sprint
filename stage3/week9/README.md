# 🚀 Stage 3 Week 9: RAG Core - The "Long-Term Memory"

**本周核心目标**：
构建 **"Legal Assistant v0.2"** 的数据底座。不仅仅是简单的“搜索”，而是要理解**非结构化文本（Unstructured Text）**如何被转化为计算机可计算的**稠密向量（Dense Vectors）**，并实现高效检索。

**CS 原理视角**：

* **RAG 本质**：这是一个**开卷考试（Open Book Exam）**系统。LLM 是考生（推理能力强，但记忆有限），Vector DB 是参考书（知识完备）。
* **数学本质**：将高维语义空间中的 Query 向量，通过**近似最近邻搜索（ANN, Approximate Nearest Neighbor）**，找到距离最近的 Document 向量。

---

## 🗓️ 每日任务清单 (Daily Tasks)

### Day 1: 文本切分策略 (Chunking Strategies)

* **目标**：解决 LLM 上下文窗口限制与信息精度的权衡问题。
* **CS 核心概念**：**Granularity (粒度)**。
* 切分太粗（Chunk size 大）：包含太多无关噪声，检索准确率下降（类似于数据库查询返回了整张表而不是需要的行）。
* 切分太细（Chunk size 小）：语义截断，丢失上下文（Context Fragmentation）。


* **核心任务**：
1. **实现切分器**：在 `src/rag/etl.py` 中使用 LangChain 的 `RecursiveCharacterTextSplitter`。
2. **对比实验**：
* 设置 `chunk_size=500` vs `chunk_size=2000`。
* 设置 `chunk_overlap=0` vs `chunk_overlap=100`（Overlap 就像 TCP 数据包的冗余校验，防止边界语义丢失）。


3. **LeetCode 必练**：**LC 206. Reverse Linked List** (链表基础，热身)。


* **代码焦点**：
* `stage3/week9/S3W9D1_Chunking_Exp.ipynb`


* **💡 面试必问**：
* *“在 RAG 中，Chunking 对检索效果有什么具体影响？对于法律文档（条款之间有强关联），你会选择什么样的切分策略？”*



### Day 2: 向量化原理 (Embedding & Latent Space)

* **目标**：将自然语言映射到数学空间。
* **CS 核心概念**：**Feature Extraction (特征提取)** 与 **Dimensionality Reduction (降维)**。
* Embedding 模型（如 BERT/BGE）是一个编码器，将变长文本压缩为定长向量（例如 768 维）。
* **语义相似度 = 向量方向的一致性**（Cosine Similarity）。


* **核心任务**：
1. **环境配置**：安装 `sentence-transformers`。
2. **模型选择**：下载中文友好的 embedding 模型（推荐 `BAAI/bge-m3` 或 `moka-ai/m3e-base`）。
3. **计算相似度**：
* 计算 "诈骗罪量刑标准" 与 "盗窃罪怎么判" 的余弦相似度（应较低）。
* 计算 "诈骗罪量刑标准" 与 "虚构事实骗取财物如何处罚" 的相似度（应较高）。


4. **LeetCode 必练**：**LC 141. Linked List Cycle** (快慢指针)。


* **代码焦点**：
* `src/rag/embedding.py`
* 实现一个单例类 `EmbeddingModel`，避免重复加载模型。


* **💡 面试必问**：
* *“为什么计算向量相似度常用 Cosine Similarity 而不是 Euclidean Distance（欧氏距离）？在什么情况下两者是等价的？（提示：当向量归一化后）”*



### Day 3: 向量数据库部署 (Vector Database)

* **目标**：实现海量向量的高效索引。
* **CS 核心概念**：**Inverted Index (倒排索引)** vs **Dense Vector Index (稠密向量索引)**。
* 传统搜索（Elasticsearch）基于关键词匹配（TF-IDF/BM25）。
* 向量搜索基于空间距离（HNSW 算法 - Hierarchical Navigable Small World）。


* **核心任务**：
1. **选型**：为了轻量化开发，使用 **ChromaDB** (本地文件存储) 或 **FAISS** (Facebook AI Similarity Search)。
2. **CRUD 操作**：实现 `src/rag/vector_db.py`。
* `add_documents(texts, metadatas)`
* `query(query_text, top_k)`


3. **LeetCode 必练**：**LC 21. Merge Two Sorted Lists** (经典归并思想)。


* **代码焦点**：
* `stage3/week9/S3W9D3_VectorDB_Demo.ipynb`


* **💡 面试必问**：
* *“简述 HNSW 索引的基本原理？它通过什么数据结构牺牲内存换取了搜索速度？”（提示：跳表 Skip List + 图 Graph）*



### Day 4: 法律文档 ETL 管道 (The Pipeline)

* **目标**：处理脏数据，构建真实的法律知识库。
* **CS 核心概念**：**ETL (Extract, Transform, Load)**。
* **核心任务**：
1. **数据源**：下载《中华人民共和国刑法》或《民法典》的纯文本/PDF。
2. **清洗 (Transform)**：编写 Regex 去除页眉、页脚、无意义的换行符。
3. **入库 (Load)**：
* Pipeline: `Raw Text` -> `Chunks` -> `Vectors` -> `ChromaDB`。
* **Metadata**：在存入向量库时，务必带上元数据（如 `{"source": "刑法", "article_id": "266"}`），以便后续溯源。


4. **LeetCode 必练**：**LC 20. Valid Parentheses** (栈的经典应用，对应 JSON 括号匹配检查)。


* **代码焦点**：
* `src/rag/ingest.py` (运行此脚本完成数据入库)。



### Day 5: 检索评估与优化 (Retrieval Evaluation)

* **目标**：验证“查得准不准”。
* **CS 核心概念**：**Precision (查准率)** vs **Recall (查全率)**。
* 在 RAG 中，我们更关注 **Hit Rate@K** (Top-K 结果里是否包含正确答案)。


* **核心任务**：
1. **构建测试集**：手动准备 10 个法律问题和对应的正确法条。
2. **自动化测试**：
* 输入问题，获取 Top-3 chunks。
* 检查正确法条是否在 Top-3 中。


3. **Debug 分析**：如果没检索到，是 Embedding 语义没对齐，还是 Chunk 把法条切断了？
4. **LeetCode 必练**：**LC 155. Min Stack** (辅助栈的设计思想)。


* **代码焦点**：
* `stage3/week9/S3W9D5_Eval.ipynb`



### Day 6: 集成 - 法律小助手 v0.2 (Integration)

* **目标**：将 RAG 模块接入之前的 CLI 工具。
* **核心任务**：
1. **修改 System Prompt**：增加指令——*“请仅根据以下提供的【参考上下文】回答问题，如果上下文中没有答案，请说不知道，严禁编造。”*
2. **流程串联**：
* User Input -> Embedding -> Vector Search -> Top-K Chunks。
* Prompt = System Prompt + Context(Top-K) + User Input。
* LLM Generation。


3. **交付物**：运行 `python stage3/week9/app_v2.py`，提问“盗窃罪数额巨大是多少钱？”，它应该能引用具体的司法解释金额。
4. **LeetCode 必练**：**LC 232. Implement Queue using Stacks** (双栈模拟)。



### Day 7: 深度复盘与模拟面试 (Deep Review)

* **目标**：理论体系化。
* **复习重点**：
* **稀疏检索 vs 稠密检索**：关键词匹配 (BM25) 和 向量匹配 (Embedding) 所谓“词法鸿沟”和“语义鸿沟”的区别。
* **维度灾难**：为什么向量维度太高（如 10000 维）会导致欧氏距离失效？


* **LeetCode 必练**：**周赛复盘** 或 **LC 142. Linked List Cycle II** (寻找环入口，数学推导)。
* **自我介绍优化**：
* 更新项：*“具备 RAG 全流程开发经验，熟悉从非结构化文档处理、Embedding 降维到向量数据库（Chroma/FAISS）索引构建的完整 ETL 链路。”*



---

## 4. 下一步 (Next Step)

高朋，我已经根据你的要求，移除了所有非 CS 的类比，并特别强化了**数据结构（HNSW、倒排索引）**和**向量计算（Cosine Similarity）**的原理部分。

**你现在的首要任务是**：
在你的本地 `AI-Interview-Sprint/stage3/week9/` 目录下创建 `README.md`，并将上述 **Daily Tasks** 复制进去作为本周的行动指南。

**是否需要我为你提供 Day 1 `RecursiveCharacterTextSplitter` 和基础切分实验的 Python 代码模版？** (这可以帮你节省查文档的时间，直接开始跑实验)。