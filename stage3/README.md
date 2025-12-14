你好，黄同学！恭喜你完成了 Stage 2 的硬仗。能手动实现 BERT 的推理流程并解决类别不平衡问题，说明你已经具备了从“调包侠”向“算法工程师”转型的潜力。

Stage 3 是目前求职市场上**最卷但含金量最高**的领域。对于算法岗，面试官不会只满足于你会调 OpenAI 的 API，他们更看重你对 **RAG 链路中数据处理、检索优化以及模型幻觉控制** 的理解。

以下是为你定制的 **Stage 3：LLM 与 RAG 应用开发 (4周冲刺计划)**。

-----

### 📂 Stage 3 工程结构更新

在开始之前，我们需要在 `src/` 下扩展新的模块，保持工程整洁（这是面试加分项）：

```text
src/
├── ... (原有的 dataset, models 等)
├── llm/              <-- 新增：负责与大模型交互
│   ├── __init__.py
│   ├── client.py     # 封装 OpenAI/Local LLM API 调用
│   └── prompts.py    # 集中管理 Prompt Templates
├── rag/              <-- 新增：负责检索增强生成核心逻辑
│   ├── __init__.py
│   ├── embedding.py  # 文本向量化接口
│   ├── vector_db.py  # 向量数据库操作 (Chroma/Faiss)
│   └── retriever.py  # 检索逻辑实现
└── utils.py          # 更新：增加文本清洗工具
```

-----

### 📅 Stage 3 详细排期 (4 Weeks)

#### Week 8: LLM 基础与 Prompt 工程 (The "Input" Layer)

**目标**：把 LLM 当作一个通用的函数调用，理解 Prompt 如何影响输出。
**类比医学图像**：Prompt Engineering 就像是给分割模型指定 **ROI (感兴趣区域)** 或者提供 **先验知识 (Prior Knowledge)**，你给的提示越准，模型分割（生成）的结果就越好。

  * **Day 1: 环境搭建与 API 封装**
      * **任务**：注册 OpenAI API (或使用国内 DeepSeek/Moonshot 等替代)。
      * **代码**：在 `src/llm/client.py` 中封装一个 `LLMClient` 类，支持 `generate(prompt)` 方法。
      * **面试点**：Temperature 参数代表什么？(创造性 vs 确定性)。
  * **Day 2: Prompt Engineering 核心范式**
      * **任务**：学习 Zero-shot, Few-shot (In-context Learning), CoT (思维链)。
      * **实践**：在 `src/llm/prompts.py` 中编写针对“法律条款解释”的 Prompt 模板。
  * **Day 3: 结构化输出 (Structured Output)**
      * **任务**：强迫 LLM 输出 JSON 格式。
      * **痛点**：模型通过正则表达式解析很难，如何保证稳定？(Function Calling / Pydantic parser)。
  * **Day 4: LangChain/LangGraph 入门**
      * **任务**：理解 LangChain 的核心概念：`Chain`, `PromptTemplate`。
      * **代码**：用 LangChain 重写 Day 1 的调用逻辑。
  * **Day 5: 简单的本地 Chatbot**
      * **项目**：做一个命令行工具，输入法律问题，LLM 返回结构化的 JSON 答案。
  * **Day 6-7: 算法题复习 + 总结**
      * **LeetCode**: 栈与队列 (LC20, LC232)。
      * **面试模拟**：什么是 In-context Learning？它和 Fine-tuning 有什么本质区别？(不用更新梯度 vs 更新参数)。

#### Week 9: RAG 核心 - 数据处理与向量化 (The "Memory" Layer)

**目标**：构建外部知识库。
**类比医学图像**：这就像 **Atlas-based Segmentation (基于图谱的分割)**。当模型自己看不清（不知道知识）时，去数据库里找一张最像的“标准图谱”（参考文档）贴上去辅助判断。

  * **Day 1: 文本切分 (Chunking) 策略**
      * **任务**：研究 `RecursiveCharacterTextSplitter`。
      * **思考**：切分块太大丢失细节，太小丢失上下文。如何选择 chunk\_size？(建议 512-1000)。
  * **Day 2: Embedding 原理**
      * **任务**：使用 `sentence-transformers` (如 `bge-m3` 或 `m3e`) 将文本转为向量。
      * **代码**：实现 `src/rag/embedding.py`。
      * **面试点**：为什么用 Cosine Similarity 计算相似度？(方向一致性)。
  * **Day 3: 向量数据库 (Vector DB)**
      * **任务**：部署 ChromaDB 或 FAISS。
      * **代码**：实现 `src/rag/vector_db.py`，包含 `add_documents` 和 `search` 方法。
  * **Day 4: 法律文档 ETL 管道**
      * **项目**：下载几部法律法规 (PDF/TXT)，清洗 -\> 切分 -\> Embedding -\> 存入 ChromaDB。
  * **Day 5: 检索评估**
      * **任务**：输入一个问题，看检索出的 Top-3 文档片段是否真的相关。
      * **优化**：尝试调整 chunk overlap (重叠部分) 解决边界语义丢失问题。
  * **Day 6-7: 算法题复习 + 总结**
      * **LeetCode**: 链表操作 (LC206, LC21)。
      * **面试模拟**：如果检索到的文档切片不仅没用，反而有误导性，RAG 会怎么处理？(会产生幻觉，引出 Week 10 的优化)。

#### Week 10: RAG 串联与生成 (The "Logic" Layer)

**目标**：完成完整的 RAG 闭环。
**类比医学图像**：这就是 **U-Net 的 Skip Connection**。Input (Query) 经过 Encoder (Embedding) 找到特征，然后与 Decoder (LLM) 结合，把检索到的特征 (Context) "Concat" 进去，生成最终结果。

  * **Day 1: Naive RAG 流程实现**
      * **代码**：在 `src/engine.py` 中实现：Query -\> Retrieve -\> Prompt Assembly (Query+Context) -\> LLM -\> Answer。
  * **Day 2: 检索优化 - Query Rewriting**
      * **问题**：用户提问可能是“它怎么判？”，指代不明。
      * **解法**：用 LLM 先把问题改写成“盗窃罪如何量刑？”，再去检索。
  * **Day 3: 检索优化 - Re-ranking (重排序)**
      * **概念**：向量检索是“粗排”，需要一个 Cross-Encoder 模型做“精排”。
      * **实践**：引入 `bge-reranker` 对检索结果进行二次排序。
  * **Day 4: 工程化落地 - 法律助手 Demo**
      * **任务**：整合 Week 8-10 的代码，提供一个 CLI 或简单的 Gradio 界面。
      * **功能**：上传一个法律文档，然后针对该文档提问。
  * **Day 5: 幻觉检测与评估**
      * **概念**：RAGAS 指标 (Context Recall, Faithfulness)。
      * **思考**：如何知道模型是在“胡说八道”？(引用归因：让模型标出答案来自第几条文档)。
  * **Day 6-7: 算法题复习 + 总结**
      * **LeetCode**: 二叉树基础 (LC104, LC102)。
      * **面试模拟**：详细画出 RAG 的 pipeline，并指出最容易出现性能瓶颈的环节在哪里？(通常是 Embedding 的质量和检索的准确率)。

#### Week 11: Agent 初探与项目复盘 (Advanced)

**目标**：让模型具备使用工具的能力。

  * **Day 1-2: Tool Use / Function Calling**
      * **任务**：定义一个“计算器”函数或“联网搜索”函数。让 LLM 决定何时调用函数，何时直接回答。
  * **Day 3-5: 项目重构与文档撰写**
      * **任务**：完善 `README.md`，画出架构图（架构图是你面试时的讲解地图）。
      * **代码**：确保 `src` 下所有代码符合 PEP8 规范，添加注释。
  * **Day 6-7: 模拟面试冲刺**
      * **重点**：针对 RAG 项目进行深挖（为什么选 Chroma？为什么选这个 Chunk size？）。

-----

### 💡 导师的一句话建议 (One More Thing)

进入 Stage 3 后，你会发现代码量不如 Stage 2 大，但**逻辑链条 (Pipeline)** 变得极长。