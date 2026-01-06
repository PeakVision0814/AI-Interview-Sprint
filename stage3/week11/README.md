# Stage 3 Week 11: Agentic RAG & System Refactor (给 AI 装上“手”)

## 🎯 本周目标 (Objectives)
1.  **Agent 原理 (First Principles)**: 理解 LLM 如何通过 "Function Calling" (工具调用) 与外部世界交互。这本质上是把自然语言分类并映射为 API 参数的过程。
2.  **Tool Use 实战**: 利用 SiliconFlow API，构建一个不仅能聊天，还能“算数”或“查询实时信息”的 Mini-Agent。
3.  **工程重构 (Refactoring)**: 告别 Spaghetti Code (面条代码)。将 RAG 流程封装进 `src/rag` 和 `src/llm`，确保代码可维护、可扩展。
4.  **面试文档化**: 输出一份高质量的架构图和 README，这是面试官最先看到的东西。

## 📅 日程安排 (Schedule)

### Day 1: Agent 的本质与 Function Calling 原理
* **Theme**: 从文本生成到结构化决策。
* **Concept**:
    * LLM 并不真的“运行”代码，它只是“生成”要运行的代码或参数（JSON）。
    * **比喻**: LLM 是主厨，Tools 是帮厨。主厨不亲自切菜，他写一张单子（JSON）给帮厨，帮厨切好后把结果（Observation）回报给主厨，主厨再决定下一步。
* **Project (Notebook)**: `S3W11D1_Function_Calling_Basics.ipynb`
    * 使用 SiliconFlow API 定义一个简单的工具 Schema (如 `add_numbers`, `get_weather`)。
    * 手动实现工具调用的 "Loop"：System Prompt -> LLM 决策 -> 解析 JSON -> 执行 Python 函数 -> 结果回填 LLM。
* **Algorithm (Tree Construction)**:
    * **LC 105. Construct Binary Tree from Preorder and Inorder Traversal** (中等 - 经典题目，考察对递归结构的理解)。

### Day 2: 赋予 RAG 工具使用能力 (Agentic RAG)
* **Theme**: 当 RAG 遇到 Agent。
* **Project (Code)**: `src/llm/tools.py` & `src/rag/agent.py`
    * 定义一个 `SearchTool` (其实就是调用我们之前的 VectorDB 检索)。
    * 实现逻辑：用户提问 -> LLM 判断是“闲聊”还是“需要查库” -> 如果查库，调用 `SearchTool` -> Rerank -> 生成回答。
    * **目标**: 让系统不再傻傻地每一句话都去查库。
* **Algorithm (BST 基础)**:
    * **LC 98. Validate Binary Search Tree** (中等 - 验证二叉搜索树，注意区间限制的传递)。

### Day 3: 工程重构 - 模块化 (The "Clean" Code)
* **Theme**: 像工程师一样思考，而不是学生。
* **Action**: 将 Week 8-10 分散在 Notebook 中的 RAG 逻辑整合。
    * 完善 `src/rag/engine.py`: 封装 `RAGPipeline` 类。
    * 完善 `src/config.py`: 统一管理 `CHUNK_SIZE`, `EMBEDDING_MODEL` 等超参数。
    * **Git**: 提交一次规范的 Refactor Commit。
* **Algorithm (BST 操作)**:
    * **LC 701. Insert into a Binary Search Tree** (中等 - 相对简单，增强信心)。
    * **LC 450. Delete Node in a BST** (中等 - 难点，涉及节点替换逻辑，必须手写一遍)。

### Day 4: 系统架构文档与可视化 (The Interview Map)
* **Theme**: 让面试官一眼看懂你的系统。
* **Project (Docs)**: 完善根目录 `README.md`。
    * **架构图**: 画出 User -> Query -> Router (Agent) -> VectorDB/Tools -> Reranker -> LLM -> Response 的流程图 (推荐用 Mermaid 或 Excalidraw)。
    * **技术选型理由**: 为什么用 Chroma? 为什么加 Reranker? (准备好话术)。
* **Algorithm (Tree Properties)**:
    * **LC 110. Balanced Binary Tree** (简单 - 很多大厂的面试热身题)。
    * **LC 236. Lowest Common Ancestor of a Binary Tree** (中等 - LCA 是二叉树面试的**必考题**，必须熟练背诵递归逻辑)。

### Day 5: 综合调试与端到端测试
* **Theme**: 跑通整个 Pipeline。
* **Project (Runner)**: 创建 `main.py` 或 `run_agent.py`。
    * 支持命令行交互：`python main.py --query "transformer的原理是什么"`。
    * 测试：故意问一些不在知识库的问题，看 Agent 是否能依靠自身知识回答（而不是强行检索）。
* **Algorithm (Tree Path)**:
    * **LC 112. Path Sum** (简单 - 复习 DFS)。
    * **LC 437. Path Sum III** (中等 - 路径不需要从根节点开始，结合前缀和思想，**高频题**)。

### Day 6: 模拟面试冲刺 (Mock Interview - RAG Special)
* **Theme**: 自我拷打。
* **Activity**: 针对以下问题整理回答（写在 `S3W11D6_Interview_Prep.md`）：
    1.  *“你的 RAG 系统中，Chunk Size 设为多少？为什么？”*
    2.  *“如果检索回来的内容不相关，你的系统怎么处理？”* (答案点：Reranker 的分数阈值过滤)。
    3.  *“Function Calling 的底层原理是什么？LLM 怎么知道调用哪个函数？”* (答案点：微调数据格式、System Prompt 中的 Schema 描述)。
* **Algorithm (Review)**:
    * 复习本周做错的树题目。

### Day 7: 阶段复盘与 Git 整理
* **Review**: 检查 `src/` 目录下的代码风格 (Pylint/Flake8)。
* **Git**: 确保所有更改已 Commit，Push 到远程仓库（如有）。
* **Rest**: 这是一个高强度的阶段结束，好好休息，下周我们将进入 **Stage 4: 项目部署与简历打磨**。

## 🛠️ 环境准备 (Environment)
* 确保 `siliconflow` 的 API Key 已配置在 `.env` 中。
* 安装可视化工具库 (可选): `pip install graphviz` (如果要在 Python 里画图)。