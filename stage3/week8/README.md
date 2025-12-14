没问题，黄同学！

你是对的，**“磨刀不误砍柴工”**。虽然我们已经开始触碰代码，但拥有一张清晰的 **Week 8 作战地图** 能让你知道每一行代码在整个知识体系中的位置，避免陷入“盲目调包”的陷阱。

Week 8 是 Stage 3 的基石。我们将从“模型微调”的思维切换到“**Prompt Engineering (提示工程) + LLM Native Development**”的思维。

以下是为你定制的 **Stage 3 Week 8 详细任务清单**。

-----

# 📅 Stage 3 Week 8: LLM 基础与 Prompt 工程

**核心目标**：构建一个 **“具备结构化输出能力的法律咨询助手 (v0.1)”**。
**技术栈**：`OpenAI API Standard` (适配 DeepSeek/Moonshot) + `LangChain (Core)` + `Pydantic`。
**简历卖点**：深入理解 Prompt Engineering 范式（CoT, Few-Shot），并解决了 LLM 在工程落地中“输出不稳定”和“上下文限制”的核心痛点。

### 📂 本周工程文件结构 (Project Structure)

请在 `stage2` 同级目录下创建 `stage3`：

```text
project_root/
├── src/                        # 沿用根目录 src，本周新增 llm 模块
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py           # (已创建) 统一 API 客户端
│   │   ├── prompts.py          # Prompt 模板管理
│   │   ├── parsers.py          # 输出解析器 (String -> JSON)
│   │   └── memory.py           # 简单的对话历史管理
│   └── utils.py
├── stage3/
│   ├── week8/
│   │   ├── data/               # 存放少量的测试用法律文本
│   │   ├── S3W8D1_API_Basics.ipynb
│   │   ├── S3W8D2_Prompt_Engineering.ipynb
│   │   ├── S3W8D3_Structured_Output.ipynb
│   │   ├── S3W8D4_LangChain_Intro.ipynb
│   │   ├── S3W8D5_Chatbot_Logic.ipynb
│   │   ├── app.py              # 最终产出的命令行工具
│   │   └── README.md
│   └── README.md
└── requirements.txt
```

-----

## 🗓️ 每日任务清单 (Daily Tasks)

### Day 1: 统一接口封装 (The Universal Adapter) [✅ 已启动]

  * **目标**：构建“模型无关”的调用层，无论后端是 GPT-4 还是本地 Llama 3，业务代码无需修改。
  * **核心任务**：
    1.  配置 API Key 和 Environment Variables (`.env`)。
    2.  封装 `LLMClient` 类，实现基础的 `generate(text)` 方法。
    3.  测试 `Temperature` 参数对输出结果的影响（从“复读机”到“狂想家”）。
  * **代码焦点**：
      * `src/llm/client.py`。
      * Error Handling (网络超时、Key 错误怎么处理)。
  * **💡 面试必问**：
      * *“Temperature 设置为 0 和 1 的区别是什么？数学上它是如何影响 Softmax 分布的？”*
      * *“OpenAI 的 API 是无状态的 (Stateless)，这意味着什么？”*

### Day 2: 提示工程范式 (Prompt Paradigms)

  * **目标**：学会“如何好好说话”。把 Prompt 当作模型的“初始化权重”。
  * **核心任务**：
    1.  **Zero-shot vs. Few-shot (ICL)**：对比直接提问和给3个示例后的效果。
    2.  **Chain of Thought (CoT)**：在 Prompt 中强制加入 *"Let's think step by step"*，观察逻辑推理能力的提升。
    3.  **System Prompt**：设定角色（“你是一名严谨的资深律师...”）。
  * **代码焦点**：
      * `src/llm/prompts.py`：构建可复用的 Prompt 模板类。
  * **类比医学图像**：
      * **Few-shot** 就像 **Atlas-based Segmentation**（给几个参考病例，模型照着切）。
  * **💡 面试必问**：
      * *“什么是 In-Context Learning (ICL)？它和 Fine-tuning 有什么本质区别？（提示：是否更新参数）”*

### Day 3: 结构化输出与解析 (Structured Output)

  * **目标**：驯服 LLM，让它不再输出“自然语言”，而是输出“机器可读代码 (JSON)”。
  * **核心任务**：
    1.  **Regex 方法**：用正则表达式从回答中提取信息（痛苦且不稳定）。
    2.  **JSON Mode / Format**：在 Prompt 中强制要求 JSON 格式。
    3.  **Pydantic 验证**：定义 Python 类来验证 LLM 返回的数据结构是否合法。
  * **场景模拟**：
      * 输入：“张三偷了李四 5000 元。” -\> 输出：`{"suspect": "张三", "amount": 5000, "crime_type": "theft"}`
  * **💡 面试必问**：
      * *“如果模型生成的 JSON 少了一个括号，工程上怎么容错？(Output Parser / Retry 机制)”*

### Day 4: LangChain 入门 (The Orchestrator)

  * **目标**：理解为什么需要框架，而不是一直手写 API 调用。
  * **核心任务**：
    1.  理解 `LCEL` (LangChain Expression Language)：`Prompt | Model | Parser` 的流式写法。
    2.  使用 `ChatPromptTemplate` 管理多轮对话格式。
    3.  用 LangChain 重写 Day 1-3 的逻辑。
  * **代码焦点**：
      * `from langchain_core.prompts import ChatPromptTemplate`
  * **💡 面试必问**：
      * *“LangChain 的核心价值是什么？(组件抽象、链式调用、生态集成)”*

### Day 5: 记忆机制与上下文 (Memory Management)

  * **目标**：让 AI 记住“我们刚才聊了什么”。
  * **核心任务**：
    1.  **Buffer Memory**：简单的列表，保存所有聊天记录。
    2.  **Window Memory**：只保留最近 k 轮对话（防止 Token 爆炸）。
    3.  **Token Calculation**：计算历史记录是否超过模型上下文窗口（如 4k/8k tokens）。
  * **代码焦点**：
      * `src/llm/memory.py`
      * 实现一个简单的 `ConversationBuffer`。
  * **💡 面试必问**：
      * *“如果对话历史太长超过了 Context Window，有哪些处理策略？(截断、摘要 Summary、滑动窗口)”*

### Day 6: 综合项目 - 法律小助手 v0.1 (Integration)

  * **目标**：整合本周所学，开发一个 CLI 工具。
  * **核心任务**：
    1.  用户输入案情描述。
    2.  系统组装 System Prompt + History + User Input。
    3.  LLM 进行分析（CoT），并返回 JSON 格式的“初步判决建议”。
    4.  打印结果并更新历史记录。
  * **交付物**：
      * 运行 `python stage3/week8/app.py` 即可交互。

### Day 7: 复盘与模拟面试 (Review)

  * **目标**：从“写代码”上升到“讲原理”。
  * **复习重点**：
      * Tokenization 原理 (BPE / Byte-level)。
      * Transformer 的 Decoder-only 架构（GPT 系列）。
      * 常见的 Prompt 攻击（Prompt Injection）。
  * **自我介绍优化**：
      * 将“熟悉 OpenAI API”改为“熟悉 LLM 开发范式，能够设计高鲁棒性的 Prompt 系统并处理结构化输出”。

