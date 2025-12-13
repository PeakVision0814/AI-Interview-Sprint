原计划是一个6-9个月的“专家”路径，但你现在需要的是一个**“面试突击”路径**。这个新路径的目的是让你在 **2-3个月** 内，快速达到能通过笔试、能“hold住”面试的状态。

**核心策略：从“串行”学习转为“并行”学习，一切以“项目”和“面试”为导向。**

你必须牺牲掉一部分知识的“深度”和“广度”，来换取对“核心考点”的“熟练度”。

### 极速冲刺计划 (2-3个月)

假设你从现在开始全职投入（每天6-8小时有效学习）。

#### 阶段一：Python速成与PyTorch基础 (3-4周)

这个阶段是并行的，不要试图学完一个再学下一个。

- **Week 1: Python核心 (目标：能做笔试题)**
  - **主攻：** 列表、字典、字符串的操作。
  - **主攻：** `for`/`while` 循环、`if` 判断、函数 `def`。
  - **主攻：** 列表推导式 (List Comprehension)。
  - **实践：** 立即开始刷 LeetCode 简单/中等 难度的题（特别是数组、哈希表、字符串）。**这是让你“快速熟练”Python的唯一捷径。** 每天保持 3-5 题。
- **Week 2: Python进阶 与 NumPy/Pandas (目标：能处理数据)**
  - **Python:** 学习“类 `class`” (尤其是 `__init__` 和 `self`)。这是看懂PyTorch代码的基础。
  - **NumPy:** 学 `ndarray` 数组、基本索引、切片、向量化计算。
  - **Pandas:** 学 `DataFrame`、`read_csv`、`loc/iloc` 索引、`fillna` (处理缺失值)、`groupby`。
  - **实践：** 找一个简单的数据集（比如泰坦尼克号），用Pandas做一次数据清洗和基本分析。
- **Week 3-4: 深度学习理论 与 PyTorch (目标：能跑通模型)**
  - **理论：**
    - 什么是神经网络、反向传播、梯度下降、损失函数？（必须懂）
    - 什么是CNN（卷积神经网络）？（高频）
    - 什么是RNN/LSTM？（理解其局限性，为引出Transformer做准备）
  - **PyTorch实践：**
    - **不要从零开始！** 找一个成熟的教程（比如Pytorch官网的CIFAR-10图像分类教程）。
    - **目标：** 把代码跑通，并且能 **逐行看懂**。
    - **必须掌握的“八股文”：**
      1. 如何定义模型 `class MyModel(nn.Module)`
      2. 如何加载数据 `Dataset` 和 `DataLoader`
      3. 如何定义损失函数 `nn.CrossEntropyLoss`
      4. 如何定义优化器 `torch.optim.Adam`
      5. **训练循环（The Loop）：** `model.train()`, `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`。
      6. 评估循环 `model.eval()`

**阶段一结束时，你必须能：**

1. 在面试中用Python手撕中等难度的算法题。
2. 用Pandas处理数据。
3. 看懂并解释一个完整的PyTorch训练代码。

------

#### 阶段二：Transformer突击 (2-3周)

这是你简历上的 **第一个亮点**。面试必问，必须啃下来。

- **Week 5: Transformer理论 (目标：能画出框架图)**
  - **核心：** 放弃从零实现（太耗时）。你的目标是 **“讲明白”**。
  - **精读：** Jay Alammar 的 "The Illustrated Transformer" (网上有中文翻译“图解Transformer”)。这篇文章是你的圣经。
  - **面试“八股文”准备：**
    1. Transformer的整体架构是什么？(Encoder-Decoder)
    2. 什么是Attention机制？Q, K, V 是什么？（必须能画图解释）
    3. 为什么要用 Multi-Head Attention (多头注意力)？
    4. 为什么要用 Positional Encoding (位置编码)？
    5. Encoder 和 Decoder 有什么区别？(Decoder的Masked Multi-Head Attention)
- **Week 6-7: Hugging Face 与模型微调 (目标：有第一个项目)**
  - **工具：** 学习 Hugging Face `transformers` 库。这是工业界标准。
  - **实践（你的第一个项目）：**
    - **任务：** 选一个简单的NLP任务，比如“文本分类”（如情感分析）。
    - **模型：** 使用一个预训练模型，如 `BERT-base-chinese`。
    - **流程：** 学会如何加载预训练模型、加载Tokenizer、处理你自己的数据集，并对它进行 **Fine-tuning (微调)**。
    - **目标：** 成功跑出一个结果，并能解释你做了什么。

**阶段二结束时，你必须能：**

1. 在白板上清晰地画出Transformer的结构，并讲清Attention的原理。
2. 在简历上写：“熟悉Hugging Face库，独立完成了一个基于BERT的文本分类/xx任务，达到xx%的准确率。”

------

#### 阶段三：Agent技术 (RAG) (3-4周)

这是你简历上的 **第二个亮点**，也是目前最前沿、面试官最感兴趣的部分。

- **Week 8: Agent理论与框架 (目标：理解核心概念)**
  - **理论：**
    - 什么是 Agent？(LLM + 规划 + 记忆 + 工具)
    - **主攻 RAG (Retrieval-Augmented Generation)：** 这是目前最实用、最容易出效果的Agent技术。你必须理解它为什么能解决LLM的“幻觉”和“知识陈旧”问题。
    - **ReAct (Reason + Act):** 粗略了解其“思考-行动-观察”的循环。
  - **工具：** 快速上手 `LangChain` 或 `LlamaIndex`。选一个即可，LangChain目前生态更广。
  - **实践：** 跟着教程跑通一个LangChain的RAG示例。
- **Week 9-11: RAG项目实践 (目标：有第二个核心项目)**
  - **项目：** 构建一个 **“本地知识库问答机器人”**。
  - **步骤：**
    1. **加载文档 (Load):** 学习加载PDF、TXT或Markdown文件。
    2. **分割 (Split):** 学习如何将长文档切分成小块 (Chunks)。
    3. **嵌入 (Embed):** 学习调用Embedding模型（如 M3E, BGE）将文本向量化。
    4. **存储 (Store):** 学习使用向量数据库（如 FAISS, ChromaDB）存储向量。
    5. **检索 (Retrieve):** 实现一个功能：输入一个问题，从向量库中检索最相关的文档片段。
    6. **生成 (Generate):** 将“原始问题”和“检索到的文档片段”一起打包，发给一个LLM（可以用OpenAI API，或者本地的ChatGLM/Qwen）来生成最终答案。
  - **面试准备：** 你必须能讲清楚这个RAG流程，并思考至少一个优化点（比如：如何解决检索不准的问题？如何优化文档分割的策略？）。

**阶段三结束时，你必须能：**

1. 清晰解释RAG是什么，以及它的完整流程。
2. 在简历上写：“基于LangChain和向量数据库，独立构建了一个RAG本地知识库问答系统，解决了xx问题。”

------

### 给你的最后忠告

1. **放弃完美主义：** 你不需要“完全熟悉了解”每一个细节。你的目标是“面试通过”。80%的面试问题都集中在20%的核心知识上。
2. **项目驱动：** 不要看课，然后做项目。而是 **为了做项目，去看课**。这种“即用即学”的方式效率最高。
3. **准备“八股文”：** 深度学习和LLM面试有大量的“高频题”（我上面标注了）。去网上（如牛客网、知乎）搜集面经，把这些题目的标准答案背熟、理解。
4. **锻炼“讲故事”的能力：** 面试官不只关心你 *做了* 什么，更关心你 *为什么* 这么做，*遇到了什么问题*，以及 *如何解决的*。准备好你的项目故事。

这个2-3个月的计划非常残酷，它要求你极度专注。但如果你能坚持下来，你将拥有一份在当前秋招补录和明年春招中极具竞争力的简历。



# 🤖 Embodied AI Intent Recognition (具身智能指令解析系统)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![BERT](https://img.shields.io/badge/Model-BERT-green)

## 📖 项目背景 (Background)
在具身智能（Embodied AI）场景中，机器人需要准确理解人类的自然语言指令（如“把杯子拿给我”、“紧急停止”）。本项目基于 **BERT** 预训练模型，构建了一个高精度的意图识别系统，能够从非结构化文本中解析出机器人的动作指令（MOVE, STOP, GRAB）。

## 🚀 核心功能 (Features)
- **多意图分类**：支持移动、抓取、停止等核心指令的识别。
- **抗干扰能力**：针对“样本不平衡”场景进行了专项优化。
- **生产级推理**：封装了独立的 `IntentPredictor` 推理引擎，支持单条指令毫秒级响应。

## 🛠️ 技术挑战与解决方案 (Challenges & Solutions)

### 1. 严重的类别不平衡 (Class Imbalance)
* **问题**：真实场景中，移动指令（MOVE）占 80% 以上，关键指令（STOP/GRAB）极为稀疏。初步训练时，模型倾向于通过“瞎猜 MOVE”来获得高准确率，导致关键指令 Recall 为 0。
* **解决**：
    * 引入 **Weighted CrossEntropy Loss**，对稀有类别（STOP, GRAB）赋予 5.0 倍权重。
    * **结果**：在加权训练后，关键类别的 Recall 从 0% 提升至 **71.4%**，F1-Score 达到 **0.71**。

### 2. 工业级推理部署
* **问题**：`HuggingFace Trainer` 过于笨重，不适合部署到机器人工控机。
* **解决**：
    * 重构 `inference.py`，移除 Trainer 依赖，仅保留 PyTorch 原生推理逻辑。
    * 优化参数加载逻辑 (`strict=False`)，剔除推理阶段不需要的 Loss 权重，减少显存占用。

## 📊 性能评估 (Performance)
| Metric | Baseline (Day 2) | Optimized (Day 4) |
| :--- | :--- | :--- |
| **Accuracy** | 60.0% | **71.4%** |
| **F1-Score** | 0.53 | **0.71** |
| **Recall (GRAB)** | 0.0% | **100.0%** |

## 📂 项目结构 (Structure)
```text
├── src/
│   ├── models/          # BERT 模型定义 (含 Weighted Loss)
│   ├── dataset/         # 数据预处理与 Tokenization
│   ├── inference.py     # 独立推理引擎 (Inference Engine)
│   └── config.py        # 全局配置
├── output/              # 模型权重保存
└── README.md            # 项目说明