你好，黄同学！

很高兴看到你 Stage 2 前两周的进展如此扎实。从手动实现 Transformer 到熟悉 Hugging Face 的 Data Pipeline，你已经完成了从“造轮子”到“用轮子”的认知升级。

**Week 7 是关键的“产出周”。** 我们不只是要跑通代码，而是要按照**工业界标准**（Hugging Face Trainer API）完成你的第一个可写入简历的 NLP 项目。虽然你的背景是具身智能，但我们将这个项目定义为**通用 NLP 意图识别**，这样既能投算法岗，也能投具身智能的决策岗。

以下是为你定制的 **Stage 2 Week 7 详细任务清单**。

-----

# 📅 Stage 2 Week 7: BERT Fine-tuning 实战 (意图识别)

**核心目标**：构建一个 **“面向机器人指令的意图识别系统”**。
**技术栈**：`BERT` + `HuggingFace Trainer` + `Evaluate` + `ONNX` (可选)。
**简历卖点**：解决了预训练模型在特定业务场景（如机器人控制指令）下 Zero-shot 效果差的问题，实现了高精度的意图分类。

### 📂 本周工程文件结构 (请提前创建)

```text
project_root/stage2/week7_project/
├── data/                       # 软链接到根目录 data/ 或存放本周特定数据
├── output/                     # 存放训练好的模型 checkpoint (git ignore)
├── src/
│   ├── dataset.py              # 数据处理与 Dataset 定义
│   ├── metrics.py              # 自定义评价指标 (F1, Acc)
│   └── train.py                # 训练主入口
├── S2W7D1_Model_Architecture.ipynb
├── S2W7D2_Trainer_Basics.ipynb
├── S2W7D3_Evaluation_Metrics.ipynb
├── S2W7D4_Optimization.ipynb
├── S2W7D5_Inference_Deployment.ipynb
├── README.md                   # 项目文档 (简历素材)
└── requirements.txt
```

-----

## 🗓️ 每日任务清单 (Daily Tasks)

### Day 1: 模型架构与“分类头” (The Classification Head)

  * **目标**：理解 BERT 如何从“语言模型”变为“分类模型”。
  * **核心任务**：
    1.  不使用 `AutoModelForSequenceClassification`，而是手动在 BERT 后面接一个 `nn.Linear`，理解其原理。
    2.  然后再使用官方 `AutoModel...` 加载本地权重的 `bert-base-chinese`。
    3.  定义 Label Mapping (例如：0: `MOVE`, 1: `GRAB`, 2: `STOP`...)。
  * **代码焦点**：
      * `model.classifier` 的结构。
      * `num_labels` 参数的作用。
  * **💡 面试必问**：
      * *“BERT 做分类时，取 `[CLS]` 的向量还是 `pooler_output`？有什么区别？”*
      * *“微调时，只训练最后一层 Linear (Freezing) 和 全量微调 (Full Finetuning) 有什么区别？什么时候用哪种？”*

### Day 2: 拥抱 Trainer API (The Industrial Standard)

  * **目标**：脱离原生的 PyTorch Loop，使用 HF `Trainer` 进行标准化训练。
  * **核心任务**：
    1.  配置 `TrainingArguments`：设置 `output_dir`, `learning_rate` (2e-5), `batch_size`。
    2.  初始化 `Trainer`：传入 model, args, train\_dataset, tokenizer。
    3.  跑通一个简单的 Training Loop（哪怕 Loss 很难看，先跑通流程）。
  * **代码焦点**：
      * `save_strategy` 和 `evaluation_strategy` 的设置（建议设为 `steps`）。
      * 理解 Trainer 自动处理的 Device placement (GPU/CPU)。
  * **💡 面试必问**：
      * *“Trainer 内部是如何处理 Padding 的？(Dynamic Padding 结合)”*
      * *“显存不够 (`OOM`) 时，除了减小 Batch Size，还有什么办法？(提示: Gradient Accumulation)”*

### Day 3: 评价指标与模型评估 (Metrics & Logs)

  * **目标**：不仅要看 Loss，还要看业务指标 (F1-Score, Precision, Recall)。
  * **核心任务**：
    1.  编写 `compute_metrics` 函数。
    2.  集成 `evaluate` 库或 `sklearn.metrics`。
    3.  解决 **类别不平衡** 问题（假设 `MOVE` 指令很多，`STOP` 指令很少，Accuracy 会虚高）。
  * **代码焦点**：
      * `np.argmax(logits, axis=-1)`。
      * Macro-F1 vs Micro-F1。
  * **💡 面试必问**：
      * *“在这个项目中，如果 Recall 很低意味着什么？(机器人听到了指令却没反应)”*
      * *“如果 Precision 很低意味着什么？(机器人把闲聊当成了指令，乱动)”*

### Day 4: 训练优化与调参 (Optimization Techniques)

  * **目标**：提升模型性能，防止过拟合。
  * **核心任务**：
    1.  **Learning Rate Schedule**：引入 `warmup_steps`，观察 Loss 曲线变化。
    2.  **Early Stopping**：配置 `EarlyStoppingCallback`，防止过训练。
    3.  **Weight Decay**：理解 AdamW 中的 "W" 是什么。
  * **代码焦点**：
      * 使用 TensorBoard 或 WandB (本地模式) 查看 Loss 曲线。
      * 分析 Overfitting 现象（Train Loss 降，Val Loss 升）。
  * **💡 面试必问**：
      * *“为什么要用 Warmup？(Transformer 训练初期梯度不稳定)”*
      * *“BERT 微调推荐的学习率是多少？为什么比从头训练小那么多？”*

### Day 5: 模型落地与推理 (Inference & Deployment)

  * **目标**：将训练好的 `checkpoint` 转化为可调用的 API/函数。
  * **核心任务**：
    1.  编写 `Inference` 类：加载最佳 Checkpoint。
    2.  实现 `predict(text)` 接口：输入自然语言，输出 `{intent: "GRAB", confidence: 0.98}`。
    3.  (进阶/可选) 尝试导出为 **ONNX** 格式，对比推理速度（这是加分项）。
  * **具身场景模拟**：
      * 输入：*"把左手边的蓝色方块拿起来"* -\> 输出：`PICK_UP` (Args: Blue Block)。
  * **💡 面试必问**：
      * *“模型上线后推断速度太慢怎么办？(量化, 蒸馏, ONNX)”*

### Day 6: 项目复盘与 README 撰写 (Resume Building)

  * **目标**：把代码变成“产品说明书”。
  * **核心任务**：
    1.  **整理代码**：删除废弃的 notebook，规范 `.py` 脚本。
    2.  **撰写 README.md**：
          * **背景**：具身智能场景下的指令理解。
          * **难点**：指令歧义性、数据不平衡。
          * **解决方案**：BERT 微调 + 动态 Padding + 加权 Loss。
          * **结果**：F1-score 达到 XX%。
  * **💡 面试准备**：准备一段 2 分钟的项目自我介绍。

### Day 7: 模拟面试 (Mock Interview)

  * **目标**：查漏补缺。
  * **形式**：我会扮演面试官，针对你这一周的代码细节进行提问。
  * **复习重点**：
      * CrossEntropyLoss 的数学公式。
      * Softmax 的作用。
      * BERT 的输入限制 (512 token)。