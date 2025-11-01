# 第二周 (Week 2) 详细学习计划：深度学习理论 与 PyTorch 基础

**本周总目标：**

1.  **理论：** 能用自己的话向面试官解释清楚神经网络、梯度下降、反向传播的核心思想。
2.  **代码：** 能在白板/IDE上，不参考任何资料，默写出 PyTorch 的标准训练循环代码，并理解每一行的作用。

-----

## **W2D1 (Day 8): 深度学习核心理论 (上) - “神经网络是什么”**

  * **今日目标：** 建立神经网络的宏观认知，理解它的基本组成和“黑话”。
  * **学习内容：**
    1.  **神经网络 (Neural Network) 的基本结构：**
          * 理解**输入层 (Input Layer)**、**隐藏层 (Hidden Layer)**、**输出层 (Output Layer)** 的概念。
          * 什么是**神经元 (Neuron)**？它在做什么？（核心：线性变换 `wx+b` + 非线性激活）。
    2.  **激活函数 (Activation Function)：**
          * **为什么需要它？** （回答：为模型引入非线性，否则多层网络等价于单层）。
          * 重点掌握 **ReLU** (`max(0, x)`)：它是什么样的，有什么优点（计算简单、防梯度消失）。
    3.  **模型如何输出预测？**
          * **回归任务：** 输出层通常是一个线性神经元。
          * **分类任务：** 输出层神经元数量等于类别数，通常会经过一个 **Softmax** 函数，将输出转换为概率。
  * **今日实践 (必须完成)：**
    1.  **观看视频：** 观看 **3Blue1Brown 关于神经网络的系列视频**（B站有官方中文翻译），重点看前两集。**不要陷入数学细节**，目标是建立直观理解。
    2.  **手绘网络：** 在纸上画一个包含 2 个输入、1 个 3 神经元的隐藏层、1 个输出的简单神经网络结构图，并标出各层名称。
    3.  **思考题：** 如果没有激活函数，一个 5 层的神经网络和一个 1 层的神经网络在表达能力上有什么区别？（答案：没有区别，都是线性的）。

-----

## **W2D2 (Day 9): 深度学习核心理论 (下) - “神经网络如何学习”**

  * **今日目标：** 理解模型训练的闭环：损失、优化、反向传播。
  * **学习内容：**
    1.  **损失函数 (Loss Function)：**
          * 它的作用是什么？（回答：衡量模型“预测值”与“真实值”之间的差距）。
          * 分类任务常用：**交叉熵损失 (Cross-Entropy Loss)**。
          * 回归任务常用：**均方误差损失 (MSE Loss)**。
    2.  **梯度下降 (Gradient Descent)：**
          * **核心思想：** 沿着损失函数梯度（导数）下降最快的方向，一小步一小步地调整模型参数（权重 `w` 和偏置 `b`），从而让损失越来越小。
          * **学习率 (Learning Rate)：** 就是“一小步”的步长。它太大或太小会发生什么？（太大：震荡不收敛；太小：训练太慢）。
    3.  **反向传播 (Backpropagation)：**
          * **暂时你只需要理解：** 它是一个高效计算出网络中所有参数梯度的“算法”，是实现梯度下降的前提。它利用链式法则，从后往前（从损失函数到输入层）逐层计算梯度。
    4.  **核心概念区分：**
          * **训练集 (Training Set):** 用来计算梯度、更新模型参数的数据。
          * **验证集 (Validation Set):** 不参与参数更新，只用来评估模型、调整超参数（如学习率）的数据。
          * **测试集 (Test Set):** 完全不用，只在最后用来报告模型最终性能的数据。
  * **今日实践 (必须完成)：**
    1.  **观看视频：** 继续观看 3Blue1Brown 关于梯度下降和反向传播的视频。
    2.  **打比方：** 尝试用“下山”的比喻来向自己或朋友解释什么是梯度下降（你在山上的位置是当前参数，山的陡峭程度是梯度，你往下走一步的步长是学习率，山底是损失最小的地方）。
    3.  **面试题模拟：** 尝试回答：“请简单说一下神经网络是怎么训练的？” （参考回答：首先，模型进行一次前向传播得到预测值；然后，通过损失函数计算预测值和真实值的差距；接着，通过反向传播计算损失函数对每个参数的梯度；最后，优化器根据梯度和学习率更新模型的参数，不断重复这个过程）。

-----

## **W2D3 (Day 10): PyTorch 核心：`Tensor` 与 `Autograd`**

  * **今日目标：** 掌握 PyTorch 的基本数据单元 `Tensor`，并亲自实践自动求导机制。
  * **学习内容：**
    1.  **`torch.Tensor` (张量)：**
          * 理解它就是 PyTorch 世界里的 `numpy.ndarray`。所有操作、索引、切片都和 NumPy 极其相似。
          * 创建张量：`torch.tensor()`, `torch.randn()`, `torch.zeros()`。
          * 数据类型：`dtype=torch.float32`, `torch.long`。
    2.  **CPU 与 GPU：**
          * `.to('cuda')` 或 `.cuda()`：将张量及其计算转移到 GPU 上，实现加速。
          * `.to('cpu')` 或 `.cpu()`：将张量转回 CPU。
    3.  **`Autograd` (自动求导系统)：**
          * **`requires_grad=True`：** 这是一个“开关”，告诉 PyTorch：“请开始追踪这个张量之后的所有计算，以便后续计算它的梯度”。模型参数默认开启。
          * **`loss.backward()`：** 这是“命令”，触发反向传播，PyTorch 会自动计算计算图中所有 `requires_grad=True` 的张量的梯度。
          * **`.grad` 属性：** 调用 `backward()`后，梯度值会累加到对应张量的 `.grad` 属性上。
  * **今日实践 (必须完成)：**
    1.  **NumPy -\> Tensor 转换：** 创建一个 NumPy 数组，然后将其转换为 PyTorch Tensor。
    2.  **自动求导实验：**
        ```python
        import torch
        
        # 1. 创建一个张量 x，并设置 requires_grad=True
        x = torch.tensor(2.0, requires_grad=True)
        
        # 2. 定义一个函数 y = x^2 + 2x + 1
        y = x**2 + 2*x + 1
        
        # 3. 执行反向传播
        y.backward()
        
        # 4. 打印 x 的梯度 (dy/dx = 2x + 2，当 x=2 时，梯度应为 6)
        print(x.grad)
        ```
    3.  理解为什么需要 `optimizer.zero_grad()`：再次运行上述代码的 `y.backward()` 和 `print(x.grad)`，你会发现梯度变成了 12。这是因为 PyTorch 的梯度是**累加**的。因此在每个训练步开始前，必须清零。

-----

## **W2D4 (Day 11): PyTorch 建模基石：`nn.Module` 与常用层**

  * **今日目标：** 学会如何用“搭积木”的方式定义一个神经网络模型。
  * **学习内容：**
    1.  **`torch.nn.Module`：**
          * 所有神经网络模型的“父类”。你的任何自定义模型都必须**继承**它。
          * 它会自动追踪所有内部定义的、可学习的参数。
    2.  **模型定义的两个核心部分：**
          * `__init__(self)` **(构造函数):** 在这里**声明**你将要用到的“积木”（即各种层，如 `nn.Linear`）。只定义，不调用。
          * `forward(self, x)` **(前向传播函数):** 在这里定义数据 `x` 是如何“流过”你在 `__init__` 中定义的那些层的。**这里是真正调用层的地方。**
    3.  **常用层 (积木)：**
          * `nn.Linear(in_features, out_features)`: 全连接层。
          * `nn.ReLU()`: ReLU 激活层。
          * `nn.Conv2d(...)`: 卷积层 (本周只需知道它是用于图像的层即可)。
          * `nn.MaxPool2d(...)`: 最大池化层 (同上)。
          * `nn.Sequential(...)`: 一个容器，可以按顺序包装多个层。
  * **今日实践 (必须完成)：**
    1.  **定义你的第一个模型：**
        ```python
        import torch
        import torch.nn as nn
        
        # 定义一个简单的多层感知机 (MLP)
        class MyMLP(nn.Module):
            def __init__(self):
                super(MyMLP, self).__init__()
                # 声明两个全连接层
                self.fc1 = nn.Linear(in_features=784, out_features=128)
                self.fc2 = nn.Linear(in_features=128, out_features=10)
                # 声明激活函数
                self.relu = nn.ReLU()
        
            def forward(self, x):
                # 定义数据流
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x
        
        # 实例化模型并打印结构
        model = MyMLP()
        print(model)
        
        # 创建一个假的输入数据并测试前向传播
        dummy_input = torch.randn(64, 784) # 假设 batch_size=64, 输入特征=784
        output = model(dummy_input)
        print(output.shape) # 应该输出 torch.Size([64, 10])
        ```

-----

## **W2D5 (Day 12): PyTorch 数据加载：`Dataset` 与 `DataLoader`**

  * **今日目标：** 学会使用 PyTorch 的标准工具来高效、批量地加载数据。
  * **学习内容：**
    1.  **`torch.utils.data.Dataset`：**
          * 一个抽象类，用来“包装”你的数据集。
          * 任何自定义 `Dataset` 都必须实现两个“魔法方法”：
              * `__len__(self)`: 返回数据集的总长度。
              * `__getitem__(self, idx)`: 根据索引 `idx` 返回一条数据（通常是数据和标签的元组）。
    2.  **`torch.utils.data.DataLoader`：**
          * 一个强大的数据加载器，它接收一个 `Dataset` 对象。
          * **核心功能 (面试常考)：**
              * **`batch_size`:** 每次迭代返回几条数据（批处理）。
              * **`shuffle=True`:** 在每个 epoch 开始时打乱数据顺序（防止模型记住顺序）。
              * **`num_workers`:** 使用多少个子进程来预加载数据（加速数据读取）。
  * **今日实践 (必须完成)：**
    1.  **使用内置数据集：**
        ```python
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader
        
        # 定义数据预处理
        transform = transforms.Compose([
            transforms.ToTensor(), # 转换为 Tensor
            transforms.Normalize((0.5,), (0.5,)) # 标准化
        ])
        
        # 加载 MNIST 训练集
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        
        # 用 DataLoader 包装
        train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
        
        # 遍历一个 batch 的数据
        data_batch, labels_batch = next(iter(train_loader))
        
        # 打印它们的形状
        print(f"Data batch shape: {data_batch.shape}")   # 应为 [64, 1, 28, 28]
        print(f"Labels batch shape: {labels_batch.shape}") # 应为 [64]
        ```

-----

## **W2D6 (Day 13): PyTorch 优化器与训练循环 (The Loop) (上)**

  * **今日目标：** 学习训练循环的前半部分，理解优化器的作用。
  * **学习内容：**
    1.  **损失函数 (Criterion)：**
          * 在 PyTorch 中通常是 `nn` 模块下的一个类，如 `nn.CrossEntropyLoss()` 或 `nn.MSELoss()`。
          * 使用方法：`loss = criterion(outputs, labels)`。
    2.  **优化器 (Optimizer)：**
          * `torch.optim` 模块下的类，如 `torch.optim.Adam` 或 `torch.optim.SGD`。
          * **初始化：** 需要告诉它要优化哪些参数，即 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`。
          * **两个核心方法：**
              * `optimizer.zero_grad()`: 清除之前计算的梯度。
              * `optimizer.step()`: 根据梯度更新参数。
    3.  **训练循环“八股文” (前向传播与计算损失)：**
          * **Step 0:** `model.train()` - 将模型设置为训练模式（会启用 Dropout 等）。
          * **Step 1:** `outputs = model(data)` - 前向传播。
          * **Step 2:** `loss = criterion(outputs, labels)` - 计算损失。
  * **今日实践 (必须完成)：**
    1.  **代码整合：** 将前几天写的 `MyMLP` 模型、`DataLoader`、损失函数和优化器都初始化好。
    2.  **写出前半段循环：** 写一个 `for` 循环遍历 `train_loader`，并在循环体内完成前向传播和计算损失，然后 `print(loss)`。**今天先不写反向传播和更新。**

-----

## **W2D7 (Day 14): 训练循环“八股文”与总结**

  * **今日目标：** **背诵并默写** 完整的训练循环，完成本周知识闭环。
  * **学习内容：**
    1.  **回顾并整合所有步骤。**
    2.  **训练循环“八股文” (完整版)：**
        ```python
        # 0. 准备工作
        # model, criterion, optimizer, train_loader 已定义好
        
        # 开启训练模式
        model.train()
        
        # 遍历数据集
        for data, labels in train_loader:
            # (如果用GPU) data, labels = data.to(device), labels.to(device)
        
            # 1. 前向传播
            outputs = model(data)
        
            # 2. 计算损失
            loss = criterion(outputs, labels)
        
            # 3. 梯度清零 (!!!)
            optimizer.zero_grad()
        
            # 4. 反向传播
            loss.backward()
        
            # 5. 更新参数
            optimizer.step()
        ```
  * **今日实践 (必须完成)：**
    1.  **默写：** 打开一个空白的 `.py` 文件，**不看任何参考**，将上述完整的训练循环默写 5 遍。
    2.  **串联代码：** 将今天默写的循环与昨天的代码整合，形成一个（伪）完整的训练脚本。尝试运行它，虽然它什么有意义的事都没做，但只要不报错，就说明你已经掌握了所有组件的连接方式。
    3.  **自我检查：**
          * 你能解释 `model.train()` 的作用吗？
          * 你能解释为什么 `optimizer.zero_grad()` 必须在 `loss.backward()` 之前吗？
          * 你能说出 `DataLoader` 的三个核心参数吗？
          * 你能说出 `nn.Module` 的两个必须实现/重写的方法吗？(`__init__`, `forward`)

-----

**Week 2 成功标准：**

当你完成本周学习后，你必须能够：

  * **口头表达：** 清晰地解释什么是梯度下降和反向传播。
  * **代码能力：** 在面试官面前的白板上，自信地写出标准的 PyTorch 训练循环，并解释每一行的功能。
  * **代码阅读：** 拿到一个简单的 PyTorch 模型定义和训练脚本，能快速看懂其结构和逻辑。

你已经为下周的 MNIST 项目实战铺平了所有道路。下周，我们将把这些独立的“积木”真正地组装成一个能解决实际问题的项目。做得很好，继续保持！