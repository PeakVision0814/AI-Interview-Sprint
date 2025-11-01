这是你 W1D1 到 W1D7 的详细学习路线图。

------

### 第一周 (Week 1) 详细学习计划：Python核心 与 数据处理

**本周总目标：**

1. **Python：** 熟练掌握核心语法，能独立完成 LeetCode 简单和中等级别的算法题（特别是数组、哈希表、字符串）。
2. **Pandas：** 能够加载、清洗、并聚合分析一个标准的数据集（如泰坦尼克号）。

------

**W1D1 (今天): Python 基础语法 & 你的第一个算法**

- **今日目标：** 搭建环境，掌握最基本的数据类型、变量和逻辑控制。
- **学习内容：**
  1. **环境：** 快速安装 Python 和 VS Code (或直接使用 Google Colab，更推荐后者，零配置)。
  2. **变量与数据类型：** `int` (整数), `float` (浮点数), `str` (字符串)。
  3. **字符串操作 (高频)：**
     - f-string (格式化): `name = "AI"; print(f"Hello, {name}")`
     - `.split(delimiter)`: `"a,b,c".split(',')` -> `['a', 'b', 'c']`
     - `.join(list)`: `"-".join(['a', 'b', 'c'])` -> `"a-b-c"`
  4. **数据结构 (入门)：** `list` (列表)。
     - 创建: `my_list = [1, "hello", 3.0]`
     - 索引: `my_list[0]` (获取第一个)
     - 切片: `my_list[1:3]` (获取第1到第2个)
  5. **控制流：** `if-elif-else` 条件判断。
- **今日实践 (必须完成)：**
  1. 写一个脚本：判断一个数字是正数、负数还是零。
  2. **LeetCode 9. 回文数 (Palindrome Number)**：
     - **目的：** 练习基本的数学运算或字符串操作与逻辑判断。
     - **提示：** 你可以转成字符串来做（练习字符串切片 `s[::-1]`），也可以用数学方法做。

------

**W1D2: Python 循环 & 核心数据结构 (字典)**

- **今日目标：** 掌握循环和哈希表（字典）这一最强大的数据结构。
- **学习内容：**
  1. **循环 (Loops)：**
     - `for` 循环: `for i in range(5):` (循环5次), `for item in my_list:` (遍历列表)。
     - `while` 循环 (了解即可)。
  2. **数据结构 (核心)：** `dict` (字典)。
     - 创建: `my_dict = {"name": "Gemini", "age": 1}`
     - 访问: `my_dict["name"]`
     - 添加/修改: `my_dict["age"] = 2`
     - 检查键是否存在: `if "name" in my_dict:` (面试高频)
- **今日实践 (必须完成)：**
  1. 写一个脚本：遍历一个列表 `[1, 2, 3, 4, 5]`，打印出所有元素的平方。
  2. **LeetCode 1. 两数之和 (Two Sum)**：
     - **目的：** 这是哈希表（`dict`）的“Hello World”题。**必须**用 `dict` 优化到 O(n) 复杂度。
     - **思路：** 遍历列表，对于每个数 `x`，检查 `target - x` 是否在你的字典里。如果不在，就把 `x` 和它的索引存入字典。

------

**W1D3: Python 函数 & “Pythonic”写法**

- **今日目标：** 学会封装代码，并掌握列表推导式。
- **学习内容：**
  1. **函数 (Function)：**
     - 定义: `def my_function(param1, param2):`
     - 返回值: `return value`
  2. **列表方法 (List Methods)：**
     - `.append(item)`: 添加到末尾
     - `.pop()`: 弹出末尾元素 (常用于模拟“栈”)
  3. **列表推导式 (List Comprehension)：**
     - **核心！** 极大提高代码效率和简洁度。
     - 示例: `squares = [x*x for x in range(10)]`
     - 带条件: `evens = [x for x in range(10) if x % 2 == 0]`
- **今日实践 (必须完成)：**
  1. 把你前两天写的 LeetCode 题解，全部用 `def` 函数封装起来。
  2. 用列表推导式：给定一个列表 `[1, -2, 3, -4, 5]`，生成一个新列表，只包含其中正数的平方 `[1, 9, 25]`。
  3. **LeetCode 20. 有效的括号 (Valid Parentheses)**：
     - **目的：** 练习 `list` 作为“栈”的使用（`.append()` 入栈, `.pop()` 出栈），以及 `dict` 的查询。
     - **思路：** 遇到左括号就入栈，遇到右括号就出栈一个左括号，看是否匹配。

------

**W1D4: Python 面向对象 (OOP) 速成**

- **今日目标：** **不求精通，只求看懂** PyTorch 的模型代码。
- **学习内容：**
  1. **概念：** `class` (类) 是蓝图，`object` (对象) 是实例。
  2. **构造函数：** `__init__(self, ...)`
     - `self` 指代实例本身。
     - `self.variable = ...` (实例属性)。
  3. **方法 (Method)：** 类里面定义的函数，第一个参数必须是 `self`。
  4. **继承 (Inheritance)：**
     - `class ChildClass(ParentClass):`
     - `super().__init__(...)` (调用父类的构造函数)。
- **今日实践 (必须完成)：**
  1. 写一个 `Person` 类：
     - `__init__` 接收 `name` 和 `age` 两个参数。
     - 有一个 `greet()` 方法，打印 `f"Hi, I am {self.name}, {self.age} years old."`
  2. **预习：** 去网上搜一个 PyTorch 定义模型的代码，例如 `class MyModel(nn.Module):`。你不需要懂 `nn.Module` 是什么，你只需要：
     - 认出 `class ... (nn.Module):` 这是“继承”。
     - 认出 `def __init__(self):` 这是在定义“层”（积木）。
     - 认出 `def forward(self, x):` 这是在定义“前向传播”（数据流）。
     - **看懂这个结构，你今天的目标就达成了。**

------

**W1D5: NumPy 核心 (PyTorch 的前身)**

- **今日目标：** 告别 `for` 循环，学会“向量化”思考。
- **学习内容：**
  1. `import numpy as np`
  2. `np.array(my_list)`: 从 Python 列表创建 `ndarray` (N维数组)。
  3. **核心属性：** `.shape` (形状), `.dtype` (数据类型)。
  4. **创建数组：** `np.zeros((3, 4))`, `np.ones(...)`, `np.arange(...)`。
  5. **索引与切片：**
     - 一维：`arr[5]`
     - 二维：`arr[1, 3]` (第1行第3列) 或 `arr[1]` (第1整行)。
  6. **向量化操作 (核心！)：**
     - `arr * 2`, `arr + 10`, `arr1 + arr2`, `np.sin(arr)`
     - **关键：** 这些操作是对数组中的 *每个元素* 同时进行的，比 Python `for` 循环快几个数量级。
- **今日实践 (必须完成)：**
  1. 创建一个 5x5 的随机数数组（`np.random.rand(5, 5)`）。
  2. **不使用 `for` 循环**，将数组中所有大于 0.5 的数变为 1，小于等于 0.5 的数变为 0。 (提示：`arr[arr > 0.5] = 1`)
  3. 提取这个 5x5 数组的第 2 行 和 第 3 列。

------

**W1D6: Pandas 核心 (上) - 数据读取与选择**

- **今日目标：** 掌握 `DataFrame`，学会“读”和“选”数据。
- **学习内容：**
  1. `import pandas as pd`
  2. **两大结构：** `Series` (1D, 带索引的列) 和 `DataFrame` (2D, 带索引的表格)。
  3. **读取数据：** `df = pd.read_csv("your_file.csv")` (这是90%的场景)。
  4. **检查数据 (数据探索)：**
     - `df.head()` (看前5行)
     - `df.info()` (看每列类型和缺失值)
     - `df.describe()` (看数值列的统计)
  5. **数据选取 (最重要！)**
     - 选列：`df['ColumnName']` (返回 Series), `df[['Col1', 'Col2']]` (返回 DataFrame)。
     - **按标签选：`df.loc[row_label, col_label]`**
     - **按位置选：`df.iloc[row_index, col_index]`**
     - **条件筛选 (高频)：`df[df['Age'] > 20]`**
- **今日实践 (必须完成)：**
  1. **[项目启动]** 去 Kaggle 搜索 "Titanic" (泰坦尼克号) 数据集，下载 `train.csv`。
  2. 用 Pandas 加载它。
  3. 用 `head()`, `info()` 检查数据。
  4. **筛选：** 选出所有 1 等舱 (`Pclass == 1`) 的乘客。
  5. **筛选：** 选出所有女性 (`Sex == 'female'`) 且年龄 (`Age`) 小于 18 岁的乘客。

------

**W1D7: Pandas 核心 (下) - 数据清洗与聚合**

- **今日目标：** 掌握数据清洗和 `groupby`，完成本周的迷你项目。
- **学习内容：**
  1. **处理缺失值 (NaN)：**
     - 检查：`df.isnull().sum()` (看每列有多少缺失)。
     - 填充：`df['Age'] = df['Age'].fillna(df['Age'].mean())` (用平均年龄填充缺失年龄)。
     - 删除：`df.dropna()` (删除有缺失的行，慎用)。
  2. **分组聚合 (Group By) (核心！)**
     - **Split-Apply-Combine** 思想。
     - `df.groupby('ColumnName')` (按某列分组)。
     - `.mean()`, `.sum()`, `.count()` (应用聚合)。
     - 示例：`df.groupby('Sex')['Survived'].mean()` (按性别分组，计算存活率均值)。
- **今日实践 (必须完成)：**
  1. **[迷你项目]** 继续你的泰坦尼克号数据：
     - 用“年龄的平均值”填充 `Age` 列的缺失值。
     - 计算“不同性别” (`Sex`) 的平均存活率 (`Survived`)。
     - 计算“不同船舱等级” (`Pclass`) 的平均存活率。
     - (进阶) 计算“不同性别”和“不同船舱等级”**组合**的平均存活率。 (提示: `df.groupby(['Sex', 'Pclass'])`)
  2. **复习：**
     - **LeetCode 217. 存在重复元素 (Contains Duplicate)**
     - **目的：** 复习 `dict` 或 `set` 的使用。
     - **挑战：** 尝试用一行代码解决 (提示：`len(set(nums))` )。

------

本周总结

如果你完成了所有实践，你现在应该：

- 对 Python 核心语法（循环、函数、字典、列表）非常自信。
- 能在 LeCode 上用 `dict` 和 `list` 解决中等难度的题。
- 能用 Pandas 独立完成一个简单数据集的清洗和分析。