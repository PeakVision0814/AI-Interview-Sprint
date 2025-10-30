import numpy as np

# 一维数组
print("--- 1. 从 Python 列表创建 ndarray ---")
my_list = [1, 2, 3, 4, 5]
arr_1d = np.array(my_list)
print(f"一维数组：{arr_1d}")

# 二维数组
my_nested_list = [[1, 2, 3],[4, 5, 6]]
arr_2d = np.array(my_nested_list)
print(f"二维数组：\n{arr_2d}")

print("\n --- 2. 检查数组的核心属性 ---")

# 查看形状
print(f"一维数组 {arr_1d} 的形状: {arr_1d.shape}")
print(f"二维数组 {arr_2d} 的形状：{arr_2d.shape}")

# 查看数据类型（dtype）
print(f"数组 {arr_1d} 的数据类型: {arr_1d.dtype}")

print("\n --- 3. 使用 NumPy 内置函数创建常用数组 ---")

# 创建一个 3x4 的全零矩阵
zeros_arr = np.zeros((3, 4))
print(f"全零矩阵：\n{zeros_arr}")

# 创建一个 2x5 的全一矩阵
one_arr = np.ones((2, 5))
print(f"全一矩阵：\n{one_arr}")

# 创建一个从 0 到 9 的一维数组
range_arr = np.arange(10)
print(f"范围数组：{range_arr}")

print("\n --- 4. 索引、切片与向量化 ---")
print("\n --- 索引与切片 ---")
# 索引与切片
arr = np.arange(10) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(f"第五个元素（索引为4）：{arr[4]}")

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# 获取单个元素：[行，列]
print(f"获取第2行第3列元素（索引1，2）：{matrix[1, 2]}")

# 获取一整行
print(f"获取第 2 整行（索引1）：{matrix[1]}")

# 获取一整列
print(f"获取第 3 整列（索引2）：{matrix[:, 2]}")

# 向量化操作
data = np.arange(5) # [0, 1, 2, 3, 4]

# 告别 for 循环
print(f"\n ---向量化操作 ---")
print(f"原数组:{data}")
print(f"所有元素乘以 2 :{data * 2}")
print(f"所有元素加上 10 :{data + 10}")

# 数组间的运算
data2 = np.ones(5)
print(f"另一个数组为:{data2}")
print(f"两个数组相加得到: {data + data2}")

# 通用函数 (Universal Functions, ufuncs)
print(f"对所有元素求正弦: {np.sin(data)}")

# 今日任务
print("\n--- 任务一：创建随机数组 ---")
arr_random = np.random.rand(5, 5)
print(f"创建一个 5x5 的随机数组:\n{arr_random}")
# 所有大于 0.5 的数变为 1，所有小于等于 0.5 的数变为 0。

print(f"\n--- 任务二：向量化条件赋值 ---")
arr_random[arr_random > 0.5] = 1
arr_random[arr_random <= 0.5] = 0
print(f"赋值之后的数组：\n{arr_random}")

print("\n--- 任务三：提取子集 ---")
print(f"提取第二整行: {arr_random[1]}")
print(f"提取第三整列: {arr_random[:, 2]}")
