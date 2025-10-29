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