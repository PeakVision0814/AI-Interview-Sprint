# --- 1. for 循环 ---

# 用法一:遍历一个列表
print("--- 遍历列表 ---")
students = ["张三", "李四", "王五"]
for student in students:
    print(f"你好, {student}同学")

# 用法二：重复固定的次数 (使用 range() 函数)
# range(5) 会生成一个从 0 到 4 的数字序列 [0, 1, 2, 3, 4]
print("\n--- 重复固定次数 ---")
for i in range(5):
    print(f"这是第 {i+1} 次循环")

# --- 2. 字典(dict) ---
# 创建一个字典
user_profile = {
    "name": "黄高朋",
    "age": 25,
    "skills": ["Python", "Pandas"],
    "is_leader": True
}
print("\n--- 字典操作 ---")
print(f"原始字典: {user_profile}")

# 访问(ACCESS) - 通过键获取值
print(f"姓名: {user_profile['name']}")

# 添加/修改 (Add / Modify) - 如果键不存在就添加, 存在就覆盖
user_profile['city'] = '杭州'
user_profile['age'] = 26
print(f"原始字典: {user_profile}")

# 检查键是否存在 (Check Existence) - 面试高频用法！

key = input("请输入一个整数，然后按回车: ")

if key in user_profile:
    print(f"字典中包含 {key} 这个值")
else:
    print(f"字典中不包含 {key} 这个值")

nums = [1, 2, 3, 4, 5]
for num in nums:
    numsqre = num * num
    print(numsqre)