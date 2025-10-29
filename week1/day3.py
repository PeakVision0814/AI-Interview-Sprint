# ---1. 函数（Function） ---

# 定义一个名为 greet 的函数，它接受收一个参数 name
def greet(name: str):
    """这是一个简单的问候函数。"""   # 这是函数的文档字符串，要养成这个习惯
    print(f"你好，{name}！欢迎来到函数的世界。")

# 定义一个名为 add 的函数，它接受两个参数为 a 和 b，并返回它们的和
def add(a: int, b: int) -> int:
    """这个函数计算两个整数的和并返回结果。"""
    result = a + b
    return result

# --- 调用我们定义的函数 ---
print("--- 调用函数 ---")

# 调用 greet 函数，并传入“黄高朋”作为参数
greet("黄高朋")

# 调用 add 函数，传入 10 和 20 作为参数
# 并用一个变量 sum_result 来接收函数返回的结果
sum_result = add(10, 20)
print(f"调用 add(10, 20) 的返回结果是：{sum_result}")
print(f"5 + 7 的结果是: {add(5, 7)}")

# ----2. 列表方法：模拟栈 ---
print("\n--- 模拟栈 ---")

stack = []    # 创建一个空列表作为我们的栈

# 入栈操作
print("入栈 -> 'A'")
stack.append('A')
print("入栈 -> 'B'")
stack.append('B')
print("入栈 -> 'C'")
stack.append('C')

print(f"当前栈中的内容：{stack}")

# 出栈操作
poppend_item = stack.pop()
print(f"出栈 -> '{poppend_item}'")
print(f"当前栈中的内容：'{stack}'")

poppend_item = stack.pop()
print(f"出栈 -> '{poppend_item}'")
print(f"当前栈中的内容：'{stack}'")

positive_numbers_squared = [num * num for num in [1, -2, 3, -4, 5] if num >0]
print(positive_numbers_squared)

nums_input = input("请输入数组（用逗号分隔）: ")
nums = list(map(int, nums_input.split(',')))  # 将输入的字符串转成整数列表
target = int(input("请输入目标值: "))
