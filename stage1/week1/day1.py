# --- 1. 变量赋值与数据类型 ---

# 我们创建了一个名为 a 的变量，并把整数 10 存进去
a = 10

# 我们创建了一个名为 pi 的变量，并把小数 3.14 存进去
pi = 3.14

# 我们创建了一个名为 message 的变量，并把一段文本存进去
message = "你好，Python！"

# --- 2. 检查它们的类型 ---

# Python 内置了一个非常有用的函数 type()，可以告诉我们一个变量到底是什么类型
print("--- 检查数据类型 ---")
print(f"变量 a 的值是: {a}, 它的类型是: {type(a)}")
print(f"变量 pi 的值是: {pi}, 它的类型是: {type(pi)}")
print(f"变量 message 的值是: {message}, 它的类型是: {type(message)}")

# --- 3. 观察不同类型如何影响操作 ---

# 两个整数相加
num1 = 100
num2 = 50
print("\n--- 观察操作 ---") # \n 是一个特殊字符，表示换行
print(f"数字 {num1} + {num2} 的结果是: {num1 + num2}")

# 两个字符串“相加”（实际上是拼接）
str1 = "深度"
str2 = "学习"
print(f"字符串 '{str1}' + '{str2}' 的结果是: {str1 + str2}")

# 尝试将数字和字符串相加，看看会发生什么
# print(f"尝试: {num1} + {str1}")
# ↑ 上面这一行代码是注释掉的。如果你去掉开头的 # 号并运行，Python会报错。
# 这正是因为它们的数据类型不同！

# --- 4. 类型转换 (Type Casting) ---

num1 = 100
str1 = "深度学习"

# 错误的做法，我们会得到 TypeError
# print(f"错误的做法: {num1 + str1}") 

# 正确的做法：在拼接前，用 str() 将 num1 转换成字符串
print("\n--- 类型转换实践 ---")
converted_num1 = str(num1)
print(f"num1 原本的类型是: {type(num1)}")
print(f"用 str() 转换后，类型变成了: {type(converted_num1)}")
print(f"成功拼接: {converted_num1 + str1}")

# 一个更常见的例子：处理用户输入
# input() 函数接收用户输入，但它返回的永远是字符串！
# user_age_str = input("请输入你的年龄: ") 
# new_age = user_age_str + 5 # 这里会报错！因为 str 不能和 int 相加
# print(f"五年后，你 {new_age} 岁了。")

# 正确处理用户输入
print("\n--- 处理用户输入 ---")
user_age_str = "25" # 我们假设用户输入了 "25"
print(f"用户输入的 '25'，类型是: {type(user_age_str)}")

# 用 int() 将其转换为整数才能进行计算
user_age_int = int(user_age_str)
print(f"转换后的 25，类型是: {type(user_age_int)}")

new_age = user_age_int + 5
print(f"五年后，你将是 {new_age} 岁。")

# --- 5. 字符串的 .split() 和 .join() ---

sentence = "Python,Pandas,PyTorch"

# 使用 .split(',') 将字符串都好切分
libs_list = sentence.split(',')

print("\n --- 字符串操作 ---")
print(f"原始字符串: '{sentence}'")
print(f"切分后的结果: {libs_list}")
print(f"切分后,数据类型变成了: {type(libs_list)}")  # 注意看！它变成了 <class 'list'>

# .split() 的结果是一个列表，我们可以单独访问里面的每一个元素
print(f"列表的第二个元素是: '{libs_list[1]}'")

# 现在我们反过来，用 .join() 把列表合并成字符串
connector = " + "
joined_string = connector.join(libs_list) # 注意语法：是 '连接符'.join(列表)

print(f"\n用 '{connector}' 连接列表后的结果是: '{joined_string}'")
print(f"合并后，数据类型又变回了: {type(joined_string)}")

# --- 6. 列表 (List) 的核心操作 ---

# 一个混合类型的列表
my_list = [10, 20.5, "Python", 40, 50]

print("\n--- 列表操作 ---")
print(f"完整的列表: {my_list}")

# 索引 (精确获取一个)
print(f"索引 0 (第一个元素): {my_list[0]}")
print(f"索引 2: {my_list[2]}")
print(f"索引 -1 (最后一个元素): {my_list[-1]}")

# 切片 (获取一段)
# 获取索引从 1 到 3 的元素 (即第2、3、4个元素)
sub_list = my_list[1:4] 
print(f"切片 [1:4] 的结果: {sub_list}")

# 一些常用的切片技巧
print(f"切片 [:3] (从头到索引2): {my_list[:3]}")
print(f"切片 [2:] (从索引2到末尾): {my_list[2:]}")
print(f"切片 [::-1] (反转整个列表): {my_list[::-1]}")

# --- 7. 控制流 if-elif-else (交互式版本) ---

print("\n--- 交互式条件判断 ---")

# 1. 使用 input() 获取用户输入，并用一个字符串变量存储它
user_input_str = input("请输入一个整数，然后按回车: ")

print(f"你输入的内容是 '{user_input_str}', 它的类型是: {type(user_input_str)}")

# 2. 使用 int() 将输入的字符串转换为整数，才能进行数学比较
#    我们把转换后的结果存入 num 变量
num = int(user_input_str)

print(f"转换后，数字是 {num}, 它的类型是: {type(num)}")

# 3. 这里的 if-elif-else 逻辑和之前完全一样！
#    它操作的是我们转换好的整数 num
if num > 0:
    print("这是一个正数")
elif num == 0:
    print("这是零")
else:
    print("这是一个负数")

print("判断结束。")