# --- 1. 定义一个 “Person” 类 （蓝图） ---
class Person:
    # __init__方法：当一个 Person 对象被创建时，此方法自动被调用
    # 它接收 name 和 age 两个“原材料”
    def __init__(self, name: str, age: int):
        print(f"一个新的 Person 对象 '{name}' 正在被创建...")

        # 使用 self 将传入的 name 和 age 存为对象自身的属性
        self.name = name
        self.age = age

    # 定义一个方法（功能），第一个参数必须是 self
    def greet(self):
        # 方法内部通过 self.name 和 self.age 来访问对象自己的属性
        print(f"你好！我是{self.name}，我今年 {self.age} 岁。")

# ---2. 根据蓝图创建真实的对象（实例化） ---
print("--- 开始创建对象 ---")
# 创建第一个 Person 对象，传入名字和数字作为初始属性
person1 = Person("ChatGPT", 3)

# 创建第二个 Person 对象，传入一个名字和数字
person2 = Person("Gemini", 2)
print("--- 对象创建完毕 ---\n")

# --- 3. 调用对象的方法 ---
print("--- 开始调用对象 ---")
# 调用 person1 对象的 greet 方法
person1.greet()

# 调用 person2 对象的 greet 方法
person2.greet()

# --- 4. 访问对象的属性 ---
print("\n--- 开始访问属性 ---")
print(f"第一个人的名字是：{person1.name}")
print(f"第二个人的年龄是：{person2.age}")