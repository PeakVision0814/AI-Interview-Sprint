import  pandas as pd

df = pd.read_csv("../../data/train.csv")

# 默认显示前 5 行
print("数据前5行预览:")
print(df.head())

print("\n数据基本信息:")
df.info()

print("\n数值列统计描述:")
print(df.describe())

# 选择单列，返回一个 Series
ages = df['Age']
print(f"\n选择 'Age' 单列的前5个值：\n{ages.head()}")

# 选择多列，注意要用两层方括号 [[]]，返回一个 DataFrame
subset = df[['Name', 'Sex', 'Age']]
print(f"\n选择 'Name', 'Sex', 'Age' 多列的前5行:\n{subset.head()}")

# .loc：选择行标签为0，1，2，列标签为 'Name' 和 'Sex' 的数据
print(f"\n使用 .loc 选择指定行和列:\n{df.loc[0:2, ['Name', 'Sex']]}")

# .iloc: 选择第 0-2 行，第 3-4 列的数据
print(f"\n使用 .iloc 选择指定位置的行和列:\n{df.iloc[0:3, 3:5]}")

# 步骤 1: 创建一个布尔 Series (条件)
condition = df['Age'] > 60
print(f"\n年龄大于60岁的乘客 (True/False Series 的前10个):\n{condition.head(10)}")

# 步骤 2: 将布尔 Series 传入 [] 中，筛选出所有满足条件的行
senior_passengers = df[condition]
print(f"\n筛选出的高龄乘客信息:\n{senior_passengers.head()}")

# 通常写成一行
young_passengers = df[df['Age'] < 18]

print("-" * 35)

print("--- Start today's tasks ---")

print("task 1: Load train.csv data:")
print("\nPreview of the first 5 rows of data:")
df = pd.read_csv("../data/train.csv")
print(df.head())

print("\nData Basic Information:")
print(df.info())

print("\nTask 2: Filter all passengers with first class (Pclass == 1)")
first_class = df[df['Pclass'] == 1]
print(f"\nPassengers in 1st class:\n{first_class.head()}")

young_female_df = df[(df['Sex'] == 'female') & (df['Age'] < 18)]
print(f"\nall female passengers who are minors (Age < 18) and female (Sex == 'female'):\n{young_female_df.head()}")