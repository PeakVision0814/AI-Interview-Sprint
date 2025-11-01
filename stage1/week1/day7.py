import pandas as pd

df = pd.read_csv("../../data/train.csv")
print(df.head())

# 精确检查缺失值：

print("\nNumber of missing values per column:")
print(df.isnull().sum())

# 填充缺失值 (.fillna()):
# 计算 Age 列的平均值
mean_age = df['Age'].mean()
print(f"\n计算出的平均年龄是: {mean_age:.2f}")

# 使用平均值填充所有 Age 列的 NaN 值
# .fillna() 会返回一个新的 Series，我们需要将其赋值回原列
df['Age'] = df['Age'].fillna(mean_age)

# 再次检查 Age 列的缺失值，确认是否已填充成功
print("\n填充后，Age列的缺失值数量:", df['Age'].isnull().sum())

# 1. 基础groupby操作：
# 按 'Sex' 列分组，然后计算 'Survived' 列的平均值
# 这行代码直接回答了“不同性别的平均存活率是多少”
survival_rate_by_sex = df.groupby('Sex')['Survived'].mean()
print(f"\n按性别划分的平均存活率：\n{survival_rate_by_sex}")

# 也可以按照 'Pclass'（船舱等级）分组
survival_rate_by_pclass = df.groupby('Pclass')['Survived'].mean()
print(f"\n按船舱划分的平均存活率：\n{survival_rate_by_pclass}")

# 2. 按多个列进行分组：
# 同时按 'Pclass' 和 'Sex' 进行分组
# 这回答了“在一等舱的男性、女性，二等舱的男性、女性...各自的存活率是多少？”
survival_rate_by_multi = df.groupby(['Pclass', 'Sex'])['Survived'].mean()
print(f"\n按船舱等级和性别共同划分的平均存活率:\n{survival_rate_by_multi}")



print("--- 今日实践任务 ---" * 3)

print("\n--- 1. 加载数据 ---")
df = pd.read_csv("../../data/train.csv")
print(df.head())

print("\n--- 2. 清晰数据：使用Age列的平均值填充该列的所有缺失值 ---")
# 计算Age列的平均值
age_mean = df['Age'].mean()

print("\n填前的缺失值数量:",df['Age'].isnull().sum())
print(f"\nAge的平均值为：{age_mean:.2f}")
# 使用平均值填充
df['Age'] = df['Age'].fillna(age_mean)
print("\n填充后的缺失值数量:",df['Age'].isnull().sum())

print("\n--- 3. 分析任务一：计算不同性别的平均存活率 ---")

survival_rate_sex = df.groupby('Sex')['Survived'].mean()
print(f"\n不同性别的平均存活率：\n{survival_rate_sex}")

survival_rate_Pclass = df.groupby('Pclass')['Survived'].mean()
print(f"\n不同船舱等级的平均存活率：\n{survival_rate_Pclass}")

survival_rate_pclass_sex = ((df.groupby(['Pclass', 'Sex'])['Survived'].mean()) * 100).round(2).astype(str) + '%'
print(f"\n不同性别 和 不同船舱等级 组合下的平均存活率：\n{survival_rate_pclass_sex}")

print("从分类分析可知，一等舱的女性存活率最高")