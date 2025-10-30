import  pandas as pd

df = pd.read_csv("../data/train.csv")

# 默认显示前 5 行
print("数据前5行预览:")
print(df.head())