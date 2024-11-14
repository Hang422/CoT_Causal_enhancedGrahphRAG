from datasets import load_dataset
import pandas as pd

# 加载数据集
dataset = load_dataset("openlifescienceai/medmcqa")

# 转换为DataFrame
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])
validation_df = pd.DataFrame(dataset['validation'])

# 保存为parquet文件（如果需要）
train_df.to_parquet('train.parquet')
test_df.to_parquet('test.parquet')
validation_df.to_parquet('validation.parquet')

print(train_df.head())