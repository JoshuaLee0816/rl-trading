import pandas as pd

df = pd.read_parquet("data/processed/full/walk_forward/WF_train_2015_2019_full.parquet")

print(df.head())       # 看前 5 列
print(df.columns)      # 看欄位名稱
print(len(df))         # 總筆數
