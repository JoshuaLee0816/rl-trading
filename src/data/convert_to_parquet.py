"""
轉成parquet會快很多 比讀取csv還要快
"""


# convert_to_parquet.py
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  
csv_path = ROOT / "data" / "processed" / "stocks_20_with_market_index_2015-2020_long.csv"
parquet_path = ROOT / "data" / "processed" / "stocks_20_with_market_index_2015-2020_long.parquet"

print(f"[INFO] 讀取 CSV → {csv_path}")
df = pd.read_csv(csv_path, parse_dates=["date"])

print(f"[INFO] 轉存 Parquet → {parquet_path}")
df.to_parquet(parquet_path, engine="pyarrow", index=False)

print("[OK] 完成轉檔")
