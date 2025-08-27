#路徑儲存工具
from pathlib import Path
import pandas as pd

def save_csv(df: pd.DataFrame, path: Path) -> None:
    """存成 UTF-8 CSV（含欄位名），自動建資料夾"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
