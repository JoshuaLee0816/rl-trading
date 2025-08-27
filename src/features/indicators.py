import pandas as pd

def add_moving_averages(df: pd.DataFrame, windows=[5, 20, 34, 60]) -> pd.DataFrame:
    """
    對每一檔股票的 close 欄位計算移動平均 (MA)，並新增欄位。
    欄位格式: {stock_id}_MA{window}
    """
    out = df.copy()
    close_cols = [c for c in df.columns if c.endswith("_close")]

    for col in close_cols:
        sid = col.split("_")[0]   # 股票代號 (e.g. "2330")
        for w in windows:
            out[f"{sid}_MA{w}"] = (
                df[col].rolling(w).mean().ffill().bfill()
            )
    return out
