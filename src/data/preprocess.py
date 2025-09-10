# src/data/preprocess.py
import pandas as pd
from pathlib import Path
import sys

# 讓 'src' 可被匯入
PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJ_ROOT))

from src.features.indicators import add_moving_averages, add_rsi

RAW_FILE = Path("data/raw/stocks_20_with_market_index_2015-2020_wide.csv")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MA_WINDOWS = [5, 20, 34, 60, 120]
RSI_WINDOW = 14


def _reorder_by_sid(df: pd.DataFrame,
                    ma_windows: list[int],
                    rsi_window: int) -> pd.DataFrame:
    """
    將欄位依『每支股票』分組排序：
    [open, high, low, close, volume, cash_dividend, stock_dividend, MA*, RSI]
    只挑存在的欄位，避免 KeyError。
    """
    base_suffix = [
        "open", "high", "low", "close", "volume",
        "cash_dividend", "stock_dividend"
    ]
    sids = sorted({c.split("_")[0] for c in df.columns if "_" in c})

    ordered_cols = []
    for sid in sids:
        # 基礎欄
        for sfx in base_suffix:
            col = f"{sid}_{sfx}"
            if col in df.columns:
                ordered_cols.append(col)
        # MA 指標
        for w in ma_windows:
            col = f"{sid}_MA{w}"
            if col in df.columns:
                ordered_cols.append(col)
        # RSI 指標
        rcol = f"{sid}_rsi{rsi_window}"
        if rcol in df.columns:
            ordered_cols.append(rcol)

    # 只回傳存在的欄位，保持原資料不丟失
    ordered_cols = [c for c in ordered_cols if c in df.columns]
    return df.reindex(columns=ordered_cols)


def main():
    print(f"[INFO] loading {RAW_FILE}")
    df = pd.read_csv(RAW_FILE, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")

    # 找出所有股票 ID（只看有 close 的）
    stock_ids = sorted({c.split("_")[0] for c in df.columns if c.endswith("_close")})
    print(f"[INFO] total {len(stock_ids)} stocks in dataset")

    drop_ids, clean_frames = [], []
    for sid in stock_ids:
        cols = [f"{sid}_open", f"{sid}_high", f"{sid}_low", f"{sid}_close",
                f"{sid}_volume", f"{sid}_cash_dividend", f"{sid}_stock_dividend"]
        # 避免缺欄位報錯，僅取存在的欄位
        sub_cols = [c for c in cols if c in df.columns]
        sub = df.reindex(columns=sub_cols).copy()

        # 缺值率檢查
        missing_ratio = sub.isna().mean().max() if not sub.empty else 1.0
        if missing_ratio > 0.1:
            print(f"[DROP] {sid} missing {missing_ratio:.1%}, skip.")
            drop_ids.append(sid)
            continue

        # 缺值補齊
        sub = sub.ffill().bfill()
        clean_frames.append(sub)

    if not clean_frames:
        print("[ERROR] no stocks left after cleaning")
        return

    clean_df = pd.concat(clean_frames, axis=1).sort_index()

    # === 指標 ===
    feat_df = add_moving_averages(clean_df, MA_WINDOWS)  # 產生 <sid>_MA{w}
    feat_df = add_rsi(feat_df, window=RSI_WINDOW)        # 產生 <sid>_rsi14

    # 丟掉暖機期（最大視窗 120）
    warmup = max(MA_WINDOWS + [RSI_WINDOW])
    feat_df = feat_df.iloc[warmup:].copy()

    # 依每支股票分組重排欄位
    feat_df = _reorder_by_sid(feat_df, MA_WINDOWS, RSI_WINDOW)

    # 全欄位四捨五入到小數第 3 位
    feat_df = feat_df.round(3)
    clean_df = clean_df.round(3)

    # 輸出
    clean_file = OUT_DIR / "stocks_20_with_market_index_2015-2020_wide_clean.csv"
    clean_df.to_csv(clean_file, encoding="utf-8-sig")
    print(f"[OK] saved clean dataset -> {clean_file}")

    feat_file = OUT_DIR / "stocks_20_with_market_index_2015-2020_wide_feature.csv"
    # 需保留日期索引；若想把 date 變成欄位，改用 reset_index()
    feat_df.to_csv(feat_file, encoding="utf-8-sig")
    print(f"[OK] saved feature dataset -> {feat_file}")


if __name__ == "__main__":
    main()
