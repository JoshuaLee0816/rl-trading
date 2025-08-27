import pandas as pd
from pathlib import Path
from src.features.indicators import add_moving_averages

RAW_FILE = Path("data/raw/all_stocks_wide.csv")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 移動平均的天數
MA_WINDOWS = [5, 20, 34, 60]

def main():
    print(f"[INFO] loading {RAW_FILE}")
    df = pd.read_csv(RAW_FILE, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")

    # 找出所有股票 ID
    stock_ids = sorted(set(c.split("_")[0] for c in df.columns))
    print(f"[INFO] total {len(stock_ids)} stocks in dataset")

    drop_ids = []
    clean_frames = []

    for sid in stock_ids:
        cols = [f"{sid}_open", f"{sid}_high", f"{sid}_low", f"{sid}_close",
                f"{sid}_volume", f"{sid}_cash_dividend", f"{sid}_stock_dividend"]
        sub = df[cols].copy()

        # 計算缺值率
        missing_ratio = sub.isna().mean().max()
        if missing_ratio > 0.1:
            print(f"[DROP] {sid} missing {missing_ratio:.1%}, skip.")
            drop_ids.append(sid)
            continue

        # 缺值補上（前值補齊 + 後值補齊）
        sub = sub.ffill().bfill()

        clean_frames.append(sub)

    # 合併乾淨資料
    if not clean_frames:
        print("[ERROR] no stocks left after cleaning")
        return

    clean_df = pd.concat(clean_frames, axis=1)
    clean_file = OUT_DIR / "all_stocks_clean.csv"
    clean_df.to_csv(clean_file, encoding="utf-8-sig")
    print(f"[OK] saved clean dataset -> {clean_file}")

    # ============= 加技術指標 (MA) =============
    feat_df = add_moving_averages(clean_df, MA_WINDOWS)

    feat_file = OUT_DIR / "all_stocks_features.csv"
    feat_df.to_csv(feat_file, encoding="utf-8-sig")
    print(f"[OK] saved feature dataset -> {feat_file}")

if __name__ == "__main__":
    main()
