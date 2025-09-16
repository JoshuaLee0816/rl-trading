import pandas as pd
from pathlib import Path
import sys

# ========= CONFIG 區塊 =========
PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJ_ROOT))

MODE = "full_300"   # === _300 方便區分 ===

RAW_FILE = PROJ_ROOT / "data" / "raw" / "ALL_RAW_DATA_2015_2024_wide.csv"
OUT_DIR = PROJ_ROOT / "data" / "processed" / MODE
WF_DIR = OUT_DIR / "walk_forward"

# === _300 ===
FEATURE_LONG_FILE = OUT_DIR / f"ALL_feature_long_2015_2024_{MODE}.parquet"
SUMMARY_FILE = WF_DIR / f"WF_summary_2015_2024_{MODE}.csv"

# 技術指標
MA_WINDOWS = [5, 20, 34, 60, 120]
RSI_WINDOW = 14
VOLUME_MA_WINDOWS = [20, 60]
MACD_PARAMS = (12, 26, 9)
KD_PARAMS = (9, 3, 3)
ROUND_DIGITS = 3

# Walk-forward 切分方式
SPLIT_MODE = "expanding"            # 可選 "expanding" 或 "rolling"
WARMUP_YEARS = 5                    # expanding 模式：至少幾年起始
ROLLING_YEARS = 5                   # rolling 模式：固定近幾年
# ========= CONFIG END =========

from src.features.indicators import (
    add_moving_averages,
    add_rsi,
    add_volume_moving_averages,
    add_macd,
    add_kd,
)


def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """將寬表 (每檔股票多欄) 轉成長表 (date, stock_id, features...)"""
    records = []
    for col in df.columns:
        if "_" not in col:
            continue
        sid, feat = col.split("_", 1)
        records.append((col, sid, feat))

    mapper = pd.DataFrame(records, columns=["col", "stock_id", "feature"])

    melted = df.reset_index().melt(
        id_vars=["date"], value_vars=mapper["col"].tolist(),
        var_name="col", value_name="value"
    )
    melted = melted.merge(mapper, on="col", how="left")

    long_df = (
        melted.pivot_table(index=["date", "stock_id"], columns="feature", values="value")
        .reset_index()
        .sort_values(["date", "stock_id"])
    )

    # 保證 stock_id 為四位數字
    long_df["stock_id"] = long_df["stock_id"].astype(str).str.extract(r"(\d{1,4})")[0].str.zfill(4)

    # === 調整欄位順序 ===
    base_cols = ["date", "stock_id", "open", "high", "low", "close",
                 "volume", "cash_dividend", "stock_dividend"]

    ma_cols = [f"MA{w}" for w in sorted(MA_WINDOWS)]
    vma_cols = [f"VMA{w}" for w in sorted(VOLUME_MA_WINDOWS)]
    rsi_cols = [f"rsi{RSI_WINDOW}"]
    macd_cols = ["macd", "macd_signal", "macd_hist"]

    ordered = base_cols + ma_cols + vma_cols + rsi_cols + macd_cols
    other_cols = [c for c in long_df.columns if c not in ordered]

    long_df = long_df[[c for c in ordered if c in long_df.columns] + other_cols]

    return long_df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WF_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] loading {RAW_FILE}")
    df = pd.read_csv(RAW_FILE, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")

    # 找出所有股票 ID
    stock_ids = sorted({c.split("_")[0] for c in df.columns if c.endswith("_close")})
    print(f"[INFO] total {len(stock_ids)} stocks in dataset (MODE={MODE})")

    # === 初步清理 (先丟掉缺漏太多的股票) ===
    pre_clean_frames = []
    for sid in stock_ids:
        cols = [f"{sid}_open", f"{sid}_high", f"{sid}_low", f"{sid}_close",
                f"{sid}_volume", f"{sid}_cash_dividend", f"{sid}_stock_dividend"]
        sub_cols = [c for c in cols if c in df.columns]
        sub = df.reindex(columns=sub_cols).copy()

        missing_ratio = sub.isna().mean().max() if not sub.empty else 1.0
        if missing_ratio > 0.1:
            print(f"[DROP] {sid} missing {missing_ratio:.1%}, skip (before turnover calc).")
            continue

        sub = sub.ffill().bfill()
        pre_clean_frames.append(sub)

    if not pre_clean_frames:
        print("[ERROR] no stocks left after initial cleaning")
        return

    pre_clean_df = pd.concat(pre_clean_frames, axis=1).sort_index()

    # === 計算平均成交金額 (price * volume) ===
    avg_turnover = {}
    for sid in stock_ids:
        if f"{sid}_close" in pre_clean_df.columns and f"{sid}_volume" in pre_clean_df.columns:
            turnover = pre_clean_df[f"{sid}_close"] * pre_clean_df[f"{sid}_volume"]
            avg_turnover[sid] = turnover.mean(skipna=True)

    # === 挑前300檔 ===
    top300 = sorted(avg_turnover.items(), key=lambda x: x[1], reverse=True)[:300]
    top300_ids = {sid for sid, _ in top300}
    print(f"[INFO] selected top {len(top300_ids)} stocks by avg turnover")

    # === 存出 Top300 股票清單 ===
    top300_df = pd.DataFrame(top300, columns=["stock_id", "avg_turnover"])
    top300_file = OUT_DIR / f"top300_list_{MODE}.csv"
    top300_df.to_csv(top300_file, index=False)
    print(f"[OK] saved top300 list -> {top300_file}")

    # === 只保留前300檔 ===
    clean_frames = []
    for sid in top300_ids:
        cols = [f"{sid}_open", f"{sid}_high", f"{sid}_low", f"{sid}_close",
                f"{sid}_volume", f"{sid}_cash_dividend", f"{sid}_stock_dividend"]
        sub_cols = [c for c in cols if c in df.columns]
        sub = df.reindex(columns=sub_cols).copy()
        sub = sub.ffill().bfill()
        clean_frames.append(sub)

    clean_df = pd.concat(clean_frames, axis=1).sort_index()

    # === 技術指標 ===
    feat_df = add_moving_averages(clean_df, MA_WINDOWS)
    feat_df = add_rsi(feat_df, window=RSI_WINDOW)
    feat_df = add_volume_moving_averages(feat_df, windows=VOLUME_MA_WINDOWS)
    feat_df = add_macd(feat_df, *MACD_PARAMS)
    feat_df = add_kd(feat_df, *KD_PARAMS)

    # 丟掉 warm-up
    warmup = max(MA_WINDOWS + [RSI_WINDOW])
    feat_df = feat_df.iloc[warmup:].copy()

    # 四捨五入
    feat_df = feat_df.round(ROUND_DIGITS)

    # === 直接轉長表 ===
    feat_long = wide_to_long(feat_df)

    # === 存 parquet (只有長表) ===
    feat_long.to_parquet(FEATURE_LONG_FILE, engine="pyarrow", index=False)
    print(f"[OK] saved feature (long) -> {FEATURE_LONG_FILE}")

    # === Walk-forward splits (用長表) ===
    years = feat_long["date"].dt.year.unique()
    start_year, end_year = years.min(), years.max()

    summary_rows = []
    for test_year in range(start_year + WARMUP_YEARS, end_year + 1):
        if SPLIT_MODE == "expanding":
            train = feat_long[feat_long["date"].dt.year < test_year]
        elif SPLIT_MODE == "rolling":
            train = feat_long[(feat_long["date"].dt.year >= test_year - ROLLING_YEARS) &
                              (feat_long["date"].dt.year < test_year)]
        else:
            raise ValueError(f"Unknown SPLIT_MODE: {SPLIT_MODE}")

        test = feat_long[feat_long["date"].dt.year == test_year]
        if train.empty or test.empty:
            continue

        # train 範圍：最小年 ~ 最大年
        train_start = train["date"].dt.year.min()
        train_end = train["date"].dt.year.max()

        # === 修改：輸出檔案名也加上 _300 ===
        train_file = WF_DIR / f"WF_train_{train_start}_{train_end}_{MODE}.parquet"
        test_file = WF_DIR / f"WF_test_{test_year}_{MODE}.parquet"

        train.to_parquet(train_file, engine="pyarrow", index=False)
        test.to_parquet(test_file, engine="pyarrow", index=False)

        summary_rows.append({
            "split_mode": SPLIT_MODE,
            "train_file": train_file.name,
            "train_years": f"{train_start}-{train_end}",
            "train_rows": len(train),
            "test_file": test_file.name,
            "test_year": test_year,
            "test_rows": len(test)
        })
        print(f"✅ {train_file.name} ({len(train)}) | {test_file.name} ({len(test)})")

    # Summary
    pd.DataFrame(summary_rows).to_csv(SUMMARY_FILE, index=False)
    print(f"[OK] summary saved -> {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
