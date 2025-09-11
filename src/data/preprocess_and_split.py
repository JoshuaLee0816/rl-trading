import pandas as pd
from pathlib import Path
import sys

# ========= CONFIG 區塊 =========
PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJ_ROOT))

MODE = "full"   # "full" (全股池) / "sample20" (隨機20檔)

RAW_FILE = PROJ_ROOT / "data" / "raw" / "ALL_RAW_DATA_2015_2024_wide.csv"
OUT_DIR = PROJ_ROOT / "data" / "preprocessed" / MODE
WF_DIR = OUT_DIR / "walk_forward"

CLEAN_FILE = OUT_DIR / f"ALL_clean_2015_2024_{MODE}.parquet"
FEATURE_FILE = OUT_DIR / f"ALL_feature_2015_2024_{MODE}.parquet"
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


def _reorder_by_sid(df: pd.DataFrame,
                    ma_windows: list[int],
                    rsi_window: int,
                    volume_ma_windows: list[int],
                    macd_suffixes=["macd", "macd_signal", "macd_hist"]) -> pd.DataFrame:
    """依股票分組，欄位順序統一"""
    base_suffix = ["open", "high", "low", "close", "volume",
                   "cash_dividend", "stock_dividend"]
    sids = sorted({c.split("_")[0] for c in df.columns if "_" in c})

    ordered_cols = []
    for sid in sids:
        # 基本欄位
        for sfx in base_suffix:
            col = f"{sid}_{sfx}"
            if col in df.columns:
                ordered_cols.append(col)
        # MA
        for w in ma_windows:
            col = f"{sid}_MA{w}"
            if col in df.columns:
                ordered_cols.append(col)
        # RSI
        rcol = f"{sid}_rsi{rsi_window}"
        if rcol in df.columns:
            ordered_cols.append(rcol)
        # Volume MA
        for w in volume_ma_windows:
            col = f"{sid}_VMA{w}"
            if col in df.columns:
                ordered_cols.append(col)
        # MACD
        for sfx in macd_suffixes:
            col = f"{sid}_{sfx}"
            if col in df.columns:
                ordered_cols.append(col)

    return df.reindex(columns=[c for c in ordered_cols if c in df.columns])


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

    clean_frames = []
    for sid in stock_ids:
        cols = [f"{sid}_open", f"{sid}_high", f"{sid}_low", f"{sid}_close",
                f"{sid}_volume", f"{sid}_cash_dividend", f"{sid}_stock_dividend"]
        sub_cols = [c for c in cols if c in df.columns]
        sub = df.reindex(columns=sub_cols).copy()

        missing_ratio = sub.isna().mean().max() if not sub.empty else 1.0
        if missing_ratio > 0.1:
            print(f"[DROP] {sid} missing {missing_ratio:.1%}, skip.")
            continue

        sub = sub.ffill().bfill()
        clean_frames.append(sub)

    if not clean_frames:
        print("[ERROR] no stocks left after cleaning")
        return

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

    # 欄位順序
    feat_df = _reorder_by_sid(feat_df, MA_WINDOWS, RSI_WINDOW, VOLUME_MA_WINDOWS)

    # 四捨五入
    clean_df = clean_df.round(ROUND_DIGITS)
    feat_df = feat_df.round(ROUND_DIGITS)

    # === 存 parquet (寬表) ===
    clean_df.to_parquet(CLEAN_FILE, engine="pyarrow")
    feat_df.to_parquet(FEATURE_FILE, engine="pyarrow")
    print(f"[OK] saved clean -> {CLEAN_FILE}")
    print(f"[OK] saved feature (wide) -> {FEATURE_FILE}")

    # === 存 parquet (長表) ===
    feat_long = wide_to_long(feat_df)
    feat_long.to_parquet(FEATURE_LONG_FILE, engine="pyarrow", index=False)
    print(f"[OK] saved feature (long) -> {FEATURE_LONG_FILE}")

    # === Walk-forward splits ===
    years = feat_df.index.year.unique()
    start_year, end_year = years.min(), years.max()

    summary_rows = []
    for test_year in range(start_year + WARMUP_YEARS, end_year + 1):
        if SPLIT_MODE == "expanding":
            train = feat_df[feat_df.index.year < test_year]
        elif SPLIT_MODE == "rolling":
            train = feat_df[(feat_df.index.year >= test_year - ROLLING_YEARS) &
                            (feat_df.index.year < test_year)]
        else:
            raise ValueError(f"Unknown SPLIT_MODE: {SPLIT_MODE}")

        test = feat_df[feat_df.index.year == test_year]
        if train.empty or test.empty:
            continue

        train_file = WF_DIR / f"WF_train_until_{test_year-1}_{MODE}.parquet"
        test_file = WF_DIR / f"WF_test_{test_year}_{MODE}.parquet"

        train.to_parquet(train_file, engine="pyarrow")
        test.to_parquet(test_file, engine="pyarrow")

        summary_rows.append({
            "split_mode": SPLIT_MODE,
            "train_file": train_file.name,
            "train_years": f"{train.index.year.min()}-{train.index.year.max()}",
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
