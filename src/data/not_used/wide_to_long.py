# 將「寬表」(每支股票當成一欄) 轉成「長表」(date, stock_id, features...)
import re
import argparse
import pandas as pd

PAT_STOCK_FEAT = re.compile(r"^(?P<stock>\d{4}(?:[.\-_][A-Za-z]+)?)[_\-.](?P<feat>.+)$")  # 1216_close 或 1216.TW_close
PAT_FEAT_STOCK = re.compile(r"^(?P<feat>.+)[_\-.](?P<stock>\d{4}(?:[.\-_][A-Za-z]+)?)$")  # close_1216 或 close_1216.TW

def detect_date_col(df: pd.DataFrame) -> str:
    for c in ["date", "Date", "DATE", "trade_date", "dt"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            return c
    # 退而求其次：試第一欄
    first = df.columns[0]
    try:
        df[first] = pd.to_datetime(df[first], errors="raise")
        return first
    except Exception:
        raise ValueError("找不到日期欄位，請確認有 'date' 或第一欄可轉成日期。")

def build_mapper(cols):
    mapper = {}
    for c in cols:
        s = str(c)
        m = re.search(r"(\d{4}(?:\.\w+)?)", s)     # 抓第一個 4 碼代號（可含 .TW 等）
        if not m:
            continue
        stock = m.group(1).replace("-", ".")
        # 移除代號本體與兩側分隔符，剩下就是特徵名（cash / stock / close / ...）
        feat = re.sub(rf"[\-_.]*{re.escape(stock)}[\-_.]*", "", s)
        feat = feat.strip("_.-") or "value"
        mapper[c] = (stock, feat)
    return mapper

def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    # 1) 清理
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")].copy()
    date_col = detect_date_col(df)

    # 2) 建立欄名對應 (原欄名 -> stock_id, feature)
    cols = [c for c in df.columns if c != date_col]
    mapper = build_mapper(cols)
    if not mapper:
        raise ValueError("無法辨識欄名格式，需像 1216_close、close_1216、1216_cash 等。")

    mapped_cols = [c for c in cols if c in mapper]
    map_df = pd.DataFrame(
        [{"col": c, "stock_id": mapper[c][0], "feature": mapper[c][1]} for c in mapped_cols]
    )
    # 正規化 stock_id：抓第一個4碼代號（可帶 .TW 等）
    map_df["stock_id"] = map_df["stock_id"].astype(str).str.extract(r"(\d{4}(?:\.\w+)?)")[0]

    # 3) 寬轉長（melt），再把 stock_id/feature 合回來
    melted = df[[date_col] + mapped_cols].melt(
        id_vars=[date_col], var_name="col", value_name="value"
    )
    melted = melted.merge(map_df, on="col", how="left")

    # 4) 以 (date, stock_id) 為索引，把 feature 攤成欄
    training = (
        melted.pivot_table(
            index=[date_col, "stock_id"],
            columns="feature",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={date_col: "date"})
    )

    ## 5) 收尾：再保險一次正規化、排序、扁平欄名
    # 抓前 1~4 位數字，完全忽略後綴（.TW / .TWO）
    training["stock_id"] = (
        training["stock_id"]
        .astype(str)
        .str.extract(r"(\d{1,4})")[0]   # 只保留數字
        .apply(lambda x: x.zfill(4))    # 補成四位數
    )


    training.columns = [str(c) for c in training.columns]
    training = training.sort_values(["date", "stock_id"]).reset_index(drop=True)

    print(training["stock_id"].unique()[:20])

    return training



def main():
    ap = argparse.ArgumentParser(description="將寬表轉成長表，輸出 training_data.csv")
    ap.add_argument("--input", "-i", required=True, help="寬表 CSV 路徑")
    ap.add_argument("--output", "-o", default="training_data.csv", help="輸出長表 CSV 路徑")
    args = ap.parse_args()

    df = pd.read_csv(args.input, dtype=str)
    training_data = wide_to_long(df)
    training_data.to_csv(args.output, index=False)
    print(f"OK → {args.output}  (rows={len(training_data)}, cols={len(training_data.columns)})")

    
if __name__ == "__main__":
    main()
