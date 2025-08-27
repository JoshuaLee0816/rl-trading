from src.data.fetch_finmind import _login_loader, _unify_ohlcv, _unify_dividend
import random
import os
import time
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

DATA_RAW = Path("data/raw")

def main():
    # 從 .env 讀取帳號密碼
    user = os.getenv("FINMIND_USER")
    password = os.getenv("FINMIND_PASSWORD")
    dl = _login_loader(user, password)

    # 取得全市場股票清單
    info = dl.taiwan_stock_info()
    all_ids = info["stock_id"].unique().tolist()

    # 隨機抽樣 20 檔
    sample_ids = random.sample(all_ids, 20)
    print("隨機挑選股票：", sample_ids)

    # 設定日期區間：昨天回推五年
    end_date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=365*5+1)).strftime("%Y-%m-%d")
    print(f"抓取區間: {start_date} ~ {end_date}")

    stock_frames = []

    for sid in sample_ids:
        print(f"[INFO] fetching {sid} {start_date}~{end_date}")

        # 股價
        px = dl.taiwan_stock_daily(stock_id=sid, start_date=start_date, end_date=end_date)
        px = _unify_ohlcv(px)
        if px.empty:
            print(f"[WARN] {sid} price empty, skip.")
            continue

        # 股利
        try:
            dv = dl.taiwan_stock_dividend(stock_id=sid, start_date=start_date, end_date=end_date)
            dv = _unify_dividend(dv)
        except Exception as e:
            print(f"[WARN] dividend fetch failed for {sid}: {e}")
            dv = pd.DataFrame(columns=["date", "stock_id", "cash_dividend", "stock_dividend"])

        # 合併
        if dv.empty:
            px["cash_dividend"] = 0.0
            px["stock_dividend"] = 0.0
            out = px
        else:
            out = px.merge(dv, on=["date", "stock_id"], how="left")
            out[["cash_dividend", "stock_dividend"]] = out[["cash_dividend", "stock_dividend"]].fillna(0.0)

        # 每檔股票 → 欄位加上股票代號前綴
        out = out.set_index("date")
        out = out.add_prefix(f"{sid}_")  # e.g. 2330_open, 2330_close
        stock_frames.append(out)

        time.sleep(0.5)  # 避免 API 過量

    # 合併所有股票成「寬表」
    if stock_frames:
        wide_df = pd.concat(stock_frames, axis=1)  # 橫向合併，日期對齊
        wide_df = wide_df.sort_index()
        out_file = DATA_RAW / "all_stocks_wide.csv"
        wide_df.to_csv(out_file, encoding="utf-8-sig")
        print(f"[OK] saved wide file -> {out_file} (rows={len(wide_df)})")
    else:
        print("[WARN] no data fetched.")

if __name__ == "__main__":
    main()
