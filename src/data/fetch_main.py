from pathlib import Path
import os
import time
import random
import pandas as pd

from fetch_finmind import (
    login_loader, _dl_daily, _dl_dividend, _unify_ohlcv, _unify_dividend
)

DATA_RAW = Path("data/raw")
UNIVERSE_FILE = Path("data/twse_universe_2015_2024.xlsx")  # 讀 Universe 清單
START_DATE = "2015-01-01"
END_DATE   = "2020-12-31"
N_SAMPLES = 20 #random amount

def _load_numeric_ids(path: Path) -> list[str]: #之後要全部讀進來 action space 要忽略掉指數不能交易的部分 
    """從 xlsx 的 'Universe'（若不存在取第一張）讀出所有「純數字」stock_id。"""
    if not path.exists():
        raise FileNotFoundError(f"universe file not found: {path}")
    xls = pd.ExcelFile(path)
    sheet = "Universe" if "Universe" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet, dtype=str)

    col = "stock_id" if "stock_id" in df.columns else df.columns[0]
    ids = (
        df[col].dropna().astype(str).str.strip()
          .str.replace(r"\.0$", "", regex=True)
    )
    # 只留 4~6 位數字代碼（排除行業/指數等）
    ids = ids[ids.str.fullmatch(r"\d{4,6}")]
    out = ids.unique().tolist()
    if not out:
        raise RuntimeError("no numeric stock_id parsed from universe file.")
    return out

def main():
    user = os.getenv("FINMIND_USER")
    password = os.getenv("FINMIND_PASSWORD")
    if not user or not password:
        raise RuntimeError("請在 .env 設定 FINMIND_USER / FINMIND_PASSWORD")
    dl = login_loader(user, password)

    # 2) 讀清單 → 隨機抽 20 檔
    all_ids = _load_numeric_ids(UNIVERSE_FILE)
    if N_SAMPLES is None:
        sample_ids = all_ids
        print("抓取全部股票，共", len(sample_ids), "檔")
    else:
        k = min(N_SAMPLES, len(all_ids))
        sample_ids = random.sample(all_ids, k)
        print("隨機挑選股票：", sample_ids)
    print(f"抓取區間: {START_DATE} ~ {END_DATE}")

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    # 3) 逐檔抓（日價 + 股利）→ 整理 → 加上前綴 → 收集成寬表
    frames = []
    for sid in sample_ids:
        print(f"[INFO] fetching {sid} {START_DATE}~{END_DATE}")

        # 股價
        px = _dl_daily(dl, sid, START_DATE, END_DATE)
        px = _unify_ohlcv(px)
        if px.empty:
            print(f"[WARN] {sid} price empty, skip.")
            continue

        # 股利
        try:
            dv = _dl_dividend(dl, sid, START_DATE, END_DATE)
            dv = _unify_dividend(dv)
        except Exception as e:
            print(f"[WARN] dividend fetch failed for {sid}: {e}")
            dv = pd.DataFrame(columns=["date", "stock_id", "cash_dividend", "stock_dividend"])

        # 合併與補值
        if dv.empty:
            px["cash_dividend"] = 0.0
            px["stock_dividend"] = 0.0
            out = px
        else:
            out = px.merge(dv, on=["date", "stock_id"], how="left")
            out["cash_dividend"] = out["cash_dividend"].fillna(0.0)
            out["stock_dividend"] = out["stock_dividend"].fillna(0.0)

        # 改成寬表欄名：<sid>_<field>
        out = out.set_index("date").sort_index()
        out = out.add_prefix(f"{sid}_")  # 例如 2330_open, 2330_close
        frames.append(out)

        # （可選）微小延遲；真正限流已在 _dl_* 內處理
        time.sleep(0.1)

    # --- 抓大盤指數 ---
    try:
        print(f"[INFO] fetching TWII {START_DATE}~{END_DATE}")
        idx = _dl_daily(dl, "^TWA00", START_DATE, END_DATE)   # 有些版本要用 "TWA00"
        idx = _unify_ohlcv(idx)
        if not idx.empty:
            idx["cash_dividend"] = 0.0
            idx["stock_dividend"] = 0.0
            idx = idx.set_index("date").sort_index()
            idx = idx.add_prefix("TWII_")   # 例如 TWII_open, TWII_close
            frames.append(idx)
            print(f"[OK] TWII appended (rows={len(idx)})")
        else:
            print("[WARN] TWII data empty, skip.")
    except Exception as e:
        print(f"[WARN] TWII fetch failed: {e}")

    # 4) 橫向合併 → 單一寬表輸出
    if frames:
        wide = pd.concat(frames, axis=1).sort_index()
        out_file = DATA_RAW / "stocks_20_with_market_index_2015-2020_wide.csv"
        wide.reset_index().to_csv(out_file, index=False, encoding="utf-8-sig")
        print(f"[OK] saved wide file -> {out_file} (rows={len(wide)})")
    else:
        print("[WARN] no data fetched.")

if __name__ == "__main__":
    main()
