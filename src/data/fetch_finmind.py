from __future__ import annotations

import argparse
import os
import sys
import pandas as pd
import time
from collections import deque
from pathlib import Path
from typing import List, Optional

# 匯入 config (settings.py)
PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJ_ROOT / "src"))
import config.settings as cfg  # type: ignore

DATA_RAW = getattr(cfg, "DATA_RAW", PROJ_ROOT / "data" / "raw")
DEFAULT_START = getattr(cfg, "DEFAULT_START")
DEFAULT_END = getattr(cfg, "DEFAULT_END")
UNIVERSE = getattr(cfg, "UNIVERSE")

#------RateLimiter------ same as build_twse_universe.py did
MAX_CALLS_PER_HOUR = 500
WINDOW_SECONDS = 3600.0

class _RateLimiter:
    def __init__(self, max_calls: int, period: float):
        self.max_calls = int(max_calls)
        self.period = float(period)
        self._ts = deque()

    def acquire(self):
        while True:
            now = time.time()
            while self._ts and (now - self._ts[0] >= self.period):
                self._ts.popleft()
            if len(self._ts) < self.max_calls:
                self._ts.append(now)
                return
            wait = self.period - (now - self._ts[0]) + 0.01
            time.sleep(max(0.01, min(wait, 5.0)))

_limiter = _RateLimiter(MAX_CALLS_PER_HOUR, WINDOW_SECONDS)

def login_loader(user: str, password: str):
    from FinMind.data import DataLoader
    _limiter.acquire()
    dl = DataLoader()
    dl.login(user, password)
    return dl

def _dl_daily(dl, stock_id: str, start: str, end: str):
    _limiter.acquire()
    if hasattr(dl, "taiwan_stock_price"):
        return dl.taiwan_stock_price(stock_id=stock_id, start_date=start, end_date=end)
    return dl.taiwan_stock_daily(stock_id=stock_id, start_date=start, end_date=end)

def _dl_dividend(dl, stock_id: str, start: str, end: str):
    _limiter.acquire()
    return dl.taiwan_stock_dividend(stock_id=stock_id, start_date=start, end_date=end)



# 欄位對齊
def _unify_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Date": "date", "date": "date", "trading_date": "date",
        "stock_id": "stock_id",
        "Open": "open", "open": "open",
        "High": "high", "high": "high", "max": "high",
        "Low": "low", "low": "low", "min": "low",
        "Close": "close", "close": "close",
        "Trading_Volume": "volume", "volume": "volume",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    keep = [c for c in ["date", "stock_id", "open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)

def _unify_dividend(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Date": "date", "date": "date",
        "stock_id": "stock_id",
        "cash_dividend": "cash_dividend", "CashEarningsDistribution": "cash_dividend",
        "stock_dividend": "stock_dividend", "StockEarningsDistribution": "stock_dividend",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    keep = [c for c in ["date", "stock_id", "cash_dividend", "stock_dividend"] if c in df.columns]
    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in ["cash_dividend", "stock_dividend"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = 0.0  # 萬一缺欄位就補 0
    return df.sort_values("date").reset_index(drop=True)
# ----------------------------
# 多檔抓取（唯一入口）
# ----------------------------
def fetch_universe(
    stock_ids: List[str],
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    out_dir: Path = DATA_RAW,
    user: Optional[str] = os.getenv("FINMIND_USER"),
    password: Optional[str] = os.getenv("FINMIND_PASSWORD"),
) -> None:
    if not user or not password:
        raise RuntimeError("請在 .env 設定 FINMIND_USER / FINMIND_PASSWORD")

    out_dir.mkdir(parents=True, exist_ok=True)
    dl = login_loader(user, password)  # 只登入一次

    from data.io_utils import save_csv  # 延遲載入避免循環匯入

    for sid in stock_ids:
        print(f"[INFO] fetching {sid} {start}~{end}")

        # 1) 日價
        px = _dl_daily(dl, sid, start, end)
        px = _unify_ohlcv(px)
        if px.empty:
            print(f"[WARN] {sid} price empty, skip.")
            continue

        # 2) 股利
        try:
            dv = _dl_dividend(dl, sid, start, end)
            dv = _unify_dividend(dv)
        except Exception as e:
            print(f"[WARN] dividend fetch failed for {sid}: {e}")
            dv = pd.DataFrame(columns=["date", "stock_id", "cash_dividend", "stock_dividend"])

        # 3) 合併與輸出
        if dv.empty:
            # 沒有股利資料 → 人工補欄位
            px["cash_dividend"] = 0.0
            px["stock_dividend"] = 0.0
            out = px
        else:
            # 確保股利欄位存在
            if "cash_dividend" not in dv.columns:
                dv["cash_dividend"] = 0.0
            if "stock_dividend" not in dv.columns:
                dv["stock_dividend"] = 0.0

            out = px.merge(dv, on=["date", "stock_id"], how="left")
            out["cash_dividend"] = out["cash_dividend"].fillna(0.0)
            out["stock_dividend"] = out["stock_dividend"].fillna(0.0)

        save_csv(out, out_dir / f"{sid}.csv")
        print(f"[OK] saved -> {out_dir / (sid + '.csv')} (rows={len(out)})")


# CLI 參數
def _parse_args():
    p = argparse.ArgumentParser(description="Fetch multiple TW stocks (OHLCV + dividends) into data/raw/ via login.")
    p.add_argument("--tickers", nargs="*", default=None, help="e.g. 2330 2317 2454; default uses settings.UNIVERSE")
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    tickers = args.tickers if args.tickers else UNIVERSE
    fetch_universe(tickers, start=args.start, end=args.end)
    print("Done.")
