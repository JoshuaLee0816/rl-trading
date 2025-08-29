#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TWSE stock universe builder (with rate limiting 600 req/hour)
- Window: 2015-01-01 ~ 2024-12-31
- Universe: TWSE main board, common shares only
- Coverage: vs 2330 trading-day proxy, >= 99% to keep
- Output: XLSX (with title and names) + CSV

Notes:
- Trading-day proxy uses stock_id "2330" (engineering assumption; 未查證)
- FinMind API calls are throttled to <=600/hour via a sliding-window limiter
"""

from __future__ import annotations
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Tuple, Iterable

import pandas as pd

# ---------------- config ----------------
START_DATE = "2015-01-01"
END_DATE   = "2024-12-31"
COVERAGE_THRESHOLD = 0.99
PROXY_STOCK_ID = "2330"

# Rate limit: at most 600 API calls per rolling 3600s window
MAX_CALLS_PER_HOUR = 600
WINDOW_SECONDS = 3600.0

# Batch progress print
BATCH = 20
SLEEP_SEC = 0.0  # 不額外延遲，由 RateLimiter 控制節流

# 非普通股（以 industry_category 排除；保守清單，未查證）
EXCLUDE_INDUSTRY = {
    "ETF", "ETN", "受益證券", "受益憑證", "特別股", "存託憑證", "DR",
    "權證", "認購權證", "認售權證", "指數投資證券", "不動產投資信託", "石油基金",
}

# 日線必備欄位（依常見 FinMind 欄位名推定；未查證）
REQUIRED_COLS = {"open", "max", "min", "close", "Trading_Volume"}


# -------------- env & deps --------------
def require_packages():
    try:
        from FinMind.data import DataLoader  # noqa
    except Exception:
        print("請先安裝 FinMind：pip install FinMind", file=sys.stderr)
        raise
    try:
        import xlsxwriter  # noqa
    except Exception:
        # 無 xlsxwriter 仍可輸出 xlsx（用 openpyxl），只是沒格式
        pass

# 嘗試自動載入 .env（根目錄 / 腳本目錄 / 上層）
try:
    from dotenv import load_dotenv, find_dotenv
    ok = load_dotenv(find_dotenv(usecwd=True), override=False)
    if not ok:
        from pathlib import Path as _P
        for _p in [_P(__file__).resolve().parent,
                   _P(__file__).resolve().parents[1],
                   _P.cwd()]:
            if (_p / ".env").exists():
                load_dotenv(_p / ".env", override=False)
                break
except Exception:
    pass


# -------------- rate limiter --------------
class RateLimiter:
    """Sliding-window limiter: <= max_calls within 'period' seconds."""
    def __init__(self, max_calls: int, period: float):
        self.max_calls = int(max_calls)
        self.period = float(period)
        self._ts = deque()  # timestamps of recent calls

    def acquire(self):
        while True:
            now = time.time()
            # drop old
            while self._ts and (now - self._ts[0] >= self.period):
                self._ts.popleft()
            if len(self._ts) < self.max_calls:
                self._ts.append(now)
                return
            # need to wait until oldest expires
            wait = self.period - (now - self._ts[0]) + 0.01
            time.sleep(max(0.01, min(wait, 5.0)))  # sleep in small chunks to be responsive


limiter = RateLimiter(MAX_CALLS_PER_HOUR, WINDOW_SECONDS)


# -------------- FinMind wrappers --------------
def login_loader():
    from FinMind.data import DataLoader
    user = os.getenv("FINMIND_USER")
    password = os.getenv("FINMIND_PASSWORD")
    if not user or not password:
        raise RuntimeError("缺少 FINMIND_USER / FINMIND_PASSWORD 環境變數。")
    dl = DataLoader()
    limiter.acquire()
    dl.login(user, password)
    return dl


def finmind_stock_info(dl) -> pd.DataFrame:
    """Wrap taiwan_stock_info with rate limiting."""
    limiter.acquire()
    return dl.taiwan_stock_info()


def finmind_daily(dl, stock_id: str, start: str, end: str) -> pd.DataFrame:
    """Get daily OHLCV with rate limiting.
    - 優先使用 taiwan_stock_price（你的 log 顯示 Dataset.TaiwanStockPrice）
    - 若該方法不存在，改用 taiwan_stock_daily（不同版本介面，未查證）
    """
    limiter.acquire()
    if hasattr(dl, "taiwan_stock_price"):
        return dl.taiwan_stock_price(stock_id=stock_id, start_date=start, end_date=end)
    if hasattr(dl, "taiwan_stock_daily"):
        return dl.taiwan_stock_daily(stock_id=stock_id, start_date=start, end_date=end)
    raise AttributeError("找不到日線方法（taiwan_stock_price / taiwan_stock_daily）。")


# -------------- core logic --------------
def load_twse_common_info(dl) -> pd.DataFrame:
    """僅留上市主板普通股（type=='twse' 且排除 EXCLUDE_INDUSTRY）"""
    info = finmind_stock_info(dl)
    info = info[info["type"].str.lower() == "twse"].copy()
    info = info[~info["industry_category"].isin(EXCLUDE_INDUSTRY)].copy()
    return info[["stock_id", "stock_name", "industry_category"]].drop_duplicates("stock_id")


def iter_chunks(seq: Iterable, n: int):
    it = iter(seq)
    while True:
        chunk = []
        try:
            for _ in range(n):
                chunk.append(next(it))
        except StopIteration:
            if chunk:
                yield chunk
            break
        yield chunk


def get_proxy_trading_days(dl, start: str, end: str, proxy_id: str = PROXY_STOCK_ID) -> pd.DatetimeIndex:
    """以 2330 日線日期為交易日 proxy（工程性假設，未查證）"""
    df = finmind_daily(dl, proxy_id, start, end)
    if df.empty:
        raise RuntimeError(f"代理標的 {proxy_id} 在 {start}~{end} 無資料，無法推導交易日。")
    cols = REQUIRED_COLS.intersection(df.columns)
    if cols:
        df = df.dropna(subset=list(cols))
    dates = pd.to_datetime(df["date"], errors="coerce").dropna().unique()
    return pd.DatetimeIndex(dates).sort_values()


def coverage_ratio_vs_proxy(dl, stock_id: str, start: str, end: str, proxy_days: pd.DatetimeIndex) -> float:
    """覆蓋率 = (該股在 proxy 交易日集合上的有效列數) / (proxy 交易日總數)"""
    df = finmind_daily(dl, stock_id, start, end)
    if df.empty or proxy_days.empty:
        return 0.0
    cols = REQUIRED_COLS.intersection(df.columns)
    valid = df.dropna(subset=list(cols)) if cols else df
    d = pd.to_datetime(valid["date"], errors="coerce")
    observed = d[d.isin(proxy_days)].nunique()
    expected = len(proxy_days)
    return (observed / expected) if expected > 0 else 0.0


def compute_universe(dl,
                     start: str,
                     end: str,
                     min_coverage: float = COVERAGE_THRESHOLD,
                     limit: int | None = None,
                     sleep_sec: float = SLEEP_SEC) -> Tuple[pd.DataFrame, pd.DataFrame]:
    info = load_twse_common_info(dl)
    # 你說總共 1239 筆（此處不強制，僅記錄；未查證）
    if limit:
        info = info.head(int(limit)).copy()

    proxy_days = get_proxy_trading_days(dl, start, end, PROXY_STOCK_ID)
    print(f"代理交易日（{PROXY_STOCK_ID}）筆數：{len(proxy_days)}")

    rows = []
    sids = info["stock_id"].tolist()
    print(f"計算覆蓋率：{len(sids)} 檔，期間 {start} ~ {end}")
    for batch in iter_chunks(sids, BATCH):
        for sid in batch:
            try:
                cov = coverage_ratio_vs_proxy(dl, sid, start, end, proxy_days)
            except Exception as e:
                # 單檔失敗時以 0 視之並繼續
                cov = 0.0
                print(f"[WARN] {sid} 覆蓋率計算失敗：{e}")
            r = info.loc[info["stock_id"] == sid].iloc[0]
            rows.append((sid, r["stock_name"], r["industry_category"], cov))
        print(f"  進度 {min(len(rows), len(sids))}/{len(sids)}")
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    meta = pd.DataFrame(rows, columns=["stock_id", "stock_name", "industry_category", "coverage_ratio"])
    keep = meta[meta["coverage_ratio"] >= float(min_coverage)].sort_values(
        ["coverage_ratio", "stock_id"], ascending=[False, True]
    )
    return keep[["stock_id"]].reset_index(drop=True), keep.reset_index(drop=True)


# -------------- output --------------
def save_to_excel(universe: pd.DataFrame, meta: pd.DataFrame, outpath: Path, title: str):
    """輸出 XLSX（含標題與格式）；若無 xlsxwriter 則退回簡易版。"""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pd.ExcelWriter(outpath, engine="xlsxwriter") as writer:
            ws1, ws2 = "Universe", "Meta"
            wb = writer.book
            fmt_title  = wb.add_format({"bold": True, "font_size": 16})
            fmt_header = wb.add_format({"bold": True, "bg_color": "#E6F0FF", "border": 1})
            fmt_pct    = wb.add_format({"num_format": "0.00%"})

            # Universe
            u = universe.merge(meta[["stock_id", "stock_name"]], on="stock_id", how="left")[["stock_id", "stock_name"]]
            u.to_excel(writer, sheet_name=ws1, index=False, startrow=2)
            s1 = writer.sheets[ws1]
            s1.merge_range(0, 0, 0, u.shape[1]-1, title, fmt_title)
            for c, name in enumerate(u.columns):
                s1.write(2, c, name, fmt_header)
                w = max(12, min(40, int(u[name].astype(str).str.len().max() * 1.2)))
                s1.set_column(c, c, w)
            s1.freeze_panes(3, 0)

            # Meta
            m = meta.copy()
            m.to_excel(writer, sheet_name=ws2, index=False, startrow=2)
            s2 = writer.sheets[ws2]
            s2.merge_range(0, 0, 0, m.shape[1]-1, f"{title} (Meta)", fmt_title)
            for c, name in enumerate(m.columns):
                s2.write(2, c, name, fmt_header)
                if name == "coverage_ratio":
                    s2.set_column(c, c, 14, fmt_pct)
                else:
                    w = max(12, min(40, int(m[name].astype(str).str.len().max() * 1.2)))
                    s2.set_column(c, c, w)
            s2.freeze_panes(3, 0)
        return
    except Exception:
        pass

    # 簡易版
    with pd.ExcelWriter(outpath) as writer:
        universe.merge(meta[["stock_id", "stock_name"]], on="stock_id", how="left") \
                .to_excel(writer, "Universe", index=False)
        meta.to_excel(writer, "Meta", index=False)


def main():
    require_packages()
    dl = login_loader()

    universe, meta = compute_universe(
        dl,
        start=START_DATE,
        end=END_DATE,
        min_coverage=COVERAGE_THRESHOLD,
        limit=None,         # 你可先設 50/100 做小批測試
        sleep_sec=SLEEP_SEC
    )

    outdir = Path("data")
    outdir.mkdir(parents=True, exist_ok=True)

    # CSV
    universe.to_csv(outdir / "twse_universe_2015_2024.csv", index=False, encoding="utf-8")
    meta.to_csv(outdir / "twse_universe_2015_2024_with_meta.csv", index=False, encoding="utf-8")

    # XLSX
    save_to_excel(universe, meta, outdir / "twse_universe_2015_2024.xlsx",
                  title="TWSE Stock Universe (2015–2024) — Proxy by 2330")

    print(f"完成：{len(universe)} 檔，輸出位於 {outdir.resolve()}")


if __name__ == "__main__":
    main()
