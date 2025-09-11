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
            out[f"{sid}_MA{w}"] = df[col].rolling(w).mean().ffill().bfill()
    return out


def _wilder_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI (EMA 平滑)。前 window 天為 NaN。"""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    roll_dn = down.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    rs = roll_up / roll_dn
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def add_rsi(df_wide: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """對寬表每檔 <sid>_close 產生 <sid>_rsi{window} 欄。"""
    out = df_wide.copy()
    sids = sorted({c.split("_")[0] for c in out.columns if c.endswith("_close")})
    for sid in sids:
        out[f"{sid}_rsi{window}"] = _wilder_rsi(out[f"{sid}_close"], window=window)
    return out


def add_volume_moving_averages(df: pd.DataFrame, windows=[20, 60]) -> pd.DataFrame:
    """
    對每一檔股票的 volume 欄位計算成交量移動平均 (VMA)，並新增欄位。
    欄位格式: {stock_id}_VMA{window}
    """
    out = df.copy()
    volume_cols = [c for c in df.columns if c.endswith("_volume")]

    for col in volume_cols:
        sid = col.split("_")[0]
        for w in windows:
            out[f"{sid}_VMA{w}"] = df[col].rolling(w).mean().ffill().bfill()
    return out


def add_macd(df: pd.DataFrame, short=12, long=26, signal=9) -> pd.DataFrame:
    """
    對每一檔股票的 close 欄位計算 MACD, Signal, Histogram。
    欄位格式: {stock_id}_macd, {stock_id}_macd_signal, {stock_id}_macd_hist
    """
    out = df.copy()
    close_cols = [c for c in df.columns if c.endswith("_close")]

    for col in close_cols:
        sid = col.split("_")[0]
        ema_short = df[col].ewm(span=short, adjust=False).mean()
        ema_long = df[col].ewm(span=long, adjust=False).mean()
        macd = ema_short - ema_long
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        hist = macd - signal_line

        out[f"{sid}_macd"] = macd
        out[f"{sid}_macd_signal"] = signal_line
        out[f"{sid}_macd_hist"] = hist

    return out
