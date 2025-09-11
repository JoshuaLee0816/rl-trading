import pandas as pd

def add_moving_averages(df: pd.DataFrame, windows=[5, 20, 34, 60]) -> pd.DataFrame:
    """對每一檔股票的 close 欄位計算移動平均 (MA)，並新增欄位。"""
    close_cols = [c for c in df.columns if c.endswith("_close")]
    new_cols = {}
    for col in close_cols:
        sid = col.split("_")[0]
        for w in windows:
            new_cols[f"{sid}_MA{w}"] = df[col].rolling(w).mean().ffill().bfill()
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


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


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """對寬表每檔 <sid>_close 產生 <sid>_rsi{window} 欄。"""
    sids = sorted({c.split("_")[0] for c in df.columns if c.endswith("_close")})
    new_cols = {}
    for sid in sids:
        new_cols[f"{sid}_rsi{window}"] = _wilder_rsi(df[f"{sid}_close"], window=window)
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_volume_moving_averages(df: pd.DataFrame, windows=[20, 60]) -> pd.DataFrame:
    """對每一檔股票的 volume 欄位計算成交量移動平均 (VMA)。"""
    volume_cols = [c for c in df.columns if c.endswith("_volume")]
    new_cols = {}
    for col in volume_cols:
        sid = col.split("_")[0]
        for w in windows:
            new_cols[f"{sid}_VMA{w}"] = df[col].rolling(w).mean().ffill().bfill()
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_macd(df: pd.DataFrame, short=12, long=26, signal=9) -> pd.DataFrame:
    """對每一檔股票的 close 欄位計算 MACD, Signal, Histogram。"""
    close_cols = [c for c in df.columns if c.endswith("_close")]
    new_cols = {}
    for col in close_cols:
        sid = col.split("_")[0]
        ema_short = df[col].ewm(span=short, adjust=False).mean()
        ema_long = df[col].ewm(span=long, adjust=False).mean()
        macd = ema_short - ema_long
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        hist = macd - signal_line
        new_cols[f"{sid}_macd"] = macd
        new_cols[f"{sid}_macd_signal"] = signal_line
        new_cols[f"{sid}_macd_hist"] = hist
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_kd(df: pd.DataFrame, n: int = 9, k_period: int = 3, d_period: int = 3) -> pd.DataFrame:
    """計算 KD 指標 (Stochastic Oscillator)，輸出 K、D。"""
    close_cols = [c for c in df.columns if c.endswith("_close")]
    new_cols = {}
    for col in close_cols:
        sid = col.split("_")[0]
        high_col = f"{sid}_high"
        low_col = f"{sid}_low"
        if high_col not in df.columns or low_col not in df.columns:
            continue

        low_min = df[low_col].rolling(n).min()
        high_max = df[high_col].rolling(n).max()
        rsv = (df[col] - low_min) / (high_max - low_min) * 100

        K = pd.Series(50.0, index=df.index)
        D = pd.Series(50.0, index=df.index)
        for i in range(1, len(df)):
            K.iloc[i] = (1/k_period) * rsv.iloc[i] + (1 - 1/k_period) * K.iloc[i-1]
            D.iloc[i] = (1/d_period) * K.iloc[i] + (1 - 1/d_period) * D.iloc[i-1]

        new_cols[f"{sid}_K"] = K
        new_cols[f"{sid}_D"] = D

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
