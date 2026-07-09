from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-12


def safe_div(a, b, default=np.nan):
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    out = np.full(np.shape(a_arr), default, dtype=float)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr) & (np.abs(b_arr) > EPS)
    out[mask] = a_arr[mask] / b_arr[mask]
    return out


def calc_ma(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    min_periods = window if min_periods is None else min_periods
    return pd.Series(series, copy=False).rolling(window=window, min_periods=min_periods).mean()


def calc_ema(series: pd.Series, span: int) -> pd.Series:
    return pd.Series(series, copy=False).ewm(span=span, adjust=False).mean()


def calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    close = pd.Series(close, copy=False)
    dif = calc_ema(close, fast) - calc_ema(close, slow)
    dea = calc_ema(dif, signal)
    macd = 2 * (dif - dea)
    return pd.DataFrame({"dif": dif, "dea": dea, "macd": macd})


def calc_kdj(df: pd.DataFrame, n: int = 9, k_period: int = 3, d_period: int = 3) -> pd.DataFrame:
    low_9 = df["low"].rolling(window=n, min_periods=1).min()
    high_9 = df["high"].rolling(window=n, min_periods=1).max()
    rsv = pd.Series(safe_div(df["close"] - low_9, high_9 - low_9) * 100.0, index=df.index)
    k = rsv.ewm(alpha=1 / k_period, adjust=False).mean()
    d = k.ewm(alpha=1 / d_period, adjust=False).mean()
    j = 3 * k - 2 * d
    return pd.DataFrame({"K": k, "D": d, "J": j})


def calc_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    close = pd.Series(close, copy=False)
    delta = close.diff()
    up = pd.Series(np.where(delta > 0, delta, 0.0), index=close.index)
    down = pd.Series(np.where(delta < 0, -delta, 0.0), index=close.index)
    avg_up = up.ewm(span=window, adjust=False).mean()
    avg_down = down.ewm(span=window, adjust=False).mean()
    rs = pd.Series(np.where(avg_down == 0, 100.0, avg_up / avg_down), index=close.index)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def calc_volume_ma(volume: pd.Series, window: int = 5, min_periods: int | None = None) -> pd.Series:
    return calc_ma(volume, window, min_periods=min_periods)


def calc_price_momentum(close: pd.Series, window: int = 3) -> pd.Series:
    return pd.Series(close, copy=False).pct_change(window)


def calc_zhixing_trend_line(close: pd.Series) -> pd.Series:
    return calc_ema(calc_ema(close, 10), 10)


def calc_zhixing_long_line(close: pd.Series) -> pd.Series:
    ma14 = calc_ma(close, 14)
    ma28 = calc_ma(close, 28)
    ma57 = calc_ma(close, 57)
    ma114 = calc_ma(close, 114)
    return (ma14 + ma28 + ma57 + ma114) / 4.0


def calc_trend_dev(close: pd.Series, trend_line: pd.Series) -> pd.Series:
    return pd.Series(safe_div(close - trend_line, trend_line), index=pd.Index(close.index))


def calc_long_dev(close: pd.Series, long_line: pd.Series) -> pd.Series:
    return pd.Series(safe_div(close - long_line, long_line), index=pd.Index(close.index))


def calc_pin_values(df: pd.DataFrame, n1: int = 3, n2: int = 21) -> pd.DataFrame:
    llv_l_n1 = df["low"].rolling(window=n1).min()
    hhv_c_n1 = df["close"].rolling(window=n1).max()
    llv_l_n2 = df["low"].rolling(window=n2).min()
    hhv_c_n2 = df["close"].rolling(window=n2).max()
    pin_short = pd.Series(safe_div(df["close"] - llv_l_n1, hhv_c_n1 - llv_l_n1) * 100.0, index=df.index)
    pin_long = pd.Series(safe_div(df["close"] - llv_l_n2, hhv_c_n2 - llv_l_n2) * 100.0, index=df.index)
    return pd.DataFrame({"pin_short": pin_short, "pin_long": pin_long, "pin_diff": pin_long - pin_short})

