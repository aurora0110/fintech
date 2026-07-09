from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import safe_div


def tdx_sma(series: pd.Series, n: int, m: int = 1) -> pd.Series:
    return pd.Series(series, copy=False).ewm(alpha=m / n, adjust=False).mean()


def calc_green_streak(green_flag: np.ndarray) -> np.ndarray:
    out = np.zeros(len(green_flag), dtype=np.int32)
    for i in range(1, len(green_flag)):
        out[i] = out[i - 1] + 1 if green_flag[i] else 0
    return out


def calc_true_streak(flag: np.ndarray) -> np.ndarray:
    out = np.zeros(len(flag), dtype=np.int32)
    for i, value in enumerate(flag):
        out[i] = (out[i - 1] + 1 if i > 0 else 1) if value else 0
    return out


def calc_pre_red_green_structure(red_flag: np.ndarray, green_flag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    green_before_current_red = np.zeros(len(red_flag), dtype=np.int32)
    red_before_pullback = np.zeros(len(red_flag), dtype=np.int32)
    red_streak = calc_true_streak(red_flag)
    for i in range(len(red_flag)):
        if not red_flag[i]:
            continue
        j = i - red_streak[i]
        green_count = 0
        while j >= 0 and green_flag[j]:
            green_count += 1
            j -= 1
        red_count = 0
        while j >= 0 and red_flag[j]:
            red_count += 1
            j -= 1
        green_before_current_red[i] = green_count
        red_before_pullback[i] = red_count
    return green_before_current_red, red_before_pullback


def rolling_max_volume_is_bearish(
    open_price: pd.Series,
    close_price: pd.Series,
    volume: pd.Series,
    window: int = 30,
) -> pd.Series:
    opens = pd.to_numeric(open_price, errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(close_price, errors="coerce").to_numpy(dtype=float)
    vols = pd.to_numeric(volume, errors="coerce").to_numpy(dtype=float)
    bearish = closes < opens
    out = np.zeros(len(vols), dtype=bool)
    for i in range(len(vols)):
        start = max(0, i - window + 1)
        window_vols = vols[start : i + 1]
        finite = np.isfinite(window_vols)
        if not finite.any():
            continue
        max_volume = np.nanmax(window_vols)
        max_positions = np.flatnonzero(finite & (window_vols == max_volume))
        if len(max_positions) == 0:
            continue
        out[i] = bool(np.any(bearish[start + max_positions]))
    return pd.Series(out, index=volume.index)


def calc_brick_values(
    df: pd.DataFrame,
    *,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    n: int = 4,
    m1: int = 6,
    m2: int = 6,
    threshold: float = 4.0,
) -> pd.DataFrame:
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    hhv = high.rolling(n).max()
    llv = low.rolling(n).min()
    den = (hhv - llv).replace(0, np.nan)
    var1 = safe_div(hhv - close, den) * 100 - 90
    var2 = tdx_sma(pd.Series(var1, index=df.index), n, 1) + 100
    var3 = safe_div(close - llv, den) * 100
    var4 = tdx_sma(pd.Series(var3, index=df.index), m1, 1)
    var5 = tdx_sma(var4, m2, 1) + 100
    var6 = var5 - var2
    brick = np.where(var6 > threshold, var6 - threshold, 0.0)
    brick = pd.Series(brick, index=df.index, dtype=float)
    prev = brick.shift(1)
    red_len = np.where(brick > prev, brick - prev, 0.0)
    green_len = np.where(brick < prev, prev - brick, 0.0)
    return pd.DataFrame(
        {
            "brick": brick,
            "brick_prev": prev,
            "brick_red_len": pd.Series(red_len, index=df.index, dtype=float),
            "brick_green_len": pd.Series(green_len, index=df.index, dtype=float),
            "brick_red": pd.Series(red_len > 0, index=df.index),
            "brick_green": pd.Series(green_len > 0, index=df.index),
        }
    )

