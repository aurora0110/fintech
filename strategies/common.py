from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def calc_trend_cn(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    short = close.ewm(span=10, adjust=False).mean()
    short = short.ewm(span=10, adjust=False).mean()
    ma14 = close.rolling(window=14).mean()
    ma28 = close.rolling(window=28).mean()
    ma57 = close.rolling(window=57).mean()
    ma114 = close.rolling(window=114).mean()
    long_line = (ma14 + ma28 + ma57 + ma114) / 4.0
    return short, long_line


def calc_kdj(high: pd.Series, low: pd.Series, close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    low_9 = low.rolling(window=9, min_periods=1).min()
    high_9 = high.rolling(window=9, min_periods=1).max()
    rsv = (close - low_9) / (high_9 - low_9 + 1e-6) * 100
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def tongdaxin_sma(series: pd.Series, n: int, m: int = 1) -> pd.Series:
    result = np.zeros(len(series))
    prev_sma = 0.0
    for i in range(len(series)):
        val = float(series.iloc[i])
        if i < n - 1:
            sma = float(series.iloc[: i + 1].sum() / (i + 1))
        else:
            sma = (val * m + prev_sma * (n - m)) / n
        result[i] = sma
        prev_sma = sma
    return pd.Series(result, index=series.index)


def calc_brick_signal(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["HHV_H4"] = out["high"].rolling(window=4).max()
    out["LLV_L4"] = out["low"].rolling(window=4).min()
    out["VAR1A"] = (out["HHV_H4"] - out["close"]) / (out["HHV_H4"] - out["LLV_L4"]) * 100 - 90
    out["VAR1A"] = out["VAR1A"].replace([np.inf, -np.inf], np.nan).fillna(0)
    out["VAR2A"] = tongdaxin_sma(out["VAR1A"], 4, 1) + 100
    out["VAR3A"] = (out["close"] - out["LLV_L4"]) / (out["HHV_H4"] - out["LLV_L4"]) * 100
    out["VAR3A"] = out["VAR3A"].replace([np.inf, -np.inf], np.nan).fillna(0)
    out["VAR4A"] = tongdaxin_sma(out["VAR3A"], 6, 1)
    out["VAR5A"] = tongdaxin_sma(out["VAR4A"], 6, 1) + 100
    out["VAR6A"] = out["VAR5A"] - out["VAR2A"]
    out["brick_value"] = np.where(out["VAR6A"] > 4, out["VAR6A"] - 4, 0)
    out["brick_change"] = out["brick_value"] - out["brick_value"].shift(1)
    out["body_len"] = out["brick_change"].abs()
    out["prev_body_len"] = out["body_len"].shift(1)
    out["red_today"] = out["brick_change"] > 0
    out["green_prev"] = out["brick_change"].shift(1) < 0
    out["brick_buy_signal"] = (
        out["green_prev"].eq(True)
        & out["red_today"].eq(True)
        & (out["body_len"] >= out["prev_body_len"] * 0.66)
    )
    return out.dropna().copy()


def calc_pin_buy_signal(df: pd.DataFrame) -> pd.DataFrame:
    # The current PIN legacy file uses the same "买入信号" definition path as the brick signal path.
    out = calc_brick_signal(df)
    out["pin_buy_signal"] = out["brick_buy_signal"]
    return out


def add_prev_close(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["prev_close"] = out["close"].shift(1)
    return out


def base_prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["short_trend"], out["long_trend"] = calc_trend_cn(out["close"])
    out["K"], out["D"], out["J"] = calc_kdj(out["high"], out["low"], out["close"])
    out = add_prev_close(out)
    return out
