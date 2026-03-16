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


def compute_base_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().reset_index(drop=True)
    x["ret1"] = x["close"].pct_change()

    # 趋势主干：所有策略都复用这套短期趋势线 / 多空线。
    x["trend_line"] = x["close"].ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()
    x["ma14"] = x["close"].rolling(14).mean()
    x["ma28"] = x["close"].rolling(28).mean()
    x["ma57"] = x["close"].rolling(57).mean()
    x["ma114"] = x["close"].rolling(114).mean()
    x["long_line"] = (x["ma14"] + x["ma28"] + x["ma57"] + x["ma114"]) / 4.0
    x["trend_ok"] = x["trend_line"] > x["long_line"]

    # KDJ 主干：B2/B3/砖型都直接复用。
    low_9 = x["low"].rolling(9).min()
    high_9 = x["high"].rolling(9).max()
    rsv = (x["close"] - low_9) / (high_9 - low_9 + EPS) * 100
    x["K"] = pd.Series(rsv, index=x.index).ewm(com=2, adjust=False).mean()
    x["D"] = x["K"].ewm(com=2, adjust=False).mean()
    x["J"] = 3 * x["K"] - 2 * x["D"]
    return x
