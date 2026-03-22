from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd

from utils.shared_market_features import compute_base_features, safe_div


MIN_BARS = 120
EPS = 1e-12

# B2 单类型参数
B2_J_MAX = 80.0
B2_UPPER_SHADOW_BODY_RATIO = 1.0 / 3.0

DATE_COL_CANDIDATES = ["date", "Date", "trade_date", "日期", "DATE"]
OPEN_COL_CANDIDATES = ["open", "Open", "开盘", "OPEN"]
HIGH_COL_CANDIDATES = ["high", "High", "最高", "HIGH"]
LOW_COL_CANDIDATES = ["low", "Low", "最低", "LOW"]
CLOSE_COL_CANDIDATES = ["close", "Close", "收盘", "CLOSE"]
VOL_COL_CANDIDATES = ["volume", "vol", "Volume", "成交量", "VOL"]
CODE_COL_CANDIDATES = ["code", "ts_code", "symbol", "代码", "CODE"]

def pick_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"缺少字段，候选字段={candidates}")
    return None


def read_csv_auto(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    return pd.read_csv(path, sep=r"\s+|\t+", engine="python")


def load_one_csv(path: str) -> Optional[pd.DataFrame]:
    raw = read_csv_auto(path)
    date_col = pick_col(raw, DATE_COL_CANDIDATES)
    open_col = pick_col(raw, OPEN_COL_CANDIDATES)
    high_col = pick_col(raw, HIGH_COL_CANDIDATES)
    low_col = pick_col(raw, LOW_COL_CANDIDATES)
    close_col = pick_col(raw, CLOSE_COL_CANDIDATES)
    vol_col = pick_col(raw, VOL_COL_CANDIDATES)
    code_col = pick_col(raw, CODE_COL_CANDIDATES, required=False)

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col], errors="coerce"),
            "open": pd.to_numeric(raw[open_col], errors="coerce"),
            "high": pd.to_numeric(raw[high_col], errors="coerce"),
            "low": pd.to_numeric(raw[low_col], errors="coerce"),
            "close": pd.to_numeric(raw[close_col], errors="coerce"),
            "volume": pd.to_numeric(raw[vol_col], errors="coerce"),
        }
    )
    if code_col:
        df["code"] = raw[code_col].astype(str).iloc[0]
    else:
        df["code"] = os.path.splitext(os.path.basename(path))[0]
    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    df = df[
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
        & (df["volume"] > 0)
    ].copy()
    if len(df) < MIN_BARS:
        return None
    return df


def add_features(
    df: pd.DataFrame,
    precomputed_base: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    x = precomputed_base.copy() if precomputed_base is not None else compute_base_features(df)
    full_range = (x["high"] - x["low"]).replace(0, np.nan)
    x["close_position"] = pd.Series(safe_div(x["close"] - x["low"], full_range), index=x.index)
    x["j_rank20"] = x["J"].rolling(20, min_periods=20).apply(
        lambda win: pd.Series(win).rank(pct=True).iloc[-1],
        raw=False,
    )
    x["j_rank20_prev"] = x["j_rank20"].shift(1)

    x["vol_ma5"] = x["volume"].rolling(5).mean()
    x["signal_vs_ma5"] = pd.Series(safe_div(x["volume"], x["vol_ma5"]), index=x.index)

    real_body = (x["close"] - x["open"]).abs()
    upper_shadow = x["high"] - np.maximum(x["open"], x["close"])
    x["small_upper_shadow"] = (real_body <= EPS) | (
        upper_shadow <= real_body * B2_UPPER_SHADOW_BODY_RATIO + EPS
    )

    x["b2_volume_ok"] = x["volume"] > x["volume"].shift(1)

    x["b2_j_ok"] = (
        (x["J"] < B2_J_MAX)
        & (x["J"] > x["J"].shift(1))
        & (
            (x["J"].shift(1) < x["J"].shift(2))
            | (
                (x["J"].shift(2) < x["J"].shift(1) * 0.8)
                & (x["J"].shift(3) > x["J"].shift(2))
            )
        )
    )

    x["b2_signal"] = (
        x["trend_ok"]
        & (x["close"] > x["open"])
        & (x["ret1"] >= 0.04)
        & x["small_upper_shadow"]
        & x["b2_volume_ok"]
        & x["b2_j_ok"]
    )

    x["any_type"] = x["b2_signal"]

    trend_slope5 = pd.Series(
        safe_div(x["trend_line"], x["trend_line"].shift(5)),
        index=x.index,
    ) - 1.0
    trend_spread = pd.Series(
        safe_div(x["trend_line"] - x["long_line"], x["close"]),
        index=x.index,
    ).fillna(0.0)
    volume_quality = (1.0 - np.minimum(np.abs(x["signal_vs_ma5"].fillna(0.0) - 1.9) / 0.6, 1.0)).clip(lower=0.0)
    close_quality = x["close_position"].fillna(0.0).clip(lower=0.0, upper=1.0)
    j_headroom = (1.0 - np.minimum(x["J"].fillna(200.0) / B2_J_MAX, 1.0)).clip(lower=0.0)
    x["sort_score"] = (
        0.35 * close_quality
        + 0.20 * volume_quality
        + 0.20 * trend_spread
        + 0.10 * j_headroom
        + 0.10 * np.maximum(trend_slope5.fillna(0.0), 0.0)
        + 0.05 * x["b2_signal"].astype(float)
    )
    return x


def check(file_path, hold_list=None, feature_cache=None):
    if feature_cache is not None:
        x = feature_cache.b2_features()
        if x is None or x.empty:
            return [-1]
    else:
        df = load_one_csv(str(file_path))
        if df is None or df.empty:
            return [-1]
        x = add_features(df)
    latest = x.iloc[-1]
    if not bool(latest["b2_signal"]):
        return [-1]

    stop_loss_price = round(float(latest["low"]), 3)
    return [
        1,
        stop_loss_price,
        float(latest["close"]),
        round(float(latest["sort_score"]), 4),
        "B2",
    ]
