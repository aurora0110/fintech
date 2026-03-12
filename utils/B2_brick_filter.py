from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd


MIN_BARS = 180
EPS = 1e-12

DATE_COL_CANDIDATES = ["date", "Date", "trade_date", "日期", "DATE"]
OPEN_COL_CANDIDATES = ["open", "Open", "开盘", "OPEN"]
HIGH_COL_CANDIDATES = ["high", "High", "最高", "HIGH"]
LOW_COL_CANDIDATES = ["low", "Low", "最低", "LOW"]
CLOSE_COL_CANDIDATES = ["close", "Close", "收盘", "CLOSE"]
VOL_COL_CANDIDATES = ["volume", "vol", "Volume", "成交量", "VOL"]
CODE_COL_CANDIDATES = ["code", "ts_code", "symbol", "代码", "CODE"]


def safe_div(a, b, default=np.nan):
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    out = np.full(np.shape(a_arr), default, dtype=float)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr) & (np.abs(b_arr) > EPS)
    out[mask] = a_arr[mask] / b_arr[mask]
    return out


def tdx_sma(series: pd.Series, n: int, m: int) -> pd.Series:
    return series.ewm(alpha=m / n, adjust=False).mean()


def calc_green_streak(green_flag: np.ndarray) -> np.ndarray:
    out = np.zeros(len(green_flag), dtype=np.int32)
    for i in range(1, len(green_flag)):
        out[i] = out[i - 1] + 1 if green_flag[i] else 0
    return out


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
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0) & (df["volume"] > 0)].copy()
    if len(df) < MIN_BARS:
        return None
    return df


def board_gain_limit(code: str) -> float:
    if code.startswith("SH#688"):
        return 0.08
    if code.startswith("SZ#300"):
        return 0.08
    return 0.07


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().reset_index(drop=True)
    x["ret1"] = x["close"].pct_change()
    x["trend_line"] = x["close"].ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()
    x["ma14"] = x["close"].rolling(14).mean()
    x["ma28"] = x["close"].rolling(28).mean()
    x["ma57"] = x["close"].rolling(57).mean()
    x["ma114"] = x["close"].rolling(114).mean()
    x["long_line"] = (x["ma14"] + x["ma28"] + x["ma57"] + x["ma114"]) / 4.0
    x["trend_ok"] = x["trend_line"] > x["long_line"]

    low_9 = x["low"].rolling(9).min()
    high_9 = x["high"].rolling(9).max()
    rsv = (x["close"] - low_9) / (high_9 - low_9 + 1e-9) * 100
    x["K"] = pd.Series(rsv, index=x.index).ewm(com=2, adjust=False).mean()
    x["D"] = x["K"].ewm(com=2, adjust=False).mean()
    x["J"] = 3 * x["K"] - 2 * x["D"]

    x["close_pullback_white"] = x["close"] < x["trend_line"] * 1.01
    x["close_above_white"] = x["close"] > x["trend_line"]
    x["vol_ma5_prev"] = x["volume"].shift(1).rolling(5).mean()
    x["signal_vs_ma5"] = pd.Series(safe_div(x["volume"], x["vol_ma5_prev"]), index=x.index)

    hhv4 = x["high"].rolling(4).max()
    llv4 = x["low"].rolling(4).min()
    den4 = (hhv4 - llv4).replace(0, np.nan)
    var1a = safe_div((hhv4 - x["close"]), den4) * 100 - 90
    var2a = tdx_sma(pd.Series(var1a, index=x.index), 4, 1) + 100
    var3a = safe_div((x["close"] - llv4), den4) * 100
    var4a = tdx_sma(pd.Series(var3a, index=x.index), 6, 1)
    var5a = tdx_sma(var4a, 6, 1) + 100
    var6a = var5a - var2a
    x["brick"] = np.where(var6a > 4, var6a - 4, 0.0)
    x["brick_prev"] = x["brick"].shift(1)
    x["brick_red_len"] = np.where(x["brick"] > x["brick_prev"], x["brick"] - x["brick_prev"], 0.0)
    x["brick_green_len"] = np.where(x["brick"] < x["brick_prev"], x["brick_prev"] - x["brick"], 0.0)
    x["brick_red"] = x["brick_red_len"] > 0
    x["brick_green"] = x["brick_green_len"] > 0
    x["prev_green_streak"] = pd.Series(calc_green_streak(x["brick_green"].to_numpy()), index=x.index).shift(1)
    x["close_slope_10"] = x["close"].rolling(10).apply(
        lambda s: np.polyfit(np.arange(len(s)), s, 1)[0] if np.isfinite(s).all() else np.nan,
        raw=False,
    )
    x["not_sideways"] = np.abs(pd.Series(safe_div(x["close_slope_10"], x["close"].rolling(10).mean()), index=x.index)) > 0.002
    x["up_leg_avg_vol"] = x["volume"].shift(4).rolling(3).mean()
    x["pullback_avg_vol"] = x["volume"].shift(1).rolling(3).mean()
    x["pullback_shrinking"] = x["pullback_avg_vol"] < x["up_leg_avg_vol"]
    x["pullback_shrink_ratio"] = pd.Series(safe_div(x["pullback_avg_vol"], x["up_leg_avg_vol"]), index=x.index)

    x["pattern_a"] = (
        (x["prev_green_streak"] >= 3)
        & x["brick_red"]
        & x["close_pullback_white"].shift(1).fillna(False)
        & x["close_above_white"]
    )
    x["pattern_b"] = (
        (pd.Series(calc_green_streak(x["brick_green"].to_numpy()), index=x.index).shift(3) >= 3)
        & x["brick_red"]
        & x["brick_green"].shift(1).fillna(False)
        & x["brick_red"].shift(2).fillna(False)
        & x["close_pullback_white"].shift(1).fillna(False)
        & x["close_above_white"]
    )
    x["rebound_ratio"] = pd.Series(safe_div(x["brick_red_len"], x["brick_green_len"].shift(1)), index=x.index)

    real_body = (x["close"] - x["open"]).abs()
    upper_shadow = x["high"] - np.maximum(x["open"], x["close"])
    x["b2_small_shadow"] = (real_body <= EPS) | (upper_shadow <= real_body * 0.3 + EPS)
    x["b2_volume_ok"] = (x["volume"] > x["volume"].shift(1)) & (x["volume"] > x["volume"].rolling(5).mean())
    x["b2_j_ok"] = (
        (x["J"] < 80)
        & (x["J"] > x["J"].shift(1))
        & (x["J"].shift(1) < x["J"].shift(2))
        & (x["J"].shift(2) < x["J"].shift(3))
    )
    x["b2_trend_start"] = (x["close"].shift(1) <= x["trend_line"].shift(1) * 1.01) & (x["close"] > x["trend_line"])
    x["b2_long_start"] = (x["close"].shift(1) <= x["long_line"].shift(1) * 1.01) & (x["close"] > x["long_line"])
    x["b2_dual_start"] = x["b2_trend_start"] & x["b2_long_start"]

    rebound_rank_raw = x["rebound_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    trend_spread = pd.Series(np.maximum((x["trend_line"] - x["long_line"]) / x["close"], 0.0), index=x.index).fillna(0.0)
    shrink_quality = (1.0 - np.minimum(np.abs(x["pullback_shrink_ratio"].fillna(9.0) - 0.8) / 0.3, 1.0)).clip(lower=0.0)
    b2_line_quality = pd.Series(np.maximum((x["close"] - np.maximum(x["trend_line"], x["long_line"])) / x["close"], 0.0), index=x.index).fillna(0.0)
    x["sort_score"] = 0.50 * shrink_quality + 0.30 * rebound_rank_raw + 0.20 * trend_spread + 0.20 * b2_line_quality
    return x


def check(file_path, hold_list=None):
    df = load_one_csv(str(file_path))
    if df is None or df.empty:
        return [-1]
    x = add_features(df)
    latest_idx = len(x) - 1
    if latest_idx < 0:
        return [-1]
    latest = x.iloc[latest_idx]
    code = str(latest["code"])
    gain_limit = board_gain_limit(code)

    brick_ok = (
        bool(latest["trend_ok"])
        and bool(latest["not_sideways"])
        and bool(latest["pullback_shrinking"])
        and float(latest["ret1"]) <= gain_limit
        and float(latest["signal_vs_ma5"]) >= 1.2
        and float(latest["signal_vs_ma5"]) <= 2.5
        and (
            (bool(latest["pattern_a"]) and float(latest["rebound_ratio"]) >= 1.0)
            or (bool(latest["pattern_b"]) and float(latest["rebound_ratio"]) >= 1.0)
        )
    )
    b2_ok = (
        bool(latest["trend_ok"])
        and bool(latest["b2_dual_start"])
        and bool(latest["b2_small_shadow"])
        and bool(latest["b2_volume_ok"])
        and bool(latest["b2_j_ok"])
        and float(latest["ret1"]) >= 0.03
    )
    if not (brick_ok and b2_ok):
        return [-1]

    stop_loss_price = round(float(latest["low"]) * 0.99, 3)
    return [1, stop_loss_price, float(latest["close"]), round(float(latest["sort_score"]), 4), "brick+B2回调启动"]
