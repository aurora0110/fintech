from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd

from utils import brick_filter


MIN_BARS = 120
EPS = 1e-12
COUNTERPARTY_LOOKBACK = 60
COUNTERPARTY_CONFIRM_DAYS = 15

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


def pick_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
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
        & (df["volume"] >= 0)
    ].copy()
    if len(df) < MIN_BARS:
        return None
    return df


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a.astype(float) / b.astype(float).replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
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
    rsv = _safe_div(x["close"] - low_9, high_9 - low_9) * 100.0
    x["K"] = rsv.ewm(com=2, adjust=False).mean()
    x["D"] = x["K"].ewm(com=2, adjust=False).mean()
    x["J"] = 3 * x["K"] - 2 * x["D"]
    x["j_rank20"] = x["J"].rolling(20, min_periods=20).apply(
        lambda win: pd.Series(win).rank(pct=True).iloc[-1], raw=False
    )
    x["j_rank20_prev"] = x["j_rank20"].shift(1)

    x["vol_ma5"] = x["volume"].rolling(5).mean()
    x["vol_ma20"] = x["volume"].rolling(20).mean()
    x["signal_vs_ma5"] = _safe_div(x["volume"], x["vol_ma5"])
    x["signal_vs_ma20"] = _safe_div(x["volume"], x["vol_ma20"])
    x["vol_vs_prev"] = _safe_div(x["volume"], x["volume"].shift(1))

    full_range = (x["high"] - x["low"]).replace(0.0, np.nan)
    real_body = (x["close"] - x["open"]).abs()
    upper_shadow = x["high"] - np.maximum(x["open"], x["close"])
    lower_shadow = np.minimum(x["open"], x["close"]) - x["low"]
    x["body_ratio"] = _safe_div(real_body, full_range).clip(lower=0.0)
    x["upper_range_ratio"] = _safe_div(upper_shadow, full_range).clip(lower=0.0)
    x["lower_range_ratio"] = _safe_div(lower_shadow, full_range).clip(lower=0.0)
    x["close_position"] = _safe_div(x["close"] - x["low"], full_range).clip(lower=0.0, upper=1.0)

    x["ret5"] = _safe_div(x["close"], x["close"].shift(5)) - 1.0
    x["ret10"] = _safe_div(x["close"], x["close"].shift(10)) - 1.0
    x["ret20"] = _safe_div(x["close"], x["close"].shift(20)) - 1.0
    x["ret5_prev"] = _safe_div(x["close"].shift(1), x["close"].shift(6)) - 1.0
    x["ret10_prev"] = _safe_div(x["close"].shift(1), x["close"].shift(11)) - 1.0
    x["ret20_prev"] = _safe_div(x["close"].shift(1), x["close"].shift(21)) - 1.0
    x["trend_slope_5"] = _safe_div(x["trend_line"], x["trend_line"].shift(5)) - 1.0
    x["trend_slope_10"] = _safe_div(x["trend_line"], x["trend_line"].shift(10)) - 1.0
    x["trend_slope_5_prev"] = _safe_div(x["trend_line"].shift(1), x["trend_line"].shift(6)) - 1.0
    x["trend_slope_10_prev"] = _safe_div(x["trend_line"].shift(1), x["trend_line"].shift(11)) - 1.0
    x["high30_prev"] = x["high"].shift(1).rolling(30).max()
    x["high60_prev"] = x["high"].shift(1).rolling(60).max()
    x["close20_max_prev"] = x["close"].shift(1).rolling(20).max()
    x["vol_rank30"] = x["volume"].rolling(30, min_periods=30).apply(
        lambda v: pd.Series(v).rank(pct=True).iloc[-1], raw=False
    )
    x["vol_rank60"] = x["volume"].rolling(60, min_periods=60).apply(
        lambda v: pd.Series(v).rank(pct=True).iloc[-1], raw=False
    )
    x["prev60_vol_max"] = x["volume"].shift(1).rolling(60, min_periods=20).max()
    x["near_20d_low"] = _safe_div(x["close"], x["low"].rolling(20).min())
    x["near_20d_high"] = _safe_div(x["close"], x["high"].rolling(20).max())
    return x


def add_distribution_labels(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["is_bear"] = x["close"] < x["open"]
    x["is_bull"] = x["close"] > x["open"]
    x["bear_vol"] = np.where(x["is_bear"], x["volume"], 0.0)
    x["bull_vol"] = np.where(x["is_bull"], x["volume"], 0.0)
    x["top_zone_now"] = (
        (x["high"] >= x["high30_prev"] * 0.97)
        | (x["high"] >= x["high60_prev"] * 0.95)
        | (x["close"] >= x["close20_max_prev"] * 0.94)
    )
    x["top_zone_recent"] = pd.Series(x["top_zone_now"].fillna(False)).rolling(12, min_periods=1).max().astype(bool)

    point_accel_heavy_bear = (
        x["is_bear"]
        & (x["body_ratio"] >= 0.45)
        & (x["close_position"] <= 0.36)
        & ((x["vol_rank30"] >= 0.90) | (x["vol_rank60"] >= 0.93) | (x["volume"] >= x["vol_ma20"] * 1.15))
        & (
            (x["ret5_prev"] >= 0.08)
            | (x["ret10_prev"] >= 0.12)
            | (x["ret20_prev"] >= 0.20)
            | (x["trend_slope_5_prev"] >= 0.03)
            | (x["trend_slope_10_prev"] >= 0.06)
        )
        & x["top_zone_now"]
    )
    point_failed_breakout = (
        (x["high"] >= x["high30_prev"] * 0.995)
        & (x["close"] <= x["high30_prev"] * 0.985)
        & (x["upper_range_ratio"] >= 0.28)
        & (x["volume"] >= x["vol_ma20"] * 1.2)
        & x["top_zone_now"]
    )
    x["point_accel_heavy_bear"] = point_accel_heavy_bear.fillna(False)
    x["point_failed_breakout"] = point_failed_breakout.fillna(False)
    x["point_any"] = x["point_accel_heavy_bear"] | x["point_failed_breakout"]

    x["bear_vol_sum10"] = pd.Series(x["bear_vol"]).rolling(10).sum()
    x["bull_vol_sum10"] = pd.Series(x["bull_vol"]).rolling(10).sum()
    x["bear_days_10"] = pd.Series(x["is_bear"].astype(int)).rolling(10).sum()
    zone_top_distribution = (
        (x["bear_days_10"] >= 5)
        & (x["bear_vol_sum10"] >= x["bull_vol_sum10"] * 1.20)
        & x["top_zone_recent"]
    )

    had_new_high_recent = pd.Series(x["top_zone_now"].shift(1).fillna(False)).rolling(8, min_periods=1).max().astype(bool)
    recent_bear_4 = pd.Series(x["is_bear"].astype(int)).rolling(4).sum() >= 2
    zone_post_new_high_selloff = (
        had_new_high_recent.fillna(False)
        & recent_bear_4.fillna(False)
        & (x["close"] < x["close20_max_prev"] * 0.96)
        & (x["bear_vol_sum10"] >= x["bull_vol_sum10"])
    )

    stair_active = pd.Series(False, index=x.index)
    point_arr = x["point_any"].fillna(False).to_numpy(dtype=bool)
    for i in range(len(x)):
        left = max(0, i - 5)
        point_idx = None
        for j in range(i - 1, left - 1, -1):
            if point_arr[j]:
                point_idx = j
                break
        if point_idx is None or i - point_idx < 2:
            continue
        sub = x.iloc[point_idx + 1 : i + 1]
        if len(sub) < 2:
            continue
        if not bool((sub["close"] < sub["open"]).all()):
            continue
        if not bool((sub["volume"].diff().fillna(0) <= 0).iloc[1:].all()):
            continue
        if not bool((sub["close"].diff().fillna(0) <= 0).iloc[1:].all()):
            continue
        stair_active.iat[i] = True

    x["zone_top_distribution"] = zone_top_distribution.fillna(False)
    x["zone_post_new_high_selloff"] = zone_post_new_high_selloff.fillna(False)
    x["zone_stair_bear"] = stair_active.fillna(False)
    x["zone_any"] = x["zone_top_distribution"] | x["zone_post_new_high_selloff"] | x["zone_stair_bear"]

    # 区间终点只在高位/中高位仍然成立时保留，避免把已经跌到低位的末端错误当成卖点。
    still_top_zone = x["top_zone_now"] | (x["near_20d_high"] >= 0.92)
    x["zone_start"] = x["zone_any"] & (~x["zone_any"].shift(1).fillna(False))
    x["zone_end"] = x["zone_any"] & (~x["zone_any"].shift(-1).fillna(False)) & still_top_zone.fillna(False)
    return x


def add_counterparty_breakdown_labels(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    anchor_df = brick_filter.last_double_bull_anchor(x, lookback=90)
    x = pd.concat([x, anchor_df], axis=1)

    x["keyk_break_close"] = x["has_double_bull_anchor"].fillna(False) & (x["close"] < x["double_bull_close"])
    x["keyk_break_low"] = x["has_double_bull_anchor"].fillna(False) & (x["close"] < x["double_bull_low"])

    close_minus_long = x["close"] - x["long_line"]
    sign_now = np.sign(close_minus_long)
    sign_prev = np.sign(close_minus_long.shift(1))
    x["long_cross_count_10"] = ((sign_now * sign_prev) < 0).rolling(10, min_periods=3).sum()
    x["near_long_10_ratio"] = (_safe_div((x["close"] - x["long_line"]).abs(), x["close"]) <= 0.03).rolling(10, min_periods=3).mean()
    x["oscillate_near_long"] = (
        (x["long_cross_count_10"] >= 2)
        & (x["near_long_10_ratio"] >= 0.50)
    )

    x["recent_point_any_60"] = x["point_any"].shift(1).rolling(COUNTERPARTY_LOOKBACK, min_periods=1).max().fillna(False).astype(bool)
    x["recent_accel_heavy_bear_90"] = (
        x["point_accel_heavy_bear"].shift(1).rolling(90, min_periods=1).max().fillna(False).astype(bool)
    )
    x["recent_zone_any_30"] = x["zone_any"].shift(1).rolling(30, min_periods=1).max().fillna(False).astype(bool)

    break_line_or_key = (
        (x["close"] < x["long_line"])
        | (x["close"] < x["trend_line"])
        | x["keyk_break_close"]
        | x["keyk_break_low"]
    )
    candidate = (
        (x["trend_line"] >= x["long_line"])
        & ~x["point_any"].fillna(False)
        & (x["ret1"] <= -0.02)
        & (x["close_position"] <= 0.30)
        & (x["volume"] >= x["volume"].shift(1) * 0.90)
        & (x["volume"] <= x["prev60_vol_max"] * 0.70)
        & break_line_or_key
    )

    quick_reclaim = np.zeros(len(x), dtype=bool)
    sideways_confirm = np.zeros(len(x), dtype=bool)
    confirmed = np.zeros(len(x), dtype=bool)
    quick_reclaim_strict = np.zeros(len(x), dtype=bool)
    sideways_confirm_strict = np.zeros(len(x), dtype=bool)
    exempt = np.zeros(len(x), dtype=bool)
    confirmation_date = np.full(len(x), np.datetime64("NaT"), dtype="datetime64[ns]")
    repeat_recent = np.zeros(len(x), dtype=bool)
    recent_candidate_30 = candidate.shift(1).rolling(30, min_periods=1).max().fillna(False).astype(bool)

    last_confirm_idx = -10_000

    for i in range(len(x)):
        if i - last_confirm_idx <= 20:
            repeat_recent[i] = True
        if not bool(candidate.iat[i]):
            continue
        if repeat_recent[i]:
            continue

        event_low = float(x.at[i, "low"])
        if not np.isfinite(event_low):
            continue

        future = x.iloc[i + 1 : i + 1 + COUNTERPARTY_CONFIRM_DAYS]
        if future.empty:
            continue

        event_close = float(x.at[i, "close"])
        event_trend = float(x.at[i, "trend_line"])
        event_long = float(x.at[i, "long_line"])

        # 路径1：次日/两日内迅速站回多空线之上，不给低位筹码。
        qr = future.iloc[:2]
        qr_hits = qr[qr["close"] > qr["long_line"]]
        if not qr_hits.empty:
            quick_reclaim[i] = True
            confirmed[i] = True
            confirmation_date[i] = qr_hits.iloc[0]["date"]
            last_confirm_idx = i
        # 路径2：最长15天缩量横盘洗盘后再确认回收。
        broke_event_low = bool((future["close"] < event_low).any())
        reclaim = future[future["close"] > future["long_line"]]
        if (not broke_event_low) and (not reclaim.empty):
            sideways_confirm[i] = True
            confirmed[i] = True
            confirmation_date[i] = reclaim.iloc[0]["date"]
            last_confirm_idx = i

        # 保守版卖出豁免标签：
        # 1) 排除当前已经属于区间出货 / 多空线附近来回震荡的样本
        # 2) 允许两条确认路径：
        #    a. 2日内先收回（哪怕未完全站回多空线）
        #    b. 15日内不再有效破低，且价格有足够回收
        if bool(x.at[i, "point_any"] or x.at[i, "zone_any"] or x.at[i, "oscillate_near_long"]):
            continue

        qr_strict = qr[
            (qr["close"] >= event_close)
            & (qr["close"] >= qr["open"])
        ]
        if not qr_strict.empty:
            quick_reclaim_strict[i] = True
            exempt[i] = True
            continue

        max_close_15 = float(future["close"].max()) if not future.empty else np.nan
        sideways_strict = (
            (not broke_event_low)
            and np.isfinite(max_close_15)
            and (max_close_15 >= event_close * 1.05)
        )
        if sideways_strict:
            sideways_confirm_strict[i] = True
            exempt[i] = True

    x["counterparty_candidate"] = candidate.fillna(False)
    x["counterparty_break_line_or_key"] = break_line_or_key.fillna(False)
    x["counterparty_quick_reclaim"] = quick_reclaim
    x["counterparty_sideways_confirm"] = sideways_confirm
    x["counterparty_confirmed"] = confirmed
    x["counterparty_quick_reclaim_strict"] = quick_reclaim_strict
    x["counterparty_sideways_confirm_strict"] = sideways_confirm_strict
    x["counterparty_exempt"] = exempt
    x["counterparty_confirmation_date"] = pd.to_datetime(confirmation_date)
    x["counterparty_repeat_recent"] = repeat_recent
    x["counterparty_recent_point_any_60"] = x["recent_point_any_60"]
    x["counterparty_recent_accel_heavy_bear_90"] = x["recent_accel_heavy_bear_90"]
    x["counterparty_oscillate_near_long"] = x["oscillate_near_long"]
    x["counterparty_recent_candidate_30"] = recent_candidate_30
    return x


def add_all_structure_labels(df: pd.DataFrame) -> pd.DataFrame:
    x = add_base_features(df)
    x = add_distribution_labels(x)
    x = add_counterparty_breakdown_labels(x)
    return x
