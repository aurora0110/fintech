from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils.shared_market_features import compute_base_features


INPUT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/data/20260226/normal")
MIN_BARS = 160
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
        if df is not None and df.shape[1] > 1:
            return df
    except Exception:
        pass
    try:
        return pd.read_csv(path, sep=r"\s+|\t+", engine="python")
    except Exception:
        return pd.DataFrame()


def load_one_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        raw = read_csv_auto(path)
        if raw is None or raw.empty:
            return None
    except Exception:
        return None
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
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0) & (df["volume"] >= 0)].copy()
    if len(df) < MIN_BARS:
        return None
    return df


def calc_green_streak(green_flag: np.ndarray) -> np.ndarray:
    out = np.zeros(len(green_flag), dtype=np.int32)
    for i in range(1, len(green_flag)):
        out[i] = out[i - 1] + 1 if green_flag[i] else 0
    return out


def triangle_quality(value: float, center: float, half_width: float) -> float:
    if not np.isfinite(value) or half_width <= 0:
        return 0.0
    return float(max(1.0 - abs(value - center) / half_width, 0.0))


def clip01(series: pd.Series) -> pd.Series:
    return series.clip(lower=0.0, upper=1.0)


def rolling_last_percentile(series: pd.Series, window: int) -> pd.Series:
    values = series.astype(float)

    def _pct_last(arr):
        arr = np.asarray(arr, dtype=float)
        if len(arr) == 0 or not np.isfinite(arr[-1]):
            return np.nan
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            return np.nan
        return float(np.sum(valid <= arr[-1]) / len(valid))

    return values.rolling(window, min_periods=window).apply(_pct_last, raw=True)


def identify_low_zones(mask_series: pd.Series) -> List[tuple]:
    mask = mask_series.fillna(False).to_numpy(dtype=bool)
    zones: List[tuple] = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            zones.append((start, i - 1))
            start = None
    if start is not None:
        zones.append((start, len(mask) - 1))
    return zones


def build_n_up_feature(df: pd.DataFrame, rank_col: str, rank_threshold: float, lookback: int = 80) -> pd.Series:
    out = np.zeros(len(df), dtype=bool)
    lows = df["low"].astype(float).to_numpy()
    highs = df["high"].astype(float).to_numpy()
    closes = df["close"].astype(float).to_numpy()
    rank_values = df[rank_col].astype(float)

    for idx in range(len(df)):
        left = max(0, idx - lookback + 1)
        sub_rank = rank_values.iloc[left : idx + 1].reset_index(drop=True)
        zones = identify_low_zones(sub_rank <= rank_threshold)
        if len(zones) < 2:
            continue
        z1, z2 = zones[-2], zones[-1]
        z1_start, z1_end = left + z1[0], left + z1[1]
        z2_start, z2_end = left + z2[0], left + z2[1]
        first_low = float(np.min(lows[z1_start : z1_end + 1]))
        second_low = float(np.min(lows[z2_start : z2_end + 1]))
        if not (second_low > first_low):
            continue
        mid_left = z1_end + 1
        mid_right = z2_start - 1
        if mid_right < mid_left:
            continue
        rebound_high = float(np.max(highs[mid_left : mid_right + 1]))
        if closes[idx] > rebound_high:
            out[idx] = True

    return pd.Series(out, index=df.index)


def build_stair_volume_flag(x: pd.DataFrame, lookback: int = 30) -> pd.Series:
    out = np.zeros(len(x), dtype=bool)
    closes = x["close"].to_numpy(dtype=float)
    opens = x["open"].to_numpy(dtype=float)
    vols = x["volume"].to_numpy(dtype=float)

    for idx in range(len(x)):
        left = max(0, idx - lookback + 1)
        hit = False
        for anchor in range(left, idx - 2):
            if not (closes[anchor] > opens[anchor]):
                continue
            window_left = max(0, anchor - lookback + 1)
            anchor_hist = vols[window_left : anchor + 1]
            if len(anchor_hist) < 5:
                continue
            if np.sum(anchor_hist >= vols[anchor]) > 3:
                continue
            for stair_len in (2, 3, 4):
                end = anchor + stair_len
                if end >= idx:
                    continue
                ok = True
                prev_vol = vols[anchor]
                for j in range(anchor + 1, end + 1):
                    if not (closes[j] < opens[j]):
                        ok = False
                        break
                    if not (vols[j] < prev_vol and vols[j] <= prev_vol * 0.95 and vols[j] >= prev_vol * 0.45):
                        ok = False
                        break
                    prev_vol = vols[j]
                if ok:
                    hit = True
                    break
            if hit:
                break
        out[idx] = hit
    return pd.Series(out, index=x.index)


def binary_score_for_index(x: pd.DataFrame, signal_idx: int) -> dict:
    i = int(signal_idx)
    row = x.loc[i]

    prev5 = x.loc[max(0, i - 5) : i - 1]
    prev5_ret = safe_div(
        np.array([x.loc[i - 1, "close"] if i >= 5 else np.nan], dtype=float),
        np.array([x.loc[max(0, i - 5), "close"] if i >= 5 else np.nan], dtype=float),
    )[0]
    prev5_ret = prev5_ret - 1.0 if np.isfinite(prev5_ret) else np.nan
    down_days_5 = int((prev5["ret1"] < 0).sum()) if len(prev5) else 0

    near_prev20_high_ratio = float(row["near_prev20_high_ratio"]) if pd.notna(row["near_prev20_high_ratio"]) else np.nan

    red_hist = x.loc[max(0, i - 20) : i, "brick_red_len"]
    red_hist = red_hist[red_hist > 0]
    red_rank20 = float((red_hist <= row["brick_red_len"]).mean()) if len(red_hist) else np.nan

    prev_green_len = float(x["brick_green_len"].shift(1).iloc[i]) if pd.notna(x["brick_green_len"].shift(1).iloc[i]) else np.nan
    green_hist = x.loc[max(0, i - 20) : i - 1, "brick_green_len"] if i >= 1 else pd.Series(dtype=float)
    green_hist = green_hist[green_hist > 0]
    prev_green_rank20 = (
        float((green_hist <= prev_green_len).mean())
        if len(green_hist) and np.isfinite(prev_green_len)
        else np.nan
    )

    positive_flags = {
        "中等放量": bool(pd.notna(row["signal_vs_ma5"]) and 1.0 <= float(row["signal_vs_ma5"]) <= 2.0),
        "2倍量": bool(row["double_prev_volume"]),
        "前5日回调": bool(pd.notna(prev5_ret) and prev5_ret <= -0.01),
        "5日3阴": bool(down_days_5 >= 3),
        "中高位活动区": bool(pd.notna(near_prev20_high_ratio) and 0.90 <= near_prev20_high_ratio <= 0.98),
        "N型或沿趋势": bool(row["recent_n_up_30d"] or row["along_trend_rise"]),
        "红砖偏大": bool(pd.notna(red_rank20) and red_rank20 >= 0.60),
        "连续绿砖": bool(pd.notna(row["prev_green_streak"]) and float(row["prev_green_streak"]) >= 3),
        "强反包": bool(pd.notna(row["rebound_ratio"]) and float(row["rebound_ratio"]) >= 1.2),
        "关键K支撑": bool(row["keyk_support_active"] or row["above_double_bull_close"]),
    }
    negative_flags = {
        "近30天阶梯量": bool(row["stair_volume_30d"]),
        "过近突破": bool(pd.notna(near_prev20_high_ratio) and near_prev20_high_ratio >= 0.98 and row["intraday_break20"]),
        "位置偏低": bool(pd.notna(near_prev20_high_ratio) and near_prev20_high_ratio <= 0.88),
        "假突破长上影": bool(row["failed_break20_long_upper"]),
        "前绿过大反包不足": bool(
            pd.notna(prev_green_rank20)
            and prev_green_rank20 >= 0.60
            and (pd.isna(row["rebound_ratio"]) or float(row["rebound_ratio"]) < 1.2)
        ),
        "过度放量": bool(pd.notna(row["signal_vs_ma5"]) and float(row["signal_vs_ma5"]) > 2.5),
    }

    positive_score = int(sum(positive_flags.values()))
    negative_score = int(sum(negative_flags.values()))
    sort_score = positive_score - negative_score
    plus_parts = [k for k, v in positive_flags.items() if v]
    minus_parts = [k for k, v in negative_flags.items() if v]
    positive_detail = ",".join(plus_parts) if plus_parts else "无"
    negative_detail = ",".join(minus_parts) if minus_parts else "无"
    detail = f"+{positive_detail}|-{negative_detail}"

    return {
        "positive_score": positive_score,
        "negative_score": negative_score,
        "sort_score": sort_score,
        "positive_detail": positive_detail,
        "negative_detail": negative_detail,
        "score_detail": detail,
        "red_rank20": red_rank20,
        "prev_green_rank20": prev_green_rank20,
        "prev5_ret": prev5_ret,
        "down_days_5": down_days_5,
        "near_prev20_high_ratio": near_prev20_high_ratio,
    }


def last_double_bull_anchor(df: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    high_anchor = np.full(len(df), np.nan, dtype=float)
    low_anchor = np.full(len(df), np.nan, dtype=float)
    close_anchor = np.full(len(df), np.nan, dtype=float)
    has_anchor = np.zeros(len(df), dtype=bool)
    is_candidate = np.zeros(len(df), dtype=bool)

    last_high = np.nan
    last_low = np.nan
    last_close = np.nan
    last_idx = -10_000

    bull = df["close"] > df["open"]
    prev_vol = df["volume"].shift(1)
    vol_ratio_prev = safe_div(df["volume"], prev_vol)
    vol_rank30 = (
        df["volume"]
        .rolling(30, min_periods=1)
        .apply(lambda s: pd.Series(s).rank(method="min", ascending=False).iloc[-1], raw=False)
    )
    vol_rank60 = (
        df["volume"]
        .rolling(60, min_periods=1)
        .apply(lambda s: pd.Series(s).rank(method="min", ascending=False).iloc[-1], raw=False)
    )
    cross_up = (df["trend_line"] > df["long_line"]) & (df["trend_line"].shift(1) <= df["long_line"].shift(1))
    future_cross_15 = np.zeros(len(df), dtype=bool)
    cross_values = cross_up.fillna(False).to_numpy(dtype=bool)
    for i in range(len(df)):
        future_cross_15[i] = cross_values[i : min(i + 16, len(df))].any()

    # 倍量柱的交易语义：
    # 1. 必须是明显突兀的放量阳柱，至少相对前一日翻倍；
    # 2. 量级需要进入近30/60日最突出的两根之一；
    # 3. 只在底部启动区才有意义，因此要求快慢线接近，
    #    且该柱后15日内能看到趋势线上穿多空线的启动证据。
    bottom_zone_now = (
        (df["trend_line"] <= df["long_line"] * 1.03)
        & (safe_div(df["long_line"] - df["trend_line"], df["close"]) < 0.08)
    )
    double_bull = (
        bull
        & (vol_ratio_prev >= 2.0)
        & ((vol_rank30 <= 2.0) | (vol_rank60 <= 2.0))
        & bottom_zone_now.fillna(False)
        & pd.Series(future_cross_15, index=df.index)
    )

    for i in range(len(df)):
        if i - last_idx > lookback:
            last_high = np.nan
            last_low = np.nan
            last_close = np.nan
            last_idx = -10_000
        high_anchor[i] = last_high
        low_anchor[i] = last_low
        close_anchor[i] = last_close
        has_anchor[i] = np.isfinite(last_close)
        if bool(double_bull.iloc[i]):
            is_candidate[i] = True
            last_high = float(df.at[i, "high"])
            last_low = float(df.at[i, "low"])
            last_close = float(df.at[i, "close"])
            last_idx = i

    return pd.DataFrame(
        {
            "has_double_bull_anchor": has_anchor,
            "double_bull_high": high_anchor,
            "double_bull_low": low_anchor,
            "double_bull_close": close_anchor,
            "double_bull_candidate": is_candidate,
            "double_bull_vol_ratio_prev": vol_ratio_prev,
            "double_bull_vol_rank30": vol_rank30,
            "double_bull_vol_rank60": vol_rank60,
            "double_bull_bottom_zone": bottom_zone_now,
        },
        index=df.index,
    )


def derive_keyk_states(df: pd.DataFrame) -> pd.DataFrame:
    prev_volume = df["volume"].shift(1).fillna(0.0)
    has_anchor = df["has_double_bull_anchor"].fillna(False)

    close_key = df["double_bull_close"]
    high_key = df["double_bull_high"]

    support_touch_close = has_anchor & (df["low"] <= close_key * 1.01) & (df["close"] >= close_key)
    support_touch_high = has_anchor & (df["low"] <= high_key * 1.01) & (df["close"] >= high_key)
    pressure_touch_close = has_anchor & (df["high"] >= close_key * 0.995) & (df["close"] <= close_key)
    pressure_touch_high = has_anchor & (df["high"] >= high_key * 0.995) & (df["close"] <= high_key)

    # 支撑/压制失效：放量有效跌破/突破后未立即收回。
    support_invalid_close = has_anchor & (df["close"] < close_key * 0.99) & (df["volume"] > prev_volume)
    support_invalid_high = has_anchor & (df["close"] < high_key * 0.99) & (df["volume"] > prev_volume)
    pressure_invalid_close = has_anchor & (df["close"] > close_key * 1.01) & (df["volume"] > prev_volume)
    pressure_invalid_high = has_anchor & (df["close"] > high_key * 1.01) & (df["volume"] > prev_volume)

    support_active = (support_touch_close | support_touch_high) & ~(support_invalid_close | support_invalid_high)
    pressure_active = (pressure_touch_close | pressure_touch_high) & ~(pressure_invalid_close | pressure_invalid_high)

    return pd.DataFrame(
        {
            "keyk_support_touch_close": support_touch_close,
            "keyk_support_touch_high": support_touch_high,
            "keyk_pressure_touch_close": pressure_touch_close,
            "keyk_pressure_touch_high": pressure_touch_high,
            "keyk_support_invalid_close": support_invalid_close,
            "keyk_support_invalid_high": support_invalid_high,
            "keyk_pressure_invalid_close": pressure_invalid_close,
            "keyk_pressure_invalid_high": pressure_invalid_high,
            "keyk_support_active": support_active,
            "keyk_pressure_active": pressure_active,
        },
        index=df.index,
    )


def add_features(df: pd.DataFrame, precomputed_base: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    x = precomputed_base.copy() if precomputed_base is not None else compute_base_features(df)
    x["prev5_ret"] = safe_div(x["close"].shift(1), x["close"].shift(5)) - 1.0
    x["close_pullback_white"] = x["close"] < x["trend_line"] * 1.02
    x["close_above_white"] = x["close"] > x["trend_line"]
    x["bull_close"] = x["close"] > x["open"]

    x["vol_ma5_prev"] = x["volume"].shift(1).rolling(5).mean()
    x["vol_ma10_prev"] = x["volume"].shift(1).rolling(10).mean()
    x["vol_vs_prev"] = safe_div(x["volume"], x["volume"].shift(1))
    x["double_prev_volume"] = x["vol_vs_prev"] >= 2.0
    x["signal_vs_ma5"] = safe_div(x["volume"], x["vol_ma5_prev"])
    x["signal_vs_ma10"] = safe_div(x["volume"], x["vol_ma10_prev"])
    x["signal_vs_ma5_valid"] = x["signal_vs_ma5"].le(3.0)
    # 在保留大多数成功案例的前提下增加一层硬筛选：
    # 要么前 5 日已经出现回调，要么当天量能不要过度偏大。
    x["retain_case_priority_ok"] = (
        (x["prev5_ret"] <= -0.01)
        | (x["signal_vs_ma5"] <= 2.0)
    )

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
    x["red_vs_prev_green_ok"] = x["brick_red_len"] > x["brick_green_len"].shift(1) * 0.66
    x["pattern_two_green_one_red"] = (
        x["brick_green"].shift(2).fillna(False)
        & x["brick_green"].shift(1).fillna(False)
        & x["brick_red"].fillna(False)
    )
    x["pattern_one_green_one_red_one_green_one_red"] = (
        x["brick_green"].shift(3).fillna(False)
        & x["brick_red"].shift(2).fillna(False)
        & x["brick_green"].shift(1).fillna(False)
        & x["brick_red"].fillna(False)
    )
    # 完美图成功案例里，“两绿两红”只出现过 1 个样本，
    # 该样本前一日红砖长度 / 当日红砖长度约为 0.0645。
    # 这里先把“前一日红必须很小”定成 <= 0.10，作为保守近似。
    x["pattern_two_green_two_red_small_prev_red"] = (
        x["brick_green"].shift(3).fillna(False)
        & x["brick_green"].shift(2).fillna(False)
        & x["brick_red"].shift(1).fillna(False)
        & x["brick_red"].fillna(False)
        & (x["brick_red_len"].shift(1) <= x["brick_red_len"] * 0.10)
    )
    x["sequence_pattern_ok"] = (
        x["pattern_two_green_one_red"]
        | x["pattern_one_green_one_red_one_green_one_red"]
        | x["pattern_two_green_two_red_small_prev_red"]
    )

    x["close_slope_10"] = (
        x["close"]
        .rolling(10)
        .apply(lambda s: np.polyfit(np.arange(len(s)), s, 1)[0] if np.isfinite(s).all() else np.nan, raw=False)
    )
    x["not_sideways"] = np.abs(safe_div(x["close_slope_10"], x["close"].rolling(10).mean())) > 0.001

    x["up_leg_avg_vol"] = x["volume"].shift(4).rolling(3).mean()
    x["pullback_avg_vol"] = x["volume"].shift(1).rolling(3).mean()
    prev5_avg = x["volume"].shift(1).rolling(5).mean()
    prev10_avg = x["volume"].shift(1).rolling(10).mean()
    x["t1_vs_prev5"] = safe_div(x["volume"].shift(1), prev5_avg)
    x["pullback_vs_prev10"] = safe_div(x["pullback_avg_vol"], prev10_avg)
    x["pullback_shrinking"] = x["pullback_avg_vol"] < x["up_leg_avg_vol"]

    x["trend_slope_3"] = safe_div(x["trend_line"], x["trend_line"].shift(3)) - 1.0
    x["trend_slope_5"] = safe_div(x["trend_line"], x["trend_line"].shift(5)) - 1.0
    x["long_slope_5"] = safe_div(x["long_line"], x["long_line"].shift(5)) - 1.0
    x["dist_trend"] = safe_div(x["close"] - x["trend_line"], x["close"])
    x["dist_long"] = safe_div(x["close"] - x["long_line"], x["close"])

    x["j_rank_20"] = rolling_last_percentile(x["J"], 20)
    x["j_rank_30"] = rolling_last_percentile(x["J"], 30)
    x["n_up_rank20_p10"] = build_n_up_feature(x, "j_rank_20", 0.10)
    x["n_up_rank30_p10"] = build_n_up_feature(x, "j_rank_30", 0.10)
    x["n_up_any"] = x["n_up_rank20_p10"] | x["n_up_rank30_p10"]
    x["recent_n_up_30d"] = (
        pd.Series(x["n_up_any"].fillna(False)).shift(1).rolling(30, min_periods=1).max().fillna(0.0).astype(bool)
    )
    x["stair_volume_30d"] = build_stair_volume_flag(x, lookback=30)
    x["high20_prev"] = x["high"].shift(1).rolling(20).max()
    x["high60_prev"] = x["high"].shift(1).rolling(60).max()
    x["near_prev20_high_ratio"] = safe_div(x["close"], x["high20_prev"])
    x["near_prev60_high_ratio"] = safe_div(x["close"], x["high60_prev"])
    x["intraday_break20"] = (x["high"] > x["high20_prev"]).fillna(False)
    x["close_break20"] = (x["close"] > x["high20_prev"]).fillna(False)

    candle_range = (x["high"] - x["low"]).replace(0, np.nan)
    body = (x["close"] - x["open"]).abs()
    x["body_ratio"] = safe_div(body, candle_range)
    x["close_position"] = safe_div(x["close"] - x["low"], candle_range)
    x["upper_shadow_ratio"] = safe_div(x["high"] - np.maximum(x["open"], x["close"]), candle_range)
    x["lower_shadow_ratio"] = safe_div(np.minimum(x["open"], x["close"]) - x["low"], candle_range)
    x["failed_break20_long_upper"] = (
        (x["high"] >= x["high20_prev"] * 0.995)
        & (x["close"] < x["high20_prev"] * 0.995)
        & (x["upper_shadow_ratio"] >= (1.0 / 3.0))
    ).fillna(False)
    x["along_trend_rise"] = (
        x["close"].gt(x["trend_line"])
        & x["low"].between(x["trend_line"] * 0.98, x["trend_line"] * 1.02)
    )

    anchor_df = last_double_bull_anchor(x, lookback=60)
    x = pd.concat([x, anchor_df], axis=1)
    keyk_state_df = derive_keyk_states(x)
    x = pd.concat([x, keyk_state_df], axis=1)
    x["above_double_bull_high"] = x["has_double_bull_anchor"] & (x["close"] > x["double_bull_high"])
    x["above_double_bull_low"] = x["has_double_bull_anchor"] & (x["close"] > x["double_bull_low"])
    x["above_double_bull_close"] = x["has_double_bull_anchor"] & (x["close"] > x["double_bull_close"])
    x["dist_double_bull_high"] = safe_div(x["close"] - x["double_bull_high"], x["close"])
    x["dist_double_bull_close"] = safe_div(x["close"] - x["double_bull_close"], x["close"])

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

    x["rebound_ratio"] = safe_div(x["brick_red_len"], x["brick_green_len"].shift(1))
    # 当前砖型主策略采用案例重训练后的序列版硬过滤：
    # 1. 必须属于以下三种砖色结构之一：
    #    - 两绿一红
    #    - 一绿一红一绿一红
    #    - 两绿两红（且前一日红砖必须很小）
    # 2. 保留红砖、阳线、站上趋势线、趋势线高于多空线、量比五日上限、当日涨幅上限；
    # 3. 保留最小反包要求：当天红砖长度 > 前一天绿砖长度 * 0.66。
    # 4. 在不损失当前成功案例覆盖的前提下，加一层保留优先硬筛选：
    #    前 5 日有回调，或当日量比五日 <= 2.0。
    x["signal_base"] = (
        x["brick_red"].fillna(False)
        & x["sequence_pattern_ok"].fillna(False)
        & x["close_above_white"].fillna(False)
        & x["bull_close"].fillna(False)
        & x["red_vs_prev_green_ok"].fillna(False)
        & x["signal_vs_ma5_valid"].fillna(False)
        & x["retain_case_priority_ok"].fillna(False)
        & x["ret1"].notna()
    )

    x["signal_vs_ma5_quality"] = x["signal_vs_ma5"].apply(lambda v: triangle_quality(v, 1.25, 0.55))
    x["signal_vs_ma10_quality"] = x["signal_vs_ma10"].apply(lambda v: triangle_quality(v, 1.0, 0.45))
    x["vol_vs_prev_quality"] = x["vol_vs_prev"].apply(lambda v: triangle_quality(v, 1.25, 0.50))
    x["shrink_quality"] = (
        0.50 * x["t1_vs_prev5"].apply(lambda v: triangle_quality(v, 0.95, 0.25))
        + 0.50 * x["pullback_vs_prev10"].apply(lambda v: triangle_quality(v, 0.75, 0.30))
    )
    x["candle_quality"] = (
        0.35 * clip01(x["body_ratio"])
        + 0.35 * clip01(x["close_position"])
        + 0.20 * clip01(1.0 - x["upper_shadow_ratio"])
        + 0.10 * clip01(1.0 - x["lower_shadow_ratio"])
    )
    x["keyk_quality"] = (
        0.30 * x["above_double_bull_high"].astype(float)
        + 0.20 * x["above_double_bull_close"].astype(float)
        + 0.10 * x["above_double_bull_low"].astype(float)
        + 0.20 * x["keyk_support_active"].astype(float)
        + 0.15 * (1.0 - x["keyk_pressure_active"].astype(float))
        + 0.05 * x["has_double_bull_anchor"].astype(float)
    )
    x["volume_quality"] = (
        0.45 * x["signal_vs_ma5_quality"]
        + 0.20 * x["signal_vs_ma10_quality"]
        + 0.20 * x["vol_vs_prev_quality"]
        + 0.15 * x["shrink_quality"]
    )
    return x


def build_signal_df(input_dir: Path) -> pd.DataFrame:
    rows: List[dict] = []
    files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {".csv", ".txt"}])
    total = len(files)
    for idx, path in enumerate(files, 1):
        df = load_one_csv(str(path))
        if df is None or df.empty:
            continue
        code = str(df["code"].iloc[0])
        x = add_features(df)
        mask = (
            x["signal_base"]
            & (x["ret1"] <= 0.10)
            & (x["trend_line"] > x["long_line"])
        )
        signal_idxs = np.flatnonzero(mask.to_numpy())
        for signal_idx in signal_idxs:
            close = float(x.at[int(signal_idx), "close"])
            trend_line = float(x.at[int(signal_idx), "trend_line"])
            long_line = float(x.at[int(signal_idx), "long_line"])
            ret1 = float(x.at[int(signal_idx), "ret1"])
            signal_vs_ma5 = float(x.at[int(signal_idx), "signal_vs_ma5"])
            rebound_ratio = float(x.at[int(signal_idx), "rebound_ratio"])
            binary_score = binary_score_for_index(x, int(signal_idx))
            pullback_avg_vol = float(x.at[int(signal_idx), "pullback_avg_vol"])
            up_leg_avg_vol = float(x.at[int(signal_idx), "up_leg_avg_vol"])
            pullback_shrink_ratio = pullback_avg_vol / up_leg_avg_vol if np.isfinite(up_leg_avg_vol) and up_leg_avg_vol > 0 else np.nan
            trend_spread_clip = max((trend_line - long_line) / close, 0.0) if np.isfinite(close) and close > 0 else 0.0
            rows.append(
                {
                    "date": x.at[int(signal_idx), "date"],
                    "code": code,
                    "signal_close": close,
                    "signal_open": float(x.at[int(signal_idx), "open"]),
                    "signal_high": float(x.at[int(signal_idx), "high"]),
                    "signal_low": float(x.at[int(signal_idx), "low"]),
                    "signal_volume": float(x.at[int(signal_idx), "volume"]),
                    "rebound_ratio": rebound_ratio,
                    "pullback_shrink_ratio": pullback_shrink_ratio,
                    "trend_spread_clip": trend_spread_clip,
                    "signal_vs_ma5": signal_vs_ma5,
                    "signal_vs_ma10": float(x.at[int(signal_idx), "signal_vs_ma10"]),
                    "vol_vs_prev": float(x.at[int(signal_idx), "vol_vs_prev"]),
                    "double_prev_volume": bool(x.at[int(signal_idx), "double_prev_volume"]),
                    "near_prev20_high_ratio": binary_score["near_prev20_high_ratio"],
                    "prev5_ret": binary_score["prev5_ret"],
                    "down_days_5": binary_score["down_days_5"],
                    "red_rank20": binary_score["red_rank20"],
                    "prev_green_rank20": binary_score["prev_green_rank20"],
                    "recent_n_up_30d": bool(x.at[int(signal_idx), "recent_n_up_30d"]),
                    "stair_volume_30d": bool(x.at[int(signal_idx), "stair_volume_30d"]),
                    "intraday_break20": bool(x.at[int(signal_idx), "intraday_break20"]),
                    "close_break20": bool(x.at[int(signal_idx), "close_break20"]),
                    "failed_break20_long_upper": bool(x.at[int(signal_idx), "failed_break20_long_upper"]),
                    "ret1_quality": triangle_quality(ret1, 0.03, 0.03),
                    "signal_vs_ma5_quality": float(x.at[int(signal_idx), "signal_vs_ma5_quality"]),
                    "signal_vs_ma10_quality": float(x.at[int(signal_idx), "signal_vs_ma10_quality"]),
                    "vol_vs_prev_quality": float(x.at[int(signal_idx), "vol_vs_prev_quality"]),
                    "shrink_quality": float(x.at[int(signal_idx), "shrink_quality"]),
                    "candle_quality": float(x.at[int(signal_idx), "candle_quality"]),
                    "volume_quality": float(x.at[int(signal_idx), "volume_quality"]),
                    "keyk_quality": float(x.at[int(signal_idx), "keyk_quality"]),
                    "keyk_support_active": bool(x.at[int(signal_idx), "keyk_support_active"]),
                    "keyk_pressure_active": bool(x.at[int(signal_idx), "keyk_pressure_active"]),
                    "body_ratio": float(x.at[int(signal_idx), "body_ratio"]),
                    "close_position": float(x.at[int(signal_idx), "close_position"]),
                    "upper_shadow_ratio": float(x.at[int(signal_idx), "upper_shadow_ratio"]),
                    "lower_shadow_ratio": float(x.at[int(signal_idx), "lower_shadow_ratio"]),
                    "trend_slope_3": float(x.at[int(signal_idx), "trend_slope_3"]),
                    "trend_slope_5": float(x.at[int(signal_idx), "trend_slope_5"]),
                    "long_slope_5": float(x.at[int(signal_idx), "long_slope_5"]),
                    "along_trend_rise": bool(x.at[int(signal_idx), "along_trend_rise"]),
                    "has_double_bull_anchor": bool(x.at[int(signal_idx), "has_double_bull_anchor"]),
                    "above_double_bull_high": bool(x.at[int(signal_idx), "above_double_bull_high"]),
                    "above_double_bull_close": bool(x.at[int(signal_idx), "above_double_bull_close"]),
                    "above_double_bull_low": bool(x.at[int(signal_idx), "above_double_bull_low"]),
                    "dist_double_bull_high": float(x.at[int(signal_idx), "dist_double_bull_high"]),
                    "dist_double_bull_close": float(x.at[int(signal_idx), "dist_double_bull_close"]),
                    "pattern_a": bool(x.at[int(signal_idx), "pattern_a"]),
                    "pattern_b": bool(x.at[int(signal_idx), "pattern_b"]),
                    "positive_score": binary_score["positive_score"],
                    "negative_score": binary_score["negative_score"],
                    "score_detail": binary_score["score_detail"],
                    "sort_score": binary_score["sort_score"],
                }
            )
        if idx % 500 == 0 or idx == total:
            print(f"特征进度: {idx}/{total}")

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["date", "code"]).reset_index(drop=True)
    out["score_pct_rank"] = out.groupby("date")["sort_score"].rank(pct=True)
    out["daily_rank"] = (
        out.sort_values(
            ["date", "sort_score", "positive_score", "rebound_ratio", "red_rank20", "code"],
            ascending=[True, False, False, False, False, True],
        )
        .groupby("date")
        .cumcount()
        + 1
    )
    return out


def apply_selection(signal_df: pd.DataFrame) -> pd.DataFrame:
    if signal_df.empty:
        return signal_df
    return signal_df.sort_values(
        ["date", "sort_score", "positive_score", "rebound_ratio", "red_rank20", "code"],
        ascending=[True, False, False, False, False, True],
    ).reset_index(drop=True)


def check(file_path, hold_list=None, feature_cache=None):
    if feature_cache is not None:
        x = feature_cache.brick_features()
        if x is None or x.empty:
            return [-1]
    else:
        df = load_one_csv(str(file_path))
        if df is None or df.empty:
            return [-1]
        x = add_features(df)
    latest_idx = len(x) - 1
    if latest_idx < 0:
        return [-1]
    latest = x.iloc[latest_idx]
    signal_ok = (
        bool(latest["signal_base"])
        and float(latest["ret1"]) <= 0.10
        and float(latest["trend_line"]) > float(latest["long_line"])
    )
    if not signal_ok:
        return [-1]
    binary_score = binary_score_for_index(x, latest_idx)
    sort_score = int(binary_score["sort_score"])
    stop_loss_price = round(float(latest["low"]) * 0.99, 3)
    reason = f"brick二值排序({sort_score:+d})"
    return [
        1,
        stop_loss_price,
        float(latest["close"]),
        int(sort_score),
        int(binary_score["positive_score"]),
        int(binary_score["negative_score"]),
        str(binary_score["positive_detail"]),
        str(binary_score["negative_detail"]),
        reason,
    ]


def main() -> None:
    signal_df = build_signal_df(INPUT_DIR)
    selected_df = apply_selection(signal_df)
    if selected_df.empty:
        return
    latest_trade_date = pd.to_datetime(selected_df["date"]).max()
    latest_df = selected_df[pd.to_datetime(selected_df["date"]) == latest_trade_date].copy()
    print(
        latest_df[
            [
                "date",
                "code",
                "daily_rank",
                "sort_score",
                "positive_score",
                "negative_score",
                "score_detail",
                "rebound_ratio",
                "signal_vs_ma5",
                "double_prev_volume",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
