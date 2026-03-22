from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils.shared_market_features import EPS, compute_base_features, safe_div


def _rolling_rank_desc(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).apply(
        lambda s: pd.Series(s).rank(method="min", ascending=False).iloc[-1],
        raw=False,
    )


def _build_segment_rise_features(
    close: pd.Series,
    is_up: pd.Series,
    lookback: int,
    allow_one_small_down: bool = True,
) -> pd.DataFrame:
    rise_return = np.full(len(close), np.nan, dtype=float)
    rise_days = np.full(len(close), np.nan, dtype=float)
    rise_slope = np.full(len(close), np.nan, dtype=float)

    close_values = close.astype(float).to_numpy()
    up_values = is_up.fillna(False).to_numpy(dtype=bool)

    for idx in range(len(close_values)):
        left = max(0, idx - lookback + 1)
        best_low = np.nan
        best_high = np.nan
        best_days = 0
        down_allow = 1 if allow_one_small_down else 0

        start = None
        downs = 0
        for j in range(left, idx + 1):
            if up_values[j]:
                if start is None:
                    start = j
                    downs = 0
            else:
                if start is None:
                    continue
                downs += 1
                if downs > down_allow:
                    seg_end = j - 1
                    if seg_end > start:
                        seg_low = float(np.min(close_values[start : seg_end + 1]))
                        seg_high = float(np.max(close_values[start : seg_end + 1]))
                        seg_days = seg_end - start + 1
                        if seg_days > best_days and np.isfinite(seg_low) and seg_low > 0:
                            best_low = seg_low
                            best_high = seg_high
                            best_days = seg_days
                    start = None
                    downs = 0
        if start is not None:
            seg_end = idx
            if seg_end > start:
                seg_low = float(np.min(close_values[start : seg_end + 1]))
                seg_high = float(np.max(close_values[start : seg_end + 1]))
                seg_days = seg_end - start + 1
                if seg_days > best_days and np.isfinite(seg_low) and seg_low > 0:
                    best_low = seg_low
                    best_high = seg_high
                    best_days = seg_days

        if best_days > 0 and np.isfinite(best_low) and best_low > 0 and np.isfinite(best_high):
            seg_ret = best_high / best_low - 1.0
            rise_return[idx] = seg_ret
            rise_days[idx] = float(best_days)
            rise_slope[idx] = seg_ret / best_days

    return pd.DataFrame(
        {
            f"segment_rise_return_{lookback}": rise_return,
            f"segment_rise_days_{lookback}": rise_days,
            f"segment_rise_slope_{lookback}": rise_slope,
        },
        index=close.index,
    )


def add_risk_features(df: pd.DataFrame, precomputed_base: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    统一计算“短期透支 + 前期出货痕迹”风险特征。

    输入要求使用英文字段：
    - open/high/low/close/volume

    输出既保留连续值，也产出第一批二值标签，供各策略做：
    - 硬过滤
    - 扣分
    - 风险提示
    """
    x = precomputed_base.copy() if precomputed_base is not None else compute_base_features(df)
    x = x.copy().reset_index(drop=True)

    close = x["close"].astype(float)
    open_ = x["open"].astype(float)
    high = x["high"].astype(float)
    low = x["low"].astype(float)
    volume = x["volume"].astype(float)
    prev_close = close.shift(1)

    is_up = close > prev_close
    is_bear = close < open_
    body = (close - open_).abs()
    full_range = (high - low).replace(0, np.nan)
    upper_shadow = (high - np.maximum(open_, close)).clip(lower=0)

    for window in (3, 5, 8, 10, 15):
        x[f"ret_{window}"] = close.pct_change(window)

    for window in (5, 8, 10):
        x[f"up_count_{window}"] = is_up.rolling(window, min_periods=window).sum()
        x[f"down_count_{window}"] = (~is_up).rolling(window, min_periods=window).sum()

    up_streak = np.zeros(len(x), dtype=int)
    for idx in range(1, len(x)):
        up_streak[idx] = up_streak[idx - 1] + 1 if bool(is_up.iloc[idx]) else 0
    x["max_up_streak_10"] = pd.Series(up_streak, index=x.index).rolling(10, min_periods=1).max()

    seg10 = _build_segment_rise_features(close, is_up, 10)
    seg15 = _build_segment_rise_features(close, is_up, 15)
    x = pd.concat([x, seg10, seg15], axis=1)

    x["dist_20d_high"] = safe_div(close, high.rolling(20, min_periods=20).max())
    x["dist_30d_high"] = safe_div(close, high.rolling(30, min_periods=30).max())
    x["close_to_trend"] = safe_div(close - x["trend_line"], x["trend_line"])
    x["trend_slope_5"] = safe_div(x["trend_line"], x["trend_line"].shift(5)) - 1.0

    x["volume_rank_20"] = _rolling_rank_desc(volume, 20)
    x["volume_rank_30"] = _rolling_rank_desc(volume, 30)
    x["upper_shadow_body_ratio"] = safe_div(upper_shadow, body)
    x["upper_shadow_range_ratio"] = safe_div(upper_shadow, full_range)
    x["body_ratio"] = safe_div(body, full_range)
    x["close_position"] = safe_div(close - low, full_range)

    x["heavy_bear_top"] = (
        is_bear
        & (x["body_ratio"] >= 0.40)
        & (x["close_position"] <= 0.35)
        & (x["volume_rank_20"] <= 3.0)
        & (x["dist_20d_high"] >= 0.92)
    )
    x["recent_heavy_bear_top_20"] = x["heavy_bear_top"].shift(1).rolling(20, min_periods=1).max().fillna(0).astype(bool)

    prior_high_20 = high.shift(1).rolling(20, min_periods=20).max()
    x["failed_breakout_top"] = (
        (safe_div(high, prior_high_20) >= 0.995)
        & (x["dist_20d_high"] >= 0.92)
        & (x["upper_shadow_range_ratio"] >= 0.30)
        & (x["upper_shadow_body_ratio"] >= 0.50)
        & (x["close_position"] <= 0.70)
        & (x["volume_rank_20"] <= 5.0)
        & (close <= prior_high_20 * 1.005)
    )
    x["recent_failed_breakout_20"] = x["failed_breakout_top"].shift(1).rolling(20, min_periods=1).max().fillna(0).astype(bool)

    bear_volume = np.where(is_bear, volume, 0.0)
    bull_volume = np.where(close > open_, volume, 0.0)
    x["bear_vol_sum_20"] = pd.Series(bear_volume, index=x.index).rolling(20, min_periods=20).sum()
    x["bull_vol_sum_20"] = pd.Series(bull_volume, index=x.index).rolling(20, min_periods=20).sum()
    x["top_distribution_20"] = (
        (x["dist_20d_high"] >= 0.90)
        & (x["bear_vol_sum_20"] > x["bull_vol_sum_20"] * 1.10)
    )

    heavy_bear_start = x["heavy_bear_top"] | (
        is_bear
        & (x["body_ratio"] >= 0.35)
        & (x["volume_rank_20"] <= 3.0)
    )
    stair_bear = np.zeros(len(x), dtype=bool)
    vol_values = volume.to_numpy(dtype=float)
    bear_values = is_bear.to_numpy(dtype=bool)
    for idx in range(3, len(x)):
        start = idx - 3
        if not bool(heavy_bear_start.iloc[start]):
            continue
        if not (bear_values[start + 1] and bear_values[start + 2] and bear_values[start + 3]):
            continue
        if not (vol_values[start] > vol_values[start + 1] > vol_values[start + 2] > vol_values[start + 3]):
            continue
        if not (close.iloc[start] >= close.iloc[start + 1] >= close.iloc[start + 2] >= close.iloc[start + 3]):
            continue
        stair_bear[idx] = True
    x["stair_bear_20"] = pd.Series(stair_bear, index=x.index)
    x["recent_stair_bear_20"] = x["stair_bear_20"].shift(1).rolling(20, min_periods=1).max().fillna(0).astype(bool)

    x["risk_fast_rise_5d_30"] = x["ret_5"] >= 0.30
    x["risk_fast_rise_5d_40"] = x["ret_5"] >= 0.40
    x["risk_fast_rise_10d_40"] = x["ret_10"] >= 0.40
    x["risk_segment_rise_slope_10_006"] = x["segment_rise_slope_10"] >= 0.06
    x["risk_segment_rise_slope_15_005"] = x["segment_rise_slope_15"] >= 0.05
    x["risk_distribution_any_20"] = (
        x["recent_heavy_bear_top_20"]
        | x["recent_failed_breakout_20"]
        | x["top_distribution_20"]
        | x["recent_stair_bear_20"]
    )

    return x


def latest_risk_snapshot(feature_df: pd.DataFrame) -> Dict[str, object]:
    if feature_df is None or feature_df.empty:
        return {}
    latest = feature_df.iloc[-1]
    return {
        "ret_5": float(latest["ret_5"]) if np.isfinite(latest["ret_5"]) else np.nan,
        "ret_10": float(latest["ret_10"]) if np.isfinite(latest["ret_10"]) else np.nan,
        "max_up_streak_10": int(latest["max_up_streak_10"]) if np.isfinite(latest["max_up_streak_10"]) else 0,
        "segment_rise_return_10": float(latest["segment_rise_return_10"]) if np.isfinite(latest["segment_rise_return_10"]) else np.nan,
        "segment_rise_slope_10": float(latest["segment_rise_slope_10"]) if np.isfinite(latest["segment_rise_slope_10"]) else np.nan,
        "recent_heavy_bear_top_20": bool(latest["recent_heavy_bear_top_20"]),
        "recent_failed_breakout_20": bool(latest["recent_failed_breakout_20"]),
        "top_distribution_20": bool(latest["top_distribution_20"]),
        "recent_stair_bear_20": bool(latest["recent_stair_bear_20"]),
        "risk_fast_rise_5d_30": bool(latest["risk_fast_rise_5d_30"]),
        "risk_fast_rise_5d_40": bool(latest["risk_fast_rise_5d_40"]),
        "risk_fast_rise_10d_40": bool(latest["risk_fast_rise_10d_40"]),
        "risk_segment_rise_slope_10_006": bool(latest["risk_segment_rise_slope_10_006"]),
        "risk_segment_rise_slope_15_005": bool(latest["risk_segment_rise_slope_15_005"]),
        "risk_distribution_any_20": bool(latest["risk_distribution_any_20"]),
    }


RISK_LABELS = {
    "recent_heavy_bear_top_20": "近20日高位巨量阴线",
    "recent_failed_breakout_20": "近20日假突破",
    "top_distribution_20": "近20日顶部出货区",
    "recent_stair_bear_20": "近20日阶梯量阴跌",
    "risk_fast_rise_5d_30": "近5日涨超30%",
    "risk_fast_rise_5d_40": "近5日涨超40%",
    "risk_fast_rise_10d_40": "近10日涨超40%",
    "risk_segment_rise_slope_10_006": "近10日阳线段斜率过陡",
    "risk_segment_rise_slope_15_005": "近15日阳线段斜率过陡",
    "risk_distribution_any_20": "近20日出货痕迹",
}


def active_risk_labels(snapshot: Optional[Dict[str, object]]) -> List[str]:
    if not snapshot:
        return []
    labels: List[str] = []
    for key, label in RISK_LABELS.items():
        if bool(snapshot.get(key, False)):
            labels.append(label)
    return labels


def format_risk_note(snapshot: Optional[Dict[str, object]]) -> str:
    labels = active_risk_labels(snapshot)
    if not labels:
        return ""
    return "风险：" + "、".join(labels)
