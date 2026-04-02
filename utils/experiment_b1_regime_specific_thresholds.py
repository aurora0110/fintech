#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
个股分状态卖点阈值研究（修正高速版）

主要特性：
1. 修正 min_valid_threshold 细扫逻辑
2. future metrics 使用 sliding_window_view 向量化
3. regime validation 使用向量化
4. threshold scan 使用“排序 + 前缀/后缀和”加速
5. 统一清洗 numeric / inf / nan
6. 输出原始/有效状态分布
7. 保留你当前实验框架和研究定义
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from numpy.lib.stride_tricks import sliding_window_view
    HAS_SLIDING_WINDOW = True
except ImportError:
    HAS_SLIDING_WINDOW = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from utils import stoploss


# =========================
# 配置
# =========================
DATA_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/data/20260324")
OUTPUT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/results/b1_regime_specific_thresholds_fast")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_HISTORY_BARS = 160
HORIZONS = [3, 5, 10]

MIN_REGIME_SAMPLE_COUNT = 80
MIN_THRESHOLD_SAMPLE_COUNT = 20
MIN_EVENT_RATE_INCREASE = 0.05

COARSE_THRESHOLDS = np.arange(1.0, 21.0, 1.0)
FINE_STEP = 0.5

REGIME_FEATURE_COLS = [
    "f_trend_gt_long",
    "f_ma20_gt_ma60",
    "f_close_gt_trend",
    "f_trend_slope_5",
    "f_long_slope_10",
    "f_ma20_slope_5",
    "f_spread_trend_long",
    "f_spread_ma20_ma60",
    "f_avg_spread_trend_long_20",
    "f_avg_spread_ma20_ma60_20",
    "f_cross_count_trend_long_20",
    "f_cross_count_ma20_ma60_20",
    "f_close_cross_trend_count_20",
    "f_trend_positive_20",
    "f_long_positive_20",
    "regime",
]

DEBUG_EXTRA_COLS = [
    "trend_line",
    "long_line",
    "ma20",
    "ma60",
    "dev_trend",
    "dev_long",
]

REGIME_VALIDATION_METRIC_COLS = [
    "future_max_rally",
    "future_close_ret",
    "future_max_dd",
    "dd_gt_3",
    "dd_gt_5",
]


# =========================
# 通用工具
# =========================

def safe_to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """指定列转 numeric，并替换 inf 为 nan。"""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    existing = [c for c in cols if c in df.columns]
    if existing:
        df[existing] = df[existing].replace([np.inf, -np.inf], np.nan)
    return df


def safe_numeric_array(arr: np.ndarray) -> np.ndarray:
    """数组级清洗；强制返回可写副本。"""
    arr = np.array(arr, dtype=float, copy=True)
    arr[~np.isfinite(arr)] = np.nan
    return arr


def compute_slope_ratio(arr: np.ndarray, lag: int) -> np.ndarray:
    """arr / arr.shift(lag) - 1 的 numpy 版。"""
    arr = safe_numeric_array(arr)
    out = np.full(len(arr), np.nan, dtype=float)
    if len(arr) <= lag:
        return out

    prev = arr[:-lag]
    curr = arr[lag:]
    valid = (~np.isnan(curr)) & (~np.isnan(prev)) & (prev != 0)
    out[lag:][valid] = curr[valid] / prev[valid] - 1.0
    return out


def compute_positive_count(arr: np.ndarray, window: int) -> np.ndarray:
    """最近 window 天 diff > 0 的天数。"""
    arr = safe_numeric_array(arr)
    diff = np.diff(arr, prepend=np.nan)
    pos = (diff > 0).astype(float)
    return pd.Series(pos).rolling(window=window, min_periods=window).sum().to_numpy()


def rolling_count_sign_flips(diff: np.ndarray, window: int, eps: float = 1e-8) -> np.ndarray:
    """
    统计滚动窗口内符号翻转次数。
    增加 eps 容忍带，减少贴线抖动误判。
    """
    diff = safe_numeric_array(diff)
    n = len(diff)
    out = np.full(n, np.nan, dtype=float)

    if n < window:
        return out

    sign = np.where(diff > eps, 1.0, np.where(diff < -eps, -1.0, np.nan))
    sign = pd.Series(sign).ffill().bfill().to_numpy()

    if np.isnan(sign).all():
        return out

    flips = np.zeros(n, dtype=float)
    valid_pair = (~np.isnan(sign[1:])) & (~np.isnan(sign[:-1]))
    flips[1:] = np.where(valid_pair & (sign[1:] * sign[:-1] < 0), 1.0, 0.0)

    flips_roll = pd.Series(flips).rolling(window=window, min_periods=window).sum().to_numpy()
    out[window - 1:] = flips_roll[window - 1:]
    return out


# =========================
# 数据加载
# =========================

def load_stock_data(file_path: str) -> pd.DataFrame | None:
    """加载股票数据。"""
    df, load_error = stoploss.load_data(file_path)
    if load_error or df is None or len(df) < MIN_HISTORY_BARS:
        return None

    df = df.sort_values("日期").reset_index(drop=True)

    numeric_cols = ["开盘", "最高", "最低", "收盘"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["收盘", "最高", "最低"]).copy()

    # 过滤明显异常价格
    df = df[(df["收盘"] > 0) & (df["最高"] > 0) & (df["最低"] > 0)].copy()

    if len(df) < MIN_HISTORY_BARS:
        return None

    return df.reset_index(drop=True)


# =========================
# 指标与特征
# =========================

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["收盘"].to_numpy(dtype=float)

    df["trend_line"] = (
        pd.Series(close)
        .ewm(span=10, adjust=False).mean()
        .ewm(span=10, adjust=False).mean()
        .to_numpy()
    )

    df["ma14"] = pd.Series(close).rolling(window=14, min_periods=14).mean().to_numpy()
    df["ma28"] = pd.Series(close).rolling(window=28, min_periods=28).mean().to_numpy()
    df["ma57"] = pd.Series(close).rolling(window=57, min_periods=57).mean().to_numpy()
    df["ma114"] = pd.Series(close).rolling(window=114, min_periods=114).mean().to_numpy()
    df["long_line"] = (df["ma14"] + df["ma28"] + df["ma57"] + df["ma114"]) / 4.0

    df["ma20"] = pd.Series(close).rolling(window=20, min_periods=20).mean().to_numpy()
    df["ma60"] = pd.Series(close).rolling(window=60, min_periods=60).mean().to_numpy()
    df["ma120"] = pd.Series(close).rolling(window=120, min_periods=120).mean().to_numpy()

    return df


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["收盘"].to_numpy(dtype=float)
    trend_line = df["trend_line"].to_numpy(dtype=float)
    long_line = df["long_line"].to_numpy(dtype=float)
    ma20 = df["ma20"].to_numpy(dtype=float)
    ma60 = df["ma60"].to_numpy(dtype=float)

    df["f_trend_gt_long"] = (trend_line > long_line).astype(float)
    df["f_ma20_gt_ma60"] = (ma20 > ma60).astype(float)
    df["f_close_gt_trend"] = (close > trend_line).astype(float)

    df["f_trend_slope_5"] = compute_slope_ratio(trend_line, 5)
    df["f_long_slope_10"] = compute_slope_ratio(long_line, 10)
    df["f_ma20_slope_5"] = compute_slope_ratio(ma20, 5)

    safe_close = np.where(close == 0, np.nan, close)
    df["f_spread_trend_long"] = np.abs(trend_line - long_line) / safe_close
    df["f_spread_ma20_ma60"] = np.abs(ma20 - ma60) / safe_close

    df["f_avg_spread_trend_long_20"] = pd.Series(df["f_spread_trend_long"]).rolling(20, min_periods=20).mean().to_numpy()
    df["f_avg_spread_ma20_ma60_20"] = pd.Series(df["f_spread_ma20_ma60"]).rolling(20, min_periods=20).mean().to_numpy()

    df["f_cross_count_trend_long_20"] = rolling_count_sign_flips(trend_line - long_line, 20)
    df["f_cross_count_ma20_ma60_20"] = rolling_count_sign_flips(ma20 - ma60, 20)
    df["f_close_cross_trend_count_20"] = rolling_count_sign_flips(close - trend_line, 20)

    df["f_trend_positive_20"] = compute_positive_count(trend_line, 20)
    df["f_long_positive_20"] = compute_positive_count(long_line, 20)

    return df


def identify_regimes(df: pd.DataFrame) -> pd.DataFrame:
    trend_regime_mask = (
        (df["f_trend_gt_long"] == 1)
        & (df["f_ma20_gt_ma60"] == 1)
        & (df["f_close_gt_trend"] == 1)
        & (df["f_trend_slope_5"] > 0)
        & (df["f_long_slope_10"] > 0)
        & (df["f_avg_spread_trend_long_20"] >= 0.01)
        & (df["f_avg_spread_ma20_ma60_20"] >= 0.015)
        & (df["f_cross_count_trend_long_20"] <= 1)
        & (df["f_cross_count_ma20_ma60_20"] <= 1)
        & (df["f_close_cross_trend_count_20"] <= 3)
    )

    range_regime_mask = (
        (df["f_cross_count_trend_long_20"] >= 3)
        | (df["f_cross_count_ma20_ma60_20"] >= 3)
        | (
            (df["f_avg_spread_trend_long_20"] < 0.01)
            & (df["f_avg_spread_ma20_ma60_20"] < 0.015)
        )
        | (df["f_close_cross_trend_count_20"] >= 5)
        | (
            (df["f_trend_positive_20"] >= 8)
            & (df["f_trend_positive_20"] <= 12)
            & (df["f_long_positive_20"] >= 8)
            & (df["f_long_positive_20"] <= 12)
        )
    )

    df["regime"] = "other"
    df.loc[trend_regime_mask, "regime"] = "trend"
    df.loc[range_regime_mask, "regime"] = "range"
    return df


# =========================
# Future metrics
# =========================

def calculate_future_metrics(df: pd.DataFrame) -> pd.DataFrame:
    close = safe_numeric_array(df["收盘"].to_numpy(dtype=float))
    high = safe_numeric_array(df["最高"].to_numpy(dtype=float))
    low = safe_numeric_array(df["最低"].to_numpy(dtype=float))

    n = len(df)
    max_h = max(HORIZONS)

    for h in HORIZONS:
        df[f"future_max_rally_{h}d"] = np.nan
        df[f"future_close_ret_{h}d"] = np.nan
        df[f"future_max_dd_{h}d"] = np.nan
        df[f"dd_gt_3_{h}d"] = np.nan
        df[f"dd_gt_5_{h}d"] = np.nan

    if n <= max_h:
        return df

    if HAS_SLIDING_WINDOW:
        high_future = sliding_window_view(high[1:], max_h)
        low_future = sliding_window_view(low[1:], max_h)
        close_future = sliding_window_view(close[1:], max_h)

        valid_len = len(high_future)
        current_close = close[:valid_len]

        valid_current = (current_close > 0) & np.isfinite(current_close)

        for h in HORIZONS:
            h_high = high_future[:, :h]
            h_low = low_future[:, :h]
            h_close = close_future[:, :h]

            max_rally = np.full(valid_len, np.nan)
            close_ret = np.full(valid_len, np.nan)
            max_dd = np.full(valid_len, np.nan)

            with np.errstate(divide="ignore", invalid="ignore"):
                max_rally[valid_current] = (np.nanmax(h_high[valid_current], axis=1) / current_close[valid_current] - 1.0) * 100.0
                close_ret[valid_current] = (h_close[valid_current, -1] / current_close[valid_current] - 1.0) * 100.0
                max_dd[valid_current] = (np.nanmin(h_low[valid_current], axis=1) / current_close[valid_current] - 1.0) * 100.0

            max_rally[~np.isfinite(max_rally)] = np.nan
            close_ret[~np.isfinite(close_ret)] = np.nan
            max_dd[~np.isfinite(max_dd)] = np.nan

            df.loc[df.index[:valid_len], f"future_max_rally_{h}d"] = max_rally
            df.loc[df.index[:valid_len], f"future_close_ret_{h}d"] = close_ret
            df.loc[df.index[:valid_len], f"future_max_dd_{h}d"] = max_dd
            df.loc[df.index[:valid_len], f"dd_gt_3_{h}d"] = (max_dd <= -3.0).astype(float)
            df.loc[df.index[:valid_len], f"dd_gt_5_{h}d"] = (max_dd <= -5.0).astype(float)
    else:
        for i in range(n - max_h):
            current_close = close[i]
            if not np.isfinite(current_close) or current_close <= 0:
                continue

            future_highs = high[i + 1:i + 1 + max_h]
            future_lows = low[i + 1:i + 1 + max_h]
            future_closes = close[i + 1:i + 1 + max_h]

            for h in HORIZONS:
                hh = future_highs[:h]
                ll = future_lows[:h]
                cc = future_closes[:h]

                with np.errstate(divide="ignore", invalid="ignore"):
                    max_rally = (np.nanmax(hh) / current_close - 1.0) * 100.0
                    close_ret = (cc[-1] / current_close - 1.0) * 100.0
                    max_dd = (np.nanmin(ll) / current_close - 1.0) * 100.0

                if not np.isfinite(max_rally):
                    max_rally = np.nan
                if not np.isfinite(close_ret):
                    close_ret = np.nan
                if not np.isfinite(max_dd):
                    max_dd = np.nan

                df.loc[i, f"future_max_rally_{h}d"] = max_rally
                df.loc[i, f"future_close_ret_{h}d"] = close_ret
                df.loc[i, f"future_max_dd_{h}d"] = max_dd
                df.loc[i, f"dd_gt_3_{h}d"] = float(max_dd <= -3.0) if pd.notna(max_dd) else np.nan
                df.loc[i, f"dd_gt_5_{h}d"] = float(max_dd <= -5.0) if pd.notna(max_dd) else np.nan

    return df


def calculate_deviations(df: pd.DataFrame) -> pd.DataFrame:
    df["dev_trend"] = (df["收盘"] - df["trend_line"]) / df["trend_line"] * 100.0
    df["dev_long"] = (df["收盘"] - df["long_line"]) / df["long_line"] * 100.0
    return df


# =========================
# 阶段1：regime validation
# =========================

def validate_regime_effectiveness(df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
    parts = []

    for h in HORIZONS:
        rename_map = {
            f"future_max_rally_{h}d": "future_max_rally",
            f"future_close_ret_{h}d": "future_close_ret",
            f"future_max_dd_{h}d": "future_max_dd",
            f"dd_gt_3_{h}d": "dd_gt_3",
            f"dd_gt_5_{h}d": "dd_gt_5",
        }
        required_cols = list(rename_map.keys()) + ["regime"]
        valid_df = df.loc[df[required_cols].notna().all(axis=1), required_cols].copy()
        if valid_df.empty:
            continue

        valid_df = valid_df.rename(columns=rename_map)
        valid_df["stock_code"] = stock_code
        valid_df["horizon"] = h
        valid_df = safe_to_numeric(valid_df, REGIME_VALIDATION_METRIC_COLS)

        parts.append(valid_df[["stock_code", "horizon", "regime"] + REGIME_VALIDATION_METRIC_COLS])

    if not parts:
        return pd.DataFrame(columns=["stock_code", "horizon", "regime"] + REGIME_VALIDATION_METRIC_COLS)

    out = pd.concat(parts, ignore_index=True)
    out = safe_to_numeric(out, REGIME_VALIDATION_METRIC_COLS)
    return out


# =========================
# 阶段2：高速阈值扫描
# =========================

def prepare_sorted_scan_arrays(valid_df: pd.DataFrame, dev_col: str, target_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    返回按 dev 升序排序后的：
    dev_sorted, y_sorted, prefix_sum, suffix_sum
    """
    dev = valid_df[dev_col].to_numpy(dtype=float)
    y = valid_df[target_col].to_numpy(dtype=float)

    order = np.argsort(dev, kind="mergesort")
    dev_sorted = dev[order]
    y_sorted = y[order]

    prefix_sum = np.cumsum(y_sorted)
    suffix_sum = np.cumsum(y_sorted[::-1])[::-1]

    return dev_sorted, y_sorted, prefix_sum, suffix_sum


def fast_threshold_stats(
    dev_sorted: np.ndarray,
    y_sorted: np.ndarray,
    prefix_sum: np.ndarray,
    suffix_sum: np.ndarray,
    threshold: float
) -> Tuple[int, int, float, float, float]:
    """
    对于规则 dev >= threshold：
    用 searchsorted 找切点，O(log n) 计算 above/below 统计。
    """
    n = len(dev_sorted)
    idx = np.searchsorted(dev_sorted, threshold, side="left")

    below_count = idx
    above_count = n - idx

    if above_count == 0 or below_count == 0:
        return above_count, below_count, np.nan, np.nan, np.nan

    below_sum = prefix_sum[idx - 1] if idx > 0 else 0.0
    above_sum = suffix_sum[idx] if idx < n else 0.0

    below_rate = below_sum / below_count
    above_rate = above_sum / above_count
    diff = above_rate - below_rate

    return above_count, below_count, float(above_rate), float(below_rate), float(diff)


def scan_thresholds_for_regime(df: pd.DataFrame, dev_col: str, horizon: int) -> Dict:
    """
    高速阈值扫描：
    1. 过滤有效正偏离样本
    2. 预排序
    3. 对阈值用 searchsorted + prefix/suffix sum 快速计算
    """
    target_col = f"dd_gt_5_{horizon}d"
    required_cols = [dev_col, target_col, "日期"]

    valid_df = df.loc[df[required_cols].notna().all(axis=1), required_cols].copy()
    valid_df = safe_to_numeric(valid_df, [dev_col, target_col])

    valid_df = valid_df[(valid_df[dev_col] > 0)].copy()
    valid_df = valid_df[np.isfinite(valid_df[dev_col]) & np.isfinite(valid_df[target_col])].copy()

    if len(valid_df) < MIN_REGIME_SAMPLE_COUNT:
        return {
            "min_valid_threshold": np.nan,
            "best_threshold": np.nan,
            "auc": np.nan,
            "base_dd_gt_5": np.nan,
            "sample_count": len(valid_df),
        }

    valid_df = valid_df.sort_values("日期").reset_index(drop=True)

    dev_sorted, y_sorted, prefix_sum, suffix_sum = prepare_sorted_scan_arrays(valid_df, dev_col, target_col)

    # 1) coarse best
    best_threshold = np.nan
    best_score = -np.inf

    for threshold in COARSE_THRESHOLDS:
        above_count, below_count, above_rate, below_rate, diff = fast_threshold_stats(
            dev_sorted, y_sorted, prefix_sum, suffix_sum, threshold
        )
        if above_count < MIN_THRESHOLD_SAMPLE_COUNT or below_count < MIN_THRESHOLD_SAMPLE_COUNT:
            continue
        if np.isnan(diff) or diff < MIN_EVENT_RATE_INCREASE:
            continue

        score = diff * np.sqrt(above_count)
        if score > best_score:
            best_score = score
            best_threshold = threshold

    # 2) fine best
    if pd.notna(best_threshold):
        fine_start = max(1.0, best_threshold - 2.0)
        fine_end = min(20.0, best_threshold + 2.0)
        fine_thresholds = np.arange(fine_start, fine_end + FINE_STEP, FINE_STEP)

        for threshold in fine_thresholds:
            above_count, below_count, above_rate, below_rate, diff = fast_threshold_stats(
                dev_sorted, y_sorted, prefix_sum, suffix_sum, threshold
            )
            if above_count < MIN_THRESHOLD_SAMPLE_COUNT or below_count < MIN_THRESHOLD_SAMPLE_COUNT:
                continue
            if np.isnan(diff) or diff < MIN_EVENT_RATE_INCREASE:
                continue

            score = diff * np.sqrt(above_count)
            if score > best_score:
                best_score = score
                best_threshold = threshold

    # 3) coarse min valid
    coarse_min_threshold = np.nan
    min_valid_threshold = np.nan

    for threshold in COARSE_THRESHOLDS:
        above_count, below_count, above_rate, below_rate, diff = fast_threshold_stats(
            dev_sorted, y_sorted, prefix_sum, suffix_sum, threshold
        )
        if above_count < MIN_THRESHOLD_SAMPLE_COUNT or below_count < MIN_THRESHOLD_SAMPLE_COUNT:
            continue
        if np.isnan(diff) or diff < MIN_EVENT_RATE_INCREASE:
            continue

        coarse_min_threshold = threshold
        min_valid_threshold = threshold
        break

    # 4) fine min valid：从更低阈值开始，第一次满足即 break
    if pd.notna(coarse_min_threshold):
        fine_min_start = max(1.0, coarse_min_threshold - 2.0)
        fine_min_end = coarse_min_threshold
        fine_min_thresholds = np.arange(fine_min_start, fine_min_end + FINE_STEP, FINE_STEP)

        for threshold in fine_min_thresholds:
            above_count, below_count, above_rate, below_rate, diff = fast_threshold_stats(
                dev_sorted, y_sorted, prefix_sum, suffix_sum, threshold
            )
            if above_count < MIN_THRESHOLD_SAMPLE_COUNT or below_count < MIN_THRESHOLD_SAMPLE_COUNT:
                continue
            if np.isnan(diff) or diff < MIN_EVENT_RATE_INCREASE:
                continue

            min_valid_threshold = threshold
            break

    # 5) 单变量时间切分 AUC
    auc = np.nan
    if SKLEARN_AVAILABLE:
        try:
            X = valid_df[[dev_col]].to_numpy(dtype=float)
            y = valid_df[target_col].to_numpy(dtype=float)

            if len(np.unique(y)) >= 2:
                split_idx = int(len(valid_df) * 0.7)
                if 0 < split_idx < len(valid_df):
                    X_train, X_test = X[:split_idx], X[split_idx:]
                    y_train, y_test = y[:split_idx], y[split_idx:]

                    if len(np.unique(y_train)) >= 2 and len(np.unique(y_test)) >= 2:
                        model = LogisticRegression(max_iter=1000, random_state=42)
                        model.fit(X_train, y_train)
                        y_pred = model.predict_proba(X_test)[:, 1]
                        auc = roc_auc_score(y_test, y_pred)
        except Exception:
            pass

    return {
        "min_valid_threshold": min_valid_threshold,
        "best_threshold": best_threshold,
        "auc": auc,
        "base_dd_gt_5": float(valid_df[target_col].mean()),
        "sample_count": len(valid_df),
    }


# =========================
# 单股处理
# =========================

def process_stock(df: pd.DataFrame, stock_code: str) -> Tuple[Dict, pd.DataFrame]:
    result = {
        "stock_code": stock_code,
        "regime_trend_sample_count": 0,
        "regime_range_sample_count": 0,
        "regime_other_sample_count": 0,
    }

    df = calculate_indicators(df)
    df = calculate_features(df)
    df = identify_regimes(df)
    df = calculate_future_metrics(df)
    df = calculate_deviations(df)

    for regime in ["trend", "range", "other"]:
        result[f"regime_{regime}_sample_count"] = int((df["regime"] == regime).sum())

    for h in HORIZONS:
        for regime in ["trend", "range"]:
            regime_df = df[df["regime"] == regime]

            for dev_type in ["trend", "long"]:
                dev_col = f"dev_{dev_type}"
                scan_result = scan_thresholds_for_regime(regime_df, dev_col, h)

                result[f"{dev_type}_threshold_in_{regime}_regime_{h}d"] = scan_result["min_valid_threshold"]
                result[f"{dev_type}_best_{regime}_regime_{h}d"] = scan_result["best_threshold"]
                result[f"{dev_type}_auc_{regime}_regime_{h}d"] = scan_result["auc"]
                result[f"{dev_type}_base_dd_gt_5_{regime}_regime_{h}d"] = scan_result["base_dd_gt_5"]

    df["stock_code"] = stock_code
    return result, df


# =========================
# 状态分布
# =========================

def analyze_regime_distribution(all_daily_df: pd.DataFrame) -> Dict:
    raw_total = len(all_daily_df)
    raw_trend = int((all_daily_df["regime"] == "trend").sum())
    raw_range = int((all_daily_df["regime"] == "range").sum())
    raw_other = int((all_daily_df["regime"] == "other").sum())

    valid_mask = all_daily_df[["regime", "f_trend_gt_long", "f_cross_count_trend_long_20"]].notna().all(axis=1)
    for h in HORIZONS:
        col = f"future_max_dd_{h}d"
        if col in all_daily_df.columns:
            valid_mask &= all_daily_df[col].notna()

    valid_df = all_daily_df.loc[valid_mask]
    valid_total = len(valid_df)
    valid_trend = int((valid_df["regime"] == "trend").sum())
    valid_range = int((valid_df["regime"] == "range").sum())
    valid_other = int((valid_df["regime"] == "other").sum())

    return {
        "raw": {
            "total_samples": raw_total,
            "trend_count": raw_trend,
            "range_count": raw_range,
            "other_count": raw_other,
            "trend_ratio": raw_trend / raw_total if raw_total > 0 else 0.0,
            "range_ratio": raw_range / raw_total if raw_total > 0 else 0.0,
            "other_ratio": raw_other / raw_total if raw_total > 0 else 0.0,
        },
        "valid": {
            "total_samples": valid_total,
            "trend_count": valid_trend,
            "range_count": valid_range,
            "other_count": valid_other,
            "trend_ratio": valid_trend / valid_total if valid_total > 0 else 0.0,
            "range_ratio": valid_range / valid_total if valid_total > 0 else 0.0,
            "other_ratio": valid_other / valid_total if valid_total > 0 else 0.0,
        },
    }


# =========================
# 结论
# =========================

def generate_conclusion(
    regime_validation_df: pd.DataFrame,
    stock_thresholds_df: pd.DataFrame,
    regime_distribution: Dict
) -> str:
    lines = ["=" * 80, "个股分状态卖点阈值研究结论", "=" * 80, ""]

    lines.append("【状态分布统计】")
    lines.append("=" * 80)

    raw = regime_distribution["raw"]
    valid = regime_distribution["valid"]

    lines.append("原始状态分布（所有已分类样本）:")
    lines.append(f"  总样本数: {raw['total_samples']:,}")
    lines.append(f"  trend regime: {raw['trend_count']:,} ({raw['trend_ratio']:.2%})")
    lines.append(f"  range regime: {raw['range_count']:,} ({raw['range_ratio']:.2%})")
    lines.append(f"  other regime: {raw['other_count']:,} ({raw['other_ratio']:.2%})")

    lines.append("\n有效样本状态分布（参与分析的样本）:")
    lines.append(f"  总样本数: {valid['total_samples']:,}")
    lines.append(f"  trend regime: {valid['trend_count']:,} ({valid['trend_ratio']:.2%})")
    lines.append(f"  range regime: {valid['range_count']:,} ({valid['range_ratio']:.2%})")
    lines.append(f"  other regime: {valid['other_count']:,} ({valid['other_ratio']:.2%})")

    lines.append("\n" + "=" * 80)
    lines.append("【阶段1：状态有效性验证（全市场汇总）】")
    lines.append("=" * 80)

    if not regime_validation_df.empty:
        regime_validation_df = regime_validation_df.copy()
        regime_validation_df = safe_to_numeric(regime_validation_df, REGIME_VALIDATION_METRIC_COLS)

        summary = (
            regime_validation_df.groupby(["horizon", "regime"], as_index=False)
            .agg({
                "future_max_rally": "mean",
                "future_close_ret": "mean",
                "future_max_dd": "mean",
                "dd_gt_3": "mean",
                "dd_gt_5": "mean",
            })
        )

        for h in HORIZONS:
            lines.append(f"\n{h}日:")
            horizon_df = summary[summary["horizon"] == h]

            trend_row = horizon_df[horizon_df["regime"] == "trend"]
            range_row = horizon_df[horizon_df["regime"] == "range"]

            has_trend = not trend_row.empty
            has_range = not range_row.empty

            if has_trend:
                row = trend_row.iloc[0]
                lines.append("  trend regime:")
                lines.append(f"    平均最大涨幅: {row['future_max_rally']:.2f}%")
                lines.append(f"    平均收盘收益: {row['future_close_ret']:.2f}%")
                lines.append(f"    平均最大回撤: {row['future_max_dd']:.2f}%")
                lines.append(f"    dd_gt_3概率: {row['dd_gt_3']:.2%}")
                lines.append(f"    dd_gt_5概率: {row['dd_gt_5']:.2%}")

            if has_range:
                row = range_row.iloc[0]
                lines.append("  range regime:")
                lines.append(f"    平均最大涨幅: {row['future_max_rally']:.2f}%")
                lines.append(f"    平均收盘收益: {row['future_close_ret']:.2f}%")
                lines.append(f"    平均最大回撤: {row['future_max_dd']:.2f}%")
                lines.append(f"    dd_gt_3概率: {row['dd_gt_3']:.2%}")
                lines.append(f"    dd_gt_5概率: {row['dd_gt_5']:.2%}")

            if has_trend and has_range:
                t = trend_row.iloc[0]
                r = range_row.iloc[0]
                lines.append("  差值 (range - trend):")
                lines.append(f"    最大涨幅差值: {r['future_max_rally'] - t['future_max_rally']:.2f}%")
                lines.append(f"    收盘收益差值: {r['future_close_ret'] - t['future_close_ret']:.2f}%")
                lines.append(f"    最大回撤差值: {r['future_max_dd'] - t['future_max_dd']:.2f}%")
                lines.append(f"    dd_gt_3概率差值: {(r['dd_gt_3'] - t['dd_gt_3']) * 100:.2f}pp")
                lines.append(f"    dd_gt_5概率差值: {(r['dd_gt_5'] - t['dd_gt_5']) * 100:.2f}pp")

    lines.append("\n" + "=" * 80)
    lines.append("【阶段2：分状态阈值统计】")
    lines.append("=" * 80)

    if not stock_thresholds_df.empty:
        for h in HORIZONS:
            lines.append(f"\n{h}日:")
            for dev_type in ["trend", "long"]:
                trend_col = f"{dev_type}_threshold_in_trend_regime_{h}d"
                range_col = f"{dev_type}_threshold_in_range_regime_{h}d"

                trend_valid = pd.to_numeric(stock_thresholds_df[trend_col], errors="coerce").dropna()
                range_valid = pd.to_numeric(stock_thresholds_df[range_col], errors="coerce").dropna()

                lines.append(f"  {dev_type}偏离:")
                lines.append(f"    trend regime有阈值: {len(trend_valid)}只")
                if len(trend_valid) > 0:
                    lines.append(f"    trend regime平均阈值: {trend_valid.mean():.2f}%")

                lines.append(f"    range regime有阈值: {len(range_valid)}只")
                if len(range_valid) > 0:
                    lines.append(f"    range regime平均阈值: {range_valid.mean():.2f}%")

                both_valid = stock_thresholds_df[
                    stock_thresholds_df[trend_col].notna() & stock_thresholds_df[range_col].notna()
                ].copy()

                if not both_valid.empty:
                    both_valid[trend_col] = pd.to_numeric(both_valid[trend_col], errors="coerce")
                    both_valid[range_col] = pd.to_numeric(both_valid[range_col], errors="coerce")
                    both_valid = both_valid.dropna(subset=[trend_col, range_col])

                    if not both_valid.empty:
                        range_lower = int((both_valid[range_col] < both_valid[trend_col]).sum())
                        lines.append(
                            f"    震荡阈值<趋势阈值: {range_lower}/{len(both_valid)} ({range_lower / len(both_valid):.1%})"
                        )

    lines.append("\n" + "=" * 80)
    lines.append("核心结论:")
    lines.append("=" * 80)
    lines.append("1. 趋势状态和震荡状态的风险收益特征可能存在明显差异")
    lines.append("2. 分状态研究阈值可能比统一阈值更有针对性")
    lines.append("3. 建议结合状态识别和分状态阈值使用")

    return "\n".join(lines)


# =========================
# 主函数
# =========================

def main() -> None:
    print("=" * 80)
    print("开始个股分状态卖点阈值研究（修正高速版）")
    print("=" * 80)

    if DATA_DIR.exists():
        stock_files = list(DATA_DIR.rglob("*.txt"))
    else:
        print(f"警告: 数据目录不存在 {DATA_DIR}")
        stock_files = []

    print(f"\n发现 {len(stock_files)} 个股票文件，开始处理...")

    all_stock_daily_results: List[pd.DataFrame] = []
    all_stock_threshold_results: List[Dict] = []
    regime_validation_parts: List[pd.DataFrame] = []

    for idx, file_path in enumerate(stock_files):
        stock_code = file_path.stem.split("#")[-1] if "#" in file_path.stem else file_path.stem

        if idx % 50 == 0:
            print(f"处理进度: {idx}/{len(stock_files)}")

        df = load_stock_data(str(file_path))
        if df is None or len(df) < MIN_HISTORY_BARS:
            continue

        try:
            threshold_result, daily_df = process_stock(df, stock_code)
            all_stock_threshold_results.append(threshold_result)
            all_stock_daily_results.append(daily_df)

            validation_df = validate_regime_effectiveness(daily_df, stock_code)
            if not validation_df.empty:
                regime_validation_parts.append(validation_df)

        except Exception as e:
            print(f"  处理 {stock_code} 时出错: {e}")
            continue

    print("\n处理完成！")
    print(f"成功处理股票数: {len(all_stock_threshold_results)}")

    if not all_stock_threshold_results:
        print("未找到符合条件的样本！")
        return

    print("\n保存结果文件...")

    all_daily_df = pd.concat(all_stock_daily_results, ignore_index=True)

    regime_output_cols = ["stock_code", "日期"] + REGIME_FEATURE_COLS + DEBUG_EXTRA_COLS
    regime_output_cols = [c for c in regime_output_cols if c in all_daily_df.columns]
    regime_output_df = all_daily_df[regime_output_cols].copy()
    regime_output_df.to_csv(OUTPUT_DIR / "daily_regime_classification.csv", index=False, encoding="utf-8-sig")
    print("每日状态分类结果已保存")

    stock_thresholds_df = pd.DataFrame(all_stock_threshold_results)
    stock_thresholds_df.to_csv(OUTPUT_DIR / "stock_regime_thresholds.csv", index=False, encoding="utf-8-sig")
    print("个股分状态阈值表已保存")

    print("\n分析状态分布...")
    regime_distribution = analyze_regime_distribution(all_daily_df)

    print("\n生成状态有效性验证...")
    if regime_validation_parts:
        regime_validation_df = pd.concat(regime_validation_parts, ignore_index=True)
        regime_validation_df = safe_to_numeric(regime_validation_df, REGIME_VALIDATION_METRIC_COLS)
        regime_validation_df.to_csv(OUTPUT_DIR / "regime_validation.csv", index=False, encoding="utf-8-sig")
        print("状态有效性验证结果已保存")
    else:
        regime_validation_df = pd.DataFrame(columns=["stock_code", "horizon", "regime"] + REGIME_VALIDATION_METRIC_COLS)

    print("\n生成结论...")
    conclusion = generate_conclusion(regime_validation_df, stock_thresholds_df, regime_distribution)

    with open(OUTPUT_DIR / "conclusion.txt", "w", encoding="utf-8") as f:
        f.write(conclusion)

    print("\n" + "=" * 80)
    print(conclusion)
    print("\n" + "=" * 80)
    print(f"\n所有结果已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()