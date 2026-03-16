from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd

from utils.shared_market_features import compute_base_features, safe_div


MIN_BARS = 120
EPS = 1e-12

# B2 第一类最优参数
TYPE1_RET1_MIN = 0.04
TYPE1_J_MAX = 90.0
TYPE1_UPPER_SHADOW_BODY_RATIO = 0.4
TYPE1_START_NEAR_RATIO = 1.02
TYPE1_J_LOW_RANK20_MAX = 0.10

# B2 第四类最优参数
TYPE4_RET1_MIN = 0.03
TYPE4_J_MAX = 100.0
TYPE4_UPPER_SHADOW_BODY_RATIO = 0.8
TYPE4_TOUCH_RATIO = 1.01
TYPE4_LOOKBACK = 20

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
    include_optional_types: bool = True,
) -> pd.DataFrame:
    x = precomputed_base.copy() if precomputed_base is not None else compute_base_features(df)
    x["j_rank20"] = x["J"].rolling(20, min_periods=20).apply(
        lambda win: pd.Series(win).rank(pct=True).iloc[-1],
        raw=False,
    )
    x["j_rank20_prev"] = x["j_rank20"].shift(1)

    x["vol_ma5"] = x["volume"].rolling(5).mean()
    x["signal_vs_ma5"] = pd.Series(safe_div(x["volume"], x["vol_ma5"]), index=x.index)

    # 启动线定义
    # 启动线定义
    x["trend_start"] = (x["close"].shift(1) <= x["trend_line"].shift(1) * TYPE1_START_NEAR_RATIO) & (
        x["close"] > x["trend_line"]
    )
    x["long_start"] = (x["close"].shift(1) <= x["long_line"].shift(1) * TYPE1_START_NEAR_RATIO) & (
        x["close"] > x["long_line"]
    )
    x["dual_start"] = x["trend_start"] & x["long_start"]

    # K 线质量
    real_body = (x["close"] - x["open"]).abs()
    upper_shadow = x["high"] - np.maximum(x["open"], x["close"])
    lower_shadow = np.minimum(x["open"], x["close"]) - x["low"]
    full_range = (x["high"] - x["low"]).replace(0, np.nan)
    x["upper_shadow_ratio"] = pd.Series(safe_div(upper_shadow, full_range), index=x.index)
    x["close_position"] = pd.Series(safe_div(x["close"] - x["low"], full_range), index=x.index)
    x["small_upper_shadow_type1"] = (real_body <= EPS) | (
        upper_shadow <= real_body * TYPE1_UPPER_SHADOW_BODY_RATIO + EPS
    )
    x["small_upper_shadow_type4"] = (real_body <= EPS) | (
        upper_shadow <= real_body * TYPE4_UPPER_SHADOW_BODY_RATIO + EPS
    )

    # B2 基础条件
    x["b2_volume_ok"] = (x["volume"] > x["volume"].shift(1)) & (x["volume"] > x["vol_ma5"])
    x["b2_j_ok_type1"] = (
        (x["J"] < TYPE1_J_MAX)
        & (x["J"] > x["J"].shift(1))
        & (x["J"].shift(1) < x["J"].shift(2))
        & (x["J"].shift(2) < x["J"].shift(3))
    )
    x["b2_j_ok_type4"] = (
        (x["J"] < TYPE4_J_MAX)
        & (x["J"] > x["J"].shift(1))
        & (x["J"].shift(1) < x["J"].shift(2))
        & (x["J"].shift(2) < x["J"].shift(3))
    )

    x["base_b2_type1"] = (
        x["trend_ok"]
        & (x["close"] > x["open"])
        & (x["ret1"] >= TYPE1_RET1_MIN)
        & x["small_upper_shadow_type1"]
        & x["b2_volume_ok"]
        & x["b2_j_ok_type1"]
    )
    x["base_b2_type4"] = (
        x["trend_ok"]
        & (x["close"] > x["open"])
        & (x["ret1"] >= TYPE4_RET1_MIN)
        & x["small_upper_shadow_type4"]
        & x["b2_volume_ok"]
        & x["b2_j_ok_type4"]
    )
    x["base_b2"] = x["base_b2_type1"] | x["base_b2_type4"]

    # 第一类：T-1 收盘在多空线附近，且 J 进入低位区间，T 日出 B2
    x["type1"] = (
        (x["close"].shift(1) <= x["long_line"].shift(1) * TYPE1_START_NEAR_RATIO)
        & (x["j_rank20_prev"] <= TYPE1_J_LOW_RANK20_MAX)
    )

    # 第二类：横盘震荡后突然放量 B2（先保留定义，当前主流程不启用）
    x["box_high40"] = x["high"].shift(1).rolling(40).max()
    x["box_low40"] = x["low"].shift(1).rolling(40).min()
    x["box_range40"] = x["box_high40"] / x["box_low40"] - 1.0
    x["box_net40"] = x["close"].shift(1) / x["close"].shift(40) - 1.0
    x["box_slope20"] = x["trend_line"].shift(1) / x["trend_line"].shift(20) - 1.0
    x["type2"] = (
        (x["box_range40"] <= 0.45)
        & (x["box_net40"].abs() <= 0.12)
        & (x["box_slope20"].abs() <= 0.05)
        & (x["volume"] > x["vol_ma5"])
    )

    # 第三类：前有巨量阳量，中间阴量被前巨量阳量和当前 B2 包住（研究脚本用，主流程默认跳过）
    x["type3"] = False
    x["middle_bear_ratio"] = np.nan
    if include_optional_types:
        vol = x["volume"].astype(float).tolist()
        opens = x["open"].astype(float).tolist()
        closes = x["close"].astype(float).tolist()
        for i in range(len(x)):
            if i < 31:
                continue
            left = max(1, i - 30)
            anchor_idx = None
            recent_top3 = x.loc[left : i - 1, "volume"].nlargest(3)
            top3_threshold = recent_top3.min() if len(recent_top3) == 3 else (recent_top3.min() if len(recent_top3) else np.nan)
            for j in range(i - 1, left - 1, -1):
                if closes[j] <= opens[j]:
                    continue
                if vol[j] < vol[j - 1] * 2:
                    continue
                if pd.isna(top3_threshold) or vol[j] + EPS < float(top3_threshold):
                    continue
                anchor_idx = j
                break
            if anchor_idx is None or anchor_idx >= i - 1:
                continue
            middle = x.iloc[anchor_idx + 1 : i]
            bears = middle[middle["close"] < middle["open"]]
            if bears.empty:
                x.at[i, "type3"] = True
                x.at[i, "middle_bear_ratio"] = 0.0
                continue
            cond = (bears["volume"] < vol[anchor_idx]) & (bears["volume"] < vol[i])
            if bool(cond.all()):
                x.at[i, "type3"] = True
                x.at[i, "middle_bear_ratio"] = len(bears) / max(len(middle), 1)

    # 第四类：趋势线第一次上穿多空线后，T-1 第一次回踩白线，T 日出 B2
    bull_cross = (x["trend_line"] > x["long_line"]) & (x["trend_line"].shift(1) <= x["long_line"].shift(1))
    x["type4"] = False
    for i in range(len(x)):
        if i < 10:
            continue
        left = max(1, i - TYPE4_LOOKBACK)
        crosses = np.where(bull_cross.iloc[left:i].to_numpy())[0]
        if len(crosses) == 0:
            continue
        cross_idx = left + int(crosses[-1])
        prev_touch = (
            (x.at[i - 1, "low"] <= x.at[i - 1, "trend_line"] * TYPE4_TOUCH_RATIO)
            or (x.at[i - 1, "close"] <= x.at[i - 1, "trend_line"] * TYPE4_TOUCH_RATIO)
        )
        if not prev_touch:
            continue
        if i - 2 > cross_idx:
            between = x.iloc[cross_idx + 1 : i - 1]
            had_touch = (
                (between["low"] <= between["trend_line"] * TYPE4_TOUCH_RATIO)
                | (between["close"] <= between["trend_line"] * TYPE4_TOUCH_RATIO)
            ).any()
            if had_touch:
                continue
        x.at[i, "type4"] = True

    x["any_type"] = x[["type1", "type2", "type3", "type4"]].any(axis=1)
    x["ordinary"] = x["base_b2"] & (~x["any_type"])

    # 排序仅用于同日相对比较，不参与硬过滤
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
    j_headroom = (1.0 - np.minimum(x["J"].fillna(200.0) / TYPE4_J_MAX, 1.0)).clip(lower=0.0)
    x["sort_score"] = (
        0.35 * close_quality
        + 0.20 * volume_quality
        + 0.20 * trend_spread
        + 0.10 * j_headroom
        + 0.10 * np.maximum(trend_slope5.fillna(0.0), 0.0)
        + 0.03 * x["type1"].astype(float)
        + 0.02 * x["type4"].astype(float)
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
    active_type1 = bool(latest["type1"]) and bool(latest["base_b2_type1"])
    active_type4 = bool(latest["type4"]) and bool(latest["base_b2_type4"])
    # 预留类型：
    # active_type2 = bool(latest["type2"])
    # active_type3 = bool(latest["type3"])
    # active_ordinary = bool(latest["ordinary"])
    if not (active_type1 or active_type4):
        return [-1]

    stop_loss_price = round(float(latest["low"]), 3)
    type_names = []
    if active_type1:
        type_names.append("第一类")
    if active_type4:
        type_names.append("第四类")
    reason = "B2" + ("+".join(type_names) if type_names else "")
    return [
        1,
        stop_loss_price,
        float(latest["close"]),
        round(float(latest["sort_score"]), 4),
        reason,
    ]
