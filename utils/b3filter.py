from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from utils import b1filter, b2filter


EPS = 1e-12

# B3 当前基础定义参数
B3_RET1_MAX = 0.02
B3_AMPLITUDE_MAX = 0.05
B3_VOL_SHRINK_MAX = 1.0
B3_BASE_B2_RET1_MIN = 0.04
B3_BASE_B2_UPPER_SHADOW_BODY_RATIO = 1.0 / 3.0
B3_PREV_B1_J_MAX = 13.0


def safe_div(a, b, default=np.nan):
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    out = np.full(np.shape(a_arr), default, dtype=float)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr) & (np.abs(b_arr) > EPS)
    out[mask] = a_arr[mask] / b_arr[mask]
    return out


def load_one_csv(path: str):
    return b2filter.load_one_csv(path)


def to_b1_daily_df(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    return pd.DataFrame(
        {
            "日期": x["date"],
            "开盘": x["open"],
            "最高": x["high"],
            "最低": x["low"],
            "收盘": x["close"],
            "成交量": x["volume"],
            # B1 日线判断不使用成交额，补零即可
            "成交额": 0.0,
        }
    )


def build_daily_b1_flags(df: pd.DataFrame) -> pd.DataFrame:
    df_cn = to_b1_daily_df(df)
    df_kdj = b1filter.technical_indicators.calculate_kdj(df_cn.copy())
    flags = ((df_kdj["J"] < B3_PREV_B1_J_MAX) & (df_kdj["J"].shift(1) > df_kdj["J"])).fillna(False)
    reasons = np.where(flags, "J<13且前一日J更高", "")
    return pd.DataFrame({"daily_b1_signal": flags.astype(bool), "daily_b1_reason": reasons}, index=df.index)


def add_features(df: pd.DataFrame, precomputed_b2: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    x = precomputed_b2.copy() if precomputed_b2 is not None else b2filter.add_features(df).copy()
    b1_flags = build_daily_b1_flags(df)
    x["daily_b1_signal"] = b1_flags["daily_b1_signal"]
    x["daily_b1_reason"] = b1_flags["daily_b1_reason"]

    x["bull_close"] = x["close"] > x["open"]
    x["amplitude"] = pd.Series(safe_div(x["high"] - x["low"], x["low"]), index=x.index)
    x["vol_vs_prev"] = pd.Series(safe_div(x["volume"], x["volume"].shift(1)), index=x.index)
    x["vol_shrink"] = x["vol_vs_prev"] < B3_VOL_SHRINK_MAX

    real_body = (x["close"] - x["open"]).abs()
    upper_shadow = x["high"] - np.maximum(x["open"], x["close"])
    x["b2_upper_shadow_small"] = (real_body > EPS) & (
        upper_shadow <= real_body * B3_BASE_B2_UPPER_SHADOW_BODY_RATIO + EPS
    )

    x["base_b2_prev_b1"] = (
        x["trend_ok"]
        & x["bull_close"]
        & (x["ret1"] >= B3_BASE_B2_RET1_MIN)
        & x["daily_b1_signal"].shift(1).fillna(False)
        & (x["volume"] > x["volume"].shift(1))
        & x["b2_upper_shadow_small"]
    )
    x["prev_b2_any"] = x["base_b2_prev_b1"].shift(1).fillna(False)

    x["b3_signal"] = (
        x["prev_b2_any"]
        & x["bull_close"]
        & (x["ret1"] < B3_RET1_MAX)
        & (x["amplitude"] < B3_AMPLITUDE_MAX)
        & x["vol_shrink"]
    )

    prev_score = x["sort_score"].shift(1).fillna(0.0)
    ret_quality = (1.0 - np.minimum(np.abs(x["ret1"].fillna(1.0)) / B3_RET1_MAX, 1.0)).clip(lower=0.0)
    amp_quality = (1.0 - np.minimum(x["amplitude"].fillna(1.0) / B3_AMPLITUDE_MAX, 1.0)).clip(lower=0.0)
    shrink_quality = (1.0 - np.minimum(np.maximum(x["vol_vs_prev"].fillna(10.0) - 0.70, 0.0) / 0.30, 1.0)).clip(lower=0.0)
    x["b3_score"] = (
        0.45 * prev_score
        + 0.20 * ret_quality
        + 0.20 * amp_quality
        + 0.12 * shrink_quality
        + 0.03 * x["prev_b2_any"].astype(float)
    )
    return x


def describe_prev_types(latest: pd.Series) -> List[str]:
    if bool(latest["prev_b2_any"]):
        return ["基础B2"]
    return []


def check(file_path, hold_list=None, feature_cache=None):
    if feature_cache is not None:
        weekly_ok, weekly_reason = feature_cache.weekly_screen()
    else:
        weekly_ok, weekly_reason = b1filter.weekly_screen(str(file_path))
    if not weekly_ok:
        return [-1]

    if feature_cache is not None:
        x = feature_cache.b3_features()
        if x is None or x.empty:
            return [-1]
    else:
        df = load_one_csv(str(file_path))
        if df is None or df.empty:
            return [-1]
        x = add_features(df)
    latest = x.iloc[-1]
    if not bool(latest["b3_signal"]):
        return [-1]

    prev_types = describe_prev_types(latest)
    daily_reason = "B3承接(" + "+".join(prev_types) + ")" if prev_types else "B3承接"
    reason = f"周线：{weekly_reason} | 日线：{daily_reason}"
    return [
        1,
        round(float(latest["low"]), 3),
        float(latest["close"]),
        round(float(latest["b3_score"]), 4),
        reason,
    ]
