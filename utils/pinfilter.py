from utils import b1filter, brick_filter, stoploss, technical_indicators
from utils.market_risk_tags import format_risk_note
import numpy as np
import pandas as pd


EPS = 1e-12
N_LOOKBACK = 80

# 单针分型说明：
# A型：缩量回踩型
#   前面已经有一段趋势推进，最近 3 天温和回踩，单针日量能缩下来，
#   更像强趋势中的洗盘/回踩后企稳。
# B型：强趋势加速型
#   背景趋势更陡、前 10 日涨幅更强，单针不是单纯缩量洗盘，
#   而是强趋势里的再次发力/加速。
# C型：结构支撑型
#   单针出现在更有结构依托的位置，例如沿趋势线上涨、N 型向上、
#   或前期关键支撑 K 有效，属于“有结构支撑的回踩型单针”。
#
# 当前默认推荐顺序（稳健优先）：
# A型 > B型 > C型
# - A型：高
# - B型：中高
# - C型：中


TYPE_PRIORITY = {
    "A型(缩量回踩)": 1,
    "B型(强趋势加速)": 2,
    "C型(结构支撑)": 3,
}

TYPE_RECOMMEND = {
    "A型(缩量回踩)": "高",
    "B型(强趋势加速)": "中高",
    "C型(结构支撑)": "中",
}


def safe_div(a, b):
    if b is None or not np.isfinite(b) or abs(b) <= EPS:
        return np.nan
    if a is None or not np.isfinite(a):
        return np.nan
    return float(a) / float(b)


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


def identify_low_zones(mask_series: pd.Series):
    mask = mask_series.fillna(False).to_numpy(dtype=bool)
    zones = []
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


def build_n_up_feature(df: pd.DataFrame, rank_col: str, rank_threshold: float) -> pd.Series:
    out = np.zeros(len(df), dtype=bool)
    lows = df["low"].astype(float).to_numpy()
    highs = df["high"].astype(float).to_numpy()
    closes = df["close"].astype(float).to_numpy()
    rank_values = df[rank_col].astype(float)

    for idx in range(len(df)):
        left = max(0, idx - N_LOOKBACK + 1)
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


def add_structure_tags(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    out["j_rank_20"] = rolling_last_percentile(out["J"], 20)
    out["j_rank_30"] = rolling_last_percentile(out["J"], 30)
    out["n_up_rank20_p10"] = build_n_up_feature(out, "j_rank_20", 0.10)
    out["n_up_rank30_p10"] = build_n_up_feature(out, "j_rank_30", 0.10)
    out["n_up_any"] = out["n_up_rank20_p10"] | out["n_up_rank30_p10"]

    anchor_df = brick_filter.last_double_bull_anchor(out, lookback=60)
    out = pd.concat([out, anchor_df], axis=1)
    keyk_df = brick_filter.derive_keyk_states(out)
    out = pd.concat([out, keyk_df], axis=1)

    trend_slope_10 = out["trend_line"] / out["trend_line"].shift(10) - 1.0
    above_trend = (out["close"] >= out["trend_line"] * 0.99).astype(float)
    above_ratio_15 = above_trend.rolling(15, min_periods=15).mean()
    dist_trend = (out["close"] - out["trend_line"]) / out["close"].replace(0, np.nan)
    dist_min_15 = dist_trend.rolling(15, min_periods=15).min()
    out["along_trend_up"] = (
        (trend_slope_10 > 0.02)
        & (above_ratio_15 >= 0.80)
        & (dist_min_15 > -0.03)
    )

    full_range = (out["high"] - out["low"]).replace(0, np.nan)
    body_ratio = (out["close"] - out["open"]).abs() / full_range
    prev_vol = out["volume"].shift(1)
    vol_rank30 = (
        out["volume"]
        .rolling(30, min_periods=1)
        .apply(lambda s: pd.Series(s).rank(method="min", ascending=False).iloc[-1], raw=False)
    )
    giant_bear = (
        (out["close"] < out["open"])
        & ((out["close"] / out["close"].shift(1) - 1.0) < -0.03)
        & (body_ratio > 0.40)
        & ((vol_rank30 <= 3.0) | ((out["volume"] / prev_vol.replace(0, np.nan)) >= 2.0))
    )
    out["no_giant_bear_30"] = ~giant_bear.rolling(30, min_periods=1).max().fillna(0.0).astype(bool)
    return out


def ensure_pin_aux_cols(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    if "along_trend_up" not in out.columns:
        trend_slope_10 = out["trend_line"] / out["trend_line"].shift(10) - 1.0
        above_trend = (out["close"] >= out["trend_line"] * 0.99).astype(float)
        above_ratio_15 = above_trend.rolling(15, min_periods=15).mean()
        dist_trend = (out["close"] - out["trend_line"]) / out["close"].replace(0, np.nan)
        dist_min_15 = dist_trend.rolling(15, min_periods=15).min()
        out["along_trend_up"] = (
            (trend_slope_10 > 0.02)
            & (above_ratio_15 >= 0.80)
            & (dist_min_15 > -0.03)
        )

    if "no_giant_bear_30" not in out.columns:
        full_range = (out["high"] - out["low"]).replace(0, np.nan)
        body_ratio = (out["close"] - out["open"]).abs() / full_range
        prev_vol = out["volume"].shift(1)
        vol_rank30 = (
            out["volume"]
            .rolling(30, min_periods=1)
            .apply(lambda s: pd.Series(s).rank(method="min", ascending=False).iloc[-1], raw=False)
        )
        giant_bear = (
            (out["close"] < out["open"])
            & ((out["close"] / out["close"].shift(1) - 1.0) < -0.03)
            & (body_ratio > 0.40)
            & ((vol_rank30 <= 3.0) | ((out["volume"] / prev_vol.replace(0, np.nan)) >= 2.0))
        )
        out["no_giant_bear_30"] = ~giant_bear.rolling(30, min_periods=1).max().fillna(0.0).astype(bool)

    return out


def build_today_features_from_feature_df(x: pd.DataFrame):
    if x is None or x.empty or len(x) < 25:
        return None

    work = x.copy().reset_index(drop=True)
    required_cols = {"open", "high", "low", "close", "volume", "trend_line", "long_line", "J"}
    if not required_cols.issubset(work.columns):
        return None

    if "n_up_any" not in work.columns or "keyk_support_active" not in work.columns:
        work = add_structure_tags(work)
    else:
        work = ensure_pin_aux_cols(work)

    trend_line = float(work["trend_line"].iloc[-1])
    long_line = float(work["long_line"].iloc[-1])
    if not np.isfinite(trend_line) or not np.isfinite(long_line):
        return None

    close = work["close"].astype(float)
    volume = work["volume"].astype(float)

    prev_trend_3 = float(work["trend_line"].iloc[-4]) if len(work) >= 4 else np.nan
    prev_trend_5 = float(work["trend_line"].iloc[-6]) if len(work) >= 6 else np.nan
    prev_long_5 = float(work["long_line"].iloc[-6]) if len(work) >= 6 else np.nan

    today_open = float(work["open"].iloc[-1])
    today_close = float(work["close"].iloc[-1])
    today_high = float(work["high"].iloc[-1])
    today_low = float(work["low"].iloc[-1])
    full_range = today_high - today_low
    body_low = min(today_open, today_close)

    return {
        "trend_line": trend_line,
        "long_line": long_line,
        "trend_slope_3": safe_div(trend_line, prev_trend_3) - 1.0,
        "trend_slope_5": safe_div(trend_line, prev_trend_5) - 1.0,
        "long_slope_5": safe_div(long_line, prev_long_5) - 1.0,
        "trend_line_lead": safe_div(trend_line - long_line, today_close),
        "ret10": safe_div(close.iloc[-1], close.iloc[-11]) - 1.0 if len(close) >= 11 else np.nan,
        "ret3": safe_div(close.iloc[-1], close.iloc[-4]) - 1.0 if len(close) >= 4 else np.nan,
        "signal_vs_ma20": safe_div(float(volume.iloc[-1]), float(volume.tail(20).mean())),
        "vol_vs_prev": safe_div(float(volume.iloc[-1]), float(volume.iloc[-2])) if len(volume) >= 2 else np.nan,
        "close_position": safe_div(today_close - today_low, full_range),
        "lower_shadow_ratio": safe_div(body_low - today_low, full_range),
        "n_up_any": bool(work["n_up_any"].iloc[-1]),
        "keyk_support_active": bool(work["keyk_support_active"].iloc[-1]),
        "along_trend_up": bool(work["along_trend_up"].iloc[-1]),
        "no_giant_bear_30": bool(work["no_giant_bear_30"].iloc[-1]),
    }


def build_today_features(df):
    df = technical_indicators.calculate_trend(df)
    df = technical_indicators.calculate_kdj(df)
    if len(df) < 25:
        return None

    x = pd.DataFrame(
        {
            "open": df["开盘"].astype(float),
            "high": df["最高"].astype(float),
            "low": df["最低"].astype(float),
            "close": df["收盘"].astype(float),
            "volume": df["成交量"].astype(float),
            "trend_line": df["知行短期趋势线"].astype(float),
            "long_line": df["知行多空线"].astype(float),
            "J": df["J"].astype(float),
        }
    ).reset_index(drop=True)
    x = add_structure_tags(x)
    return build_today_features_from_feature_df(x)


def subtype_a_ok(feat):
    return (
        np.isfinite(feat["trend_slope_5"]) and feat["trend_slope_5"] >= 0.03
        and np.isfinite(feat["trend_line_lead"]) and feat["trend_line_lead"] >= 0.03
        and np.isfinite(feat["ret10"]) and feat["ret10"] >= 0.05
        and np.isfinite(feat["ret3"]) and feat["ret3"] <= 0.01
        and np.isfinite(feat["signal_vs_ma20"]) and feat["signal_vs_ma20"] <= 1.0
        and np.isfinite(feat["vol_vs_prev"]) and feat["vol_vs_prev"] <= 1.1
        and np.isfinite(feat["lower_shadow_ratio"]) and feat["lower_shadow_ratio"] <= 0.25
    )


def subtype_b_ok(feat):
    return (
        np.isfinite(feat["trend_slope_3"]) and feat["trend_slope_3"] >= 0.02
        and np.isfinite(feat["trend_slope_5"]) and feat["trend_slope_5"] >= 0.04
        and np.isfinite(feat["long_slope_5"]) and feat["long_slope_5"] >= 0.0
        and np.isfinite(feat["trend_line_lead"]) and feat["trend_line_lead"] >= 0.02
        and np.isfinite(feat["ret10"]) and feat["ret10"] >= 0.10
        and np.isfinite(feat["ret3"]) and feat["ret3"] <= 0.05
        and np.isfinite(feat["signal_vs_ma20"]) and 0.90 <= feat["signal_vs_ma20"] <= 1.50
        and np.isfinite(feat["vol_vs_prev"]) and feat["vol_vs_prev"] <= 1.50
        and np.isfinite(feat["close_position"]) and feat["close_position"] <= 0.20
        and np.isfinite(feat["lower_shadow_ratio"]) and feat["lower_shadow_ratio"] <= 0.10
    )


def subtype_c_ok(feat):
    return (
        (feat["along_trend_up"] or feat["n_up_any"] or feat["keyk_support_active"])
        and np.isfinite(feat["trend_slope_5"]) and feat["trend_slope_5"] >= 0.015
        and np.isfinite(feat["trend_line_lead"]) and feat["trend_line_lead"] >= 0.02
        and np.isfinite(feat["ret10"]) and feat["ret10"] >= -0.04
        and np.isfinite(feat["ret3"]) and feat["ret3"] <= 0.01
        and np.isfinite(feat["signal_vs_ma20"]) and feat["signal_vs_ma20"] <= 1.0
        and np.isfinite(feat["vol_vs_prev"]) and feat["vol_vs_prev"] <= 1.1
        and np.isfinite(feat["close_position"]) and feat["close_position"] <= 0.30
        and np.isfinite(feat["lower_shadow_ratio"]) and feat["lower_shadow_ratio"] <= 0.25
    )


def detect_subtypes(feat):
    matched = []
    if subtype_a_ok(feat):
        matched.append("A型(缩量回踩)")
    if subtype_b_ok(feat):
        matched.append("B型(强趋势加速)")
    if subtype_c_ok(feat):
        matched.append("C型(结构支撑)")
    return matched


def build_recommendation_order(matched_subtypes):
    ordered = sorted(
        matched_subtypes,
        key=lambda x: (TYPE_PRIORITY.get(x, 999), x),
    )
    return " > ".join(f"{name}({TYPE_RECOMMEND.get(name, '未知')})" for name in ordered)


def check(file_path, feature_cache=None):
    """
    当前单针简化版买点：
    1. 保持周线条件
    2. 趋势线 > 多空线
    3. 满足单针定义：长期 >= 85，短期 <= 30
    4. 当日成交量 < 5日均量
    """
    if feature_cache is not None:
        weekly_ok, weekly_reason = feature_cache.weekly_screen()
    else:
        weekly_ok, weekly_reason = b1filter.weekly_screen(str(file_path))
    if not weekly_ok:
        return [-1]

    if feature_cache is not None:
        df = feature_cache.daily_cn_df()
        if df is None or len(df) < 40:
            return [-1]
        feat = feature_cache.pin_today_features()
    else:
        df, load_error = stoploss.load_data(file_path)
        if load_error or df is None or len(df) < 40:
            return [-1]
        feat = build_today_features(df)
    if feat is None:
        return [-1]
    if feat["trend_line"] <= feat["long_line"]:
        return [-1]

    if not technical_indicators.caculate_pin(df, short_threshold=30, long_threshold=85):
        return [-1]

    vol_ma5 = pd.to_numeric(df["成交量"], errors="coerce").rolling(5, min_periods=5).mean().iloc[-1]
    today_vol = float(pd.to_numeric(df["成交量"], errors="coerce").iloc[-1])
    if not np.isfinite(vol_ma5) or not np.isfinite(today_vol) or not (today_vol < vol_ma5):
        return [-1]

    daily_reason = "周线通过+趋势线强于多空线+单针(长期>=85且短期<=30)+当日量小于5日均量"
    risk_note = format_risk_note(feature_cache.risk_snapshot()) if feature_cache is not None else ""
    note = f"周线：{weekly_reason} | 日线：{daily_reason}"
    if risk_note:
        note = f"{note} | {risk_note}"
    return [
        1,
        daily_reason,
        "固定条件版",
        note,
    ]
