from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd

from utils import stoploss, technical_indicators
from utils.market_risk_tags import format_risk_note


EPS = 1e-12
J_RANK_WINDOW = 20
J_RANK_MAX = 0.10
B1_ABSOLUTE_J_MAX = 13.0
B1_CONFIRM_RET1_MAX = 0.02
WEEKLY_J_RANK_WINDOW = 20
WEEKLY_J_RANK_MAX = 0.10
WEEKLY_SLOPE_WEEKS = 3
PRICE_RANGE_DOWN = 3.0
PRICE_RANGE_UP = 2.5
LINE_PULLBACK_TOLERANCE = 2.0
YANG_OVER_YIN_RATIO = 1.382
STOP_LOSS_RATIO = 0.97


def _stock_code(file_path: str) -> str:
    return Path(file_path).stem.split("#")[-1]


def _within_pct_range(price: float, anchor: float, down_pct: float, up_pct: float) -> bool:
    if anchor is None or pd.isna(anchor) or abs(anchor) < 1e-9:
        return False
    change_pct = (price - anchor) / anchor * 100
    return (-down_pct) <= change_pct <= up_pct


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


def _series_slope(series: pd.Series, lookback: int) -> float:
    if len(series) <= lookback:
        return np.nan
    current = float(series.iloc[-1])
    prev = float(series.iloc[-(lookback + 1)])
    if not np.isfinite(current) or not np.isfinite(prev) or abs(prev) < EPS:
        return np.nan
    return current / prev - 1.0


@lru_cache(maxsize=8192)
def _weekly_screen(file_path: str) -> tuple[bool, str]:
    weekly_df = technical_indicators.calculate_week_price(file_path)
    return _evaluate_weekly_screen_df(weekly_df)


def _evaluate_weekly_screen_df(weekly_df: pd.DataFrame) -> tuple[bool, str]:
    if weekly_df is None or weekly_df.empty or len(weekly_df) < WEEKLY_SLOPE_WEEKS + 1:
        return False, ""

    weekly_df = technical_indicators.calculate_trend(weekly_df.copy())
    weekly_df = technical_indicators.calculate_kdj(weekly_df)
    weekly_df = technical_indicators.calculate_daily_ma(weekly_df)

    last = weekly_df.iloc[-1]
    trend_line = float(last["知行短期趋势线"]) if pd.notna(last["知行短期趋势线"]) else np.nan
    long_line = float(last["知行多空线"]) if pd.notna(last["知行多空线"]) else np.nan
    close_price = float(last["收盘"]) if pd.notna(last["收盘"]) else np.nan

    ma20_slope = _series_slope(weekly_df["MA20"], WEEKLY_SLOPE_WEEKS)
    ma30_slope = _series_slope(weekly_df["MA30"], WEEKLY_SLOPE_WEEKS)
    ma60_slope = _series_slope(weekly_df["MA60"], WEEKLY_SLOPE_WEEKS)

    if not np.isfinite(trend_line) or not np.isfinite(long_line) or not np.isfinite(close_price):
        return False, ""
    if not np.isfinite(ma20_slope) or not np.isfinite(ma30_slope):
        return False, ""
    if trend_line <= long_line:
        return False, ""
    if ma20_slope <= 0 or ma30_slope <= 0:
        return False, ""

    weekly_reason = ""
    if close_price >= trend_line:
        weekly_reason = "周线强势"
    elif close_price < trend_line and close_price >= long_line:
        weekly_reason = "周线进碗"
    else:
        return False, ""

    extras = []
    weekly_j_rank20 = rolling_last_percentile(weekly_df["J"], WEEKLY_J_RANK_WINDOW)
    current_weekly_j_rank20 = (
        float(weekly_j_rank20.iloc[-1])
        if len(weekly_j_rank20) > 0 and pd.notna(weekly_j_rank20.iloc[-1])
        else np.nan
    )
    if np.isfinite(current_weekly_j_rank20) and current_weekly_j_rank20 < WEEKLY_J_RANK_MAX:
        extras.append("周J低位")
    if np.isfinite(ma60_slope) and ma60_slope > 0:
        extras.append("周MA60上行")

    if extras:
        return True, f"{weekly_reason}+{'+'.join(extras)}"
    return True, weekly_reason


def weekly_screen(file_path: str) -> tuple[bool, str]:
    """供其他策略复用的周线筛选。"""
    return _weekly_screen(file_path)


def weekly_screen_from_weekly_df(weekly_df: pd.DataFrame) -> tuple[bool, str]:
    return _evaluate_weekly_screen_df(weekly_df)


def weekly_screen_from_daily_df(daily_df: pd.DataFrame) -> tuple[bool, str]:
    weekly_df = technical_indicators.calculate_week_price_from_df(daily_df)
    return _evaluate_weekly_screen_df(weekly_df)


def build_weekly_screen_history_from_daily_df(daily_df: pd.DataFrame) -> pd.DataFrame:
    weekly_df = technical_indicators.calculate_week_price_from_df(daily_df)
    if weekly_df is None or weekly_df.empty or len(weekly_df) < WEEKLY_SLOPE_WEEKS + 1:
        return pd.DataFrame(columns=["week_end", "weekly_ok", "weekly_reason"])

    weekly_df = technical_indicators.calculate_trend(weekly_df.copy())
    weekly_df = technical_indicators.calculate_kdj(weekly_df)
    weekly_df = technical_indicators.calculate_daily_ma(weekly_df)

    rows = []
    for idx in range(len(weekly_df)):
        sub = weekly_df.iloc[: idx + 1].copy()
        ok, reason = _evaluate_weekly_screen_df(sub)
        rows.append(
            {
                "week_end": pd.Timestamp(sub["日期"].iloc[-1]),
                "weekly_ok": bool(ok),
                "weekly_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def map_weekly_screen_to_daily_df(daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df is None or daily_df.empty:
        return pd.DataFrame(columns=["日期", "weekly_ok", "weekly_reason"])
    base = daily_df[["日期"]].copy().sort_values("日期").reset_index(drop=True)
    weekly_hist = build_weekly_screen_history_from_daily_df(daily_df)
    if weekly_hist.empty:
        base["weekly_ok"] = False
        base["weekly_reason"] = ""
        return base
    mapped = pd.merge_asof(
        base,
        weekly_hist.sort_values("week_end"),
        left_on="日期",
        right_on="week_end",
        direction="backward",
    )
    mapped["weekly_ok"] = mapped["weekly_ok"].fillna(False).astype(bool)
    mapped["weekly_reason"] = mapped["weekly_reason"].fillna("")
    return mapped[["日期", "weekly_ok", "weekly_reason"]]


def _local_extrema_indices(values: np.ndarray, order: int, mode: str) -> np.ndarray:
    if len(values) < 2 * order + 1:
        return np.array([], dtype=int)

    indices = []
    for idx in range(order, len(values) - order):
        center = values[idx]
        left = values[idx - order: idx]
        right = values[idx + 1: idx + order + 1]
        neighbors = np.concatenate([left, right])
        if np.isnan(center) or np.isnan(neighbors).any():
            continue
        if mode == "max" and np.all(center > neighbors):
            indices.append(idx)
        if mode == "min" and np.all(center < neighbors):
            indices.append(idx)
    return np.array(indices, dtype=int)


def get_last_n_high(
    df: pd.DataFrame,
    order: int = 5,
    ab_period: tuple[int, int] = (3, 15),
    ab_retracement: tuple[float, float] = (0.03, 0.2),
    bc_period: tuple[int, int] = (3, 15),
    ac_deviation: float = 0.05,
):
    """获取最近一个正N结构的C点价格。"""
    required_cols = {"日期", "最高", "最低"}
    if not required_cols.issubset(df.columns):
        return None

    work = df.sort_values("日期").reset_index(drop=True)
    highs_idx = _local_extrema_indices(work["最高"].to_numpy(dtype=float), order=order, mode="max")
    lows_idx = _local_extrema_indices(work["最低"].to_numpy(dtype=float), order=order, mode="min")

    if len(highs_idx) == 0 or len(lows_idx) == 0:
        return None

    highs = [(int(i), float(work.at[i, "最高"])) for i in highs_idx]
    lows = [(int(i), float(work.at[i, "最低"])) for i in lows_idx]

    for a_pos, a_price in reversed(highs):
        candidate_bs = [
            (l_pos, l_price)
            for l_pos, l_price in lows
            if l_pos > a_pos
            and ab_period[0] <= l_pos - a_pos <= ab_period[1]
            and ab_retracement[0] <= (a_price - l_price) / a_price <= ab_retracement[1]
        ]
        for b_pos, b_price in candidate_bs:
            candidate_cs = [
                (h_pos, h_price)
                for h_pos, h_price in highs
                if h_pos > b_pos
                and bc_period[0] <= h_pos - b_pos <= bc_period[1]
                and h_price > b_price
                and abs(h_price - a_price) / a_price <= ac_deviation
            ]
            if candidate_cs:
                return max(candidate_cs, key=lambda x: x[0])[1]
    return None


def _is_low_volume_low_price(df: pd.DataFrame) -> bool:
    for period in (5, 10, 20, 30):
        recent_volume = df["成交量"].iloc[-(period + 1):-1] if len(df) >= period + 1 else df["成交量"].iloc[:-1]
        recent_close = df["收盘"].iloc[-(period + 1):-1] if len(df) >= period + 1 else df["收盘"].iloc[:-1]
        if recent_volume.empty or recent_close.empty:
            continue
        if float(df["成交量"].iloc[-1]) < float(recent_volume.min()) and float(df["收盘"].iloc[-1]) < float(recent_close.min()):
            return True
    return False


def _pullback_to_key_lines(df: pd.DataFrame, df_trend: pd.DataFrame, df_ma: pd.DataFrame) -> tuple[bool, bool]:
    today_low = float(df["最低"].iloc[-1])
    ma_pullback = any(
        _within_pct_range(today_low, float(df_ma[col].iloc[-1]), LINE_PULLBACK_TOLERANCE, LINE_PULLBACK_TOLERANCE)
        for col in ("MA20", "MA60")
    )
    trend_pullback = any(
        _within_pct_range(today_low, float(df_trend[col].iloc[-1]), LINE_PULLBACK_TOLERANCE, LINE_PULLBACK_TOLERANCE)
        for col in ("知行多空线", "知行短期趋势线")
    )
    return ma_pullback, trend_pullback


def _is_sb1(today: pd.Series, yesterday: pd.Series, df_kdj: pd.DataFrame) -> bool:
    return (
        float(today["收盘"]) < float(today["开盘"]) < float(yesterday["收盘"])
        and float(df_kdj["J"].iloc[-1]) < 0
        and float(df_kdj["J"].iloc[-2]) < 0
    )


def _recent_max_volume_is_bullish(df: pd.DataFrame) -> bool:
    recent = df.tail(60)
    if recent.empty:
        return False
    top_row = recent.sort_values("成交量", ascending=False).head(1)
    if top_row.empty:
        return False
    return bool((top_row["收盘"] > top_row["开盘"]).all())


def _bullish_volume_dominance(df: pd.DataFrame) -> bool:
    recent = df.tail(60).copy()
    recent["阳线成交量"] = np.where(recent["收盘"] > recent["开盘"], recent["成交量"], 0.0)
    recent["阴线成交量"] = np.where(recent["收盘"] < recent["开盘"], recent["成交量"], 0.0)
    yang_sum = recent["阳线成交量"].tail(30).sum()
    yin_sum = recent["阴线成交量"].tail(30).sum()
    if yin_sum <= 1e-9:
        return yang_sum > 0
    return yang_sum > yin_sum * YANG_OVER_YIN_RATIO


def _first_pullback_after_cross(df: pd.DataFrame, df_trend: pd.DataFrame) -> bool:
    trend_line = df_trend["知行短期趋势线"]
    long_line = df_trend["知行多空线"]
    cross_idx = None
    for idx in range(1, len(df)):
        if trend_line.iloc[idx - 1] < long_line.iloc[idx - 1] and trend_line.iloc[idx] > long_line.iloc[idx]:
            cross_idx = idx
    if cross_idx is None:
        return False

    seen_pullback = False
    for idx in range(cross_idx + 1, len(df)):
        low_price = float(df["最低"].iloc[idx])
        near_long = _within_pct_range(low_price, float(long_line.iloc[idx]), LINE_PULLBACK_TOLERANCE, LINE_PULLBACK_TOLERANCE)
        near_trend = _within_pct_range(low_price, float(trend_line.iloc[idx]), LINE_PULLBACK_TOLERANCE, LINE_PULLBACK_TOLERANCE)
        if not (near_long or near_trend):
            continue
        if idx == len(df) - 1 and not seen_pullback:
            return True
        seen_pullback = True
    return False


def _has_gap_up_followed_by_big_bullish(df: pd.DataFrame, df_trend: pd.DataFrame) -> bool:
    recent = df.tail(20).reset_index(drop=True)
    recent_trend = df_trend.tail(20).reset_index(drop=True)
    if len(recent) < 3:
        return False

    for idx in range(1, len(recent)):
        if float(recent.at[idx, "收盘"]) <= float(recent.at[idx - 1, "最高"]):
            continue
        if float(recent_trend.at[idx, "知行多空线"]) <= float(recent_trend.at[idx, "知行短期趋势线"]):
            continue

        gap_close = float(recent.at[idx, "收盘"])
        if not bool((recent.loc[idx:, "收盘"] >= gap_close).all()):
            continue

        for jdx in range(idx + 1, len(recent)):
            prev_close = float(recent.at[jdx - 1, "收盘"])
            current_close = float(recent.at[jdx, "收盘"])
            prev_volume = float(recent.at[jdx - 1, "成交量"])
            current_volume = float(recent.at[jdx, "成交量"])
            if prev_close <= 0 or prev_volume <= 0:
                continue
            change_pct = (current_close - prev_close) / prev_close * 100
            volume_ratio = current_volume / prev_volume
            if change_pct >= 8 and volume_ratio >= 3:
                return True
    return False


def _has_long_negative_short_volume(df: pd.DataFrame) -> bool:
    if len(df) < 30:
        return False
    current = df.iloc[-1]
    prev = df.iloc[-2]
    prev_diff_pct = abs((float(prev["收盘"]) - float(prev["开盘"])) / float(prev["开盘"])) * 100
    current_diff_pct = abs((float(current["收盘"]) - float(current["开盘"])) / float(current["开盘"])) * 100
    max_30_volume = float(df["成交量"].tail(30).max())
    return (
        prev_diff_pct < 2
        and current_diff_pct > 4
        and float(current["成交量"]) < float(prev["成交量"])
        and float(current["成交量"]) < max_30_volume / 2
    )


def _filter_reason(
    sb1: bool,
    ma_pullback: bool,
    trend_pullback: bool,
    first_pullback: bool,
    low_volume_low_price: bool,
    gap_up: bool,
    long_negative: bool,
    declining_volume: bool,
    consistent: bool,
) -> str:
    if sb1 and consistent:
        return "SB1条件"
    if ma_pullback:
        return "回踩均线"
    if trend_pullback:
        return "回踩趋势线"
    if first_pullback:
        return "第一次回踩"
    if low_volume_low_price:
        return "地量低价"
    if gap_up:
        return "跳空K线"
    if long_negative:
        return "长阴短柱"
    if declining_volume and consistent:
        return "缩量且涨幅一致"
    return "其他原因"


def _compute_old_b1_passed(df: pd.DataFrame, df_trend: pd.DataFrame, df_kdj: pd.DataFrame, df_ma: pd.DataFrame) -> pd.Series:
    n = len(df)
    trend_line = df_trend["知行短期趋势线"]
    long_line = df_trend["知行多空线"]
    j_rank20 = rolling_last_percentile(df_kdj["J"], J_RANK_WINDOW)

    passed = pd.Series(False, index=df.index)
    for i in range(1, n):
        today_idx = i
        yesterday_idx = i - 1

        lt = float(trend_line.iloc[today_idx])
        ll = float(long_line.iloc[today_idx])
        if not np.isfinite(lt) or not np.isfinite(ll) or ll > lt:
            continue
        j_rank = float(j_rank20.iloc[today_idx]) if pd.notna(j_rank20.iloc[today_idx]) else np.nan
        if not np.isfinite(j_rank) or j_rank >= J_RANK_MAX:
            continue

        today_row = df.iloc[today_idx]
        yesterday_row = df.iloc[yesterday_idx]

        today_close = float(today_row["收盘"])
        today_open = float(today_row["开盘"])
        yesterday_close = float(yesterday_row["收盘"])
        today_low = float(today_row["最低"])

        decl_vol = float(df["成交量"].iloc[today_idx]) < float(df["成交量"].iloc[yesterday_idx])
        consistent = _within_pct_range(today_close, yesterday_close, PRICE_RANGE_DOWN, PRICE_RANGE_UP)
        lvp = _is_low_volume_low_price(df.iloc[:i + 1])
        ma_pb, trend_pb = _pullback_to_key_lines(df.iloc[:i + 1], df_trend.iloc[:i + 1], df_ma.iloc[:i + 1])
        sb = _is_sb1(today_row, yesterday_row, df_kdj.iloc[:i + 1])
        all_bull = _recent_max_volume_is_bullish(df.iloc[:i + 1])
        bull_vol = _bullish_volume_dominance(df.iloc[:i + 1])
        first_pb = _first_pullback_after_cross(df.iloc[:i + 1], df_trend.iloc[:i + 1])
        gap_up = _has_gap_up_followed_by_big_bullish(df.iloc[:i + 1], df_trend.iloc[:i + 1])
        long_neg = _has_long_negative_short_volume(df.iloc[:i + 1])

        trigger = ma_pb or trend_pb or first_pb or lvp or gap_up or long_neg
        route_a = sb and all_bull and bull_vol
        route_b = trigger and consistent and decl_vol and all_bull and bull_vol
        passed.iat[today_idx] = route_a or route_b

    return passed


def _compute_today_confirm(df: pd.DataFrame, df_kdj: pd.DataFrame) -> pd.Series:
    j_rank20 = rolling_last_percentile(df_kdj["J"], J_RANK_WINDOW)
    confirm = pd.Series(False, index=df.index)
    close_col = "收盘" if "收盘" in df.columns else "close"
    open_col = "开盘" if "开盘" in df.columns else "open"
    for i in range(1, len(df)):
        today_idx = i
        j_rank = float(j_rank20.iloc[today_idx]) if pd.notna(j_rank20.iloc[today_idx]) else np.nan
        j_abs = float(df_kdj["J"].iloc[today_idx])
        today_close = float(df[close_col].iloc[today_idx])
        today_open = float(df[open_col].iloc[today_idx])
        today_ret1 = (
            float(df["ret1"].iloc[today_idx])
            if "ret1" in df.columns and pd.notna(df["ret1"].iloc[today_idx])
            else (today_close / float(df[close_col].iloc[today_idx - 1]) - 1)
        )

        j_low = (np.isfinite(j_rank) and j_rank < J_RANK_MAX) or (np.isfinite(j_abs) and j_abs < B1_ABSOLUTE_J_MAX)
        bull_close = today_close > today_open
        small_ret = today_ret1 < B1_CONFIRM_RET1_MAX
        confirm.iat[today_idx] = j_low and bull_close and small_ret
    return confirm


def check(file_path, hold_list, feature_cache=None):
    del hold_list

    if feature_cache is not None:
        bundle = feature_cache.b1_daily_bundle()
        if bundle is None:
            return [-1]
        df = bundle["df"].copy()
        df_trend = bundle["df_trend"].copy()
        df_kdj = bundle["df_kdj"].copy()
        df_ma = bundle["df_ma"].copy()
        weekly_ok, weekly_reason = feature_cache.weekly_screen()
    else:
        df, load_error = stoploss.load_data(file_path)
        if load_error or df is None or len(df) < 120:
            return [-1]
        weekly_ok, weekly_reason = weekly_screen(file_path)
        df_trend = technical_indicators.calculate_trend(df.copy())
        df_kdj = technical_indicators.calculate_kdj(df.copy())
        df_ma = technical_indicators.calculate_daily_ma(df.copy())

    if not weekly_ok:
        return [-1]

    stock_code = _stock_code(file_path)

    old_b1_passed = _compute_old_b1_passed(df, df_trend, df_kdj, df_ma)
    today_confirm = _compute_today_confirm(df, df_kdj)

    prev_b1_signal = old_b1_passed.shift(1).fillna(False)
    final_b1_today = prev_b1_signal & today_confirm

    today_idx = len(df) - 1
    if not bool(final_b1_today.iloc[today_idx]):
        return [-1]

    today = df.iloc[today_idx]
    yesterday = df.iloc[today_idx - 1]

    today_close = float(today["收盘"])
    today_low = float(today["最低"])
    yesterday_low = float(yesterday["最低"])

    stop_loss_price = np.round(min(yesterday_low, today_low) * STOP_LOSS_RATIO, 1)
    near_high_price = get_last_n_high(df)
    if near_high_price and near_high_price > stop_loss_price and today_close > stop_loss_price:
        ratio = np.round((near_high_price - today_close) / (today_close - stop_loss_price), 1)
    else:
        ratio = "请人工判断盈亏比！"

    prev_reason = _filter_reason_for_row(
        df, df_trend, df_kdj, df_ma, today_idx - 1
    )
    today_confirm_reason = "低位阳线确认" if today_confirm.iloc[today_idx] else ""
    daily_reason = f"昨日：{prev_reason} | 今日：{today_confirm_reason}"
    risk_note = format_risk_note(feature_cache.risk_snapshot()) if feature_cache is not None else ""
    filter_reason = f"周线：{weekly_reason} | 日线：{daily_reason}"
    if risk_note:
        filter_reason = f"{filter_reason} | {risk_note}"

    return [1, stop_loss_price, today_close, ratio, filter_reason]


def _filter_reason_for_row(df: pd.DataFrame, df_trend: pd.DataFrame, df_kdj: pd.DataFrame, df_ma: pd.DataFrame, idx: int) -> str:
    if idx < 1:
        return "无信号"

    today_row = df.iloc[idx]
    yesterday_row = df.iloc[idx - 1]

    today_close = float(today_row["收盘"])
    today_open = float(today_row["开盘"])
    yesterday_close = float(yesterday_row["收盘"])

    decl_vol = float(df["成交量"].iloc[idx]) < float(df["成交量"].iloc[idx - 1])
    consistent = _within_pct_range(today_close, yesterday_close, PRICE_RANGE_DOWN, PRICE_RANGE_UP)
    lvp = _is_low_volume_low_price(df.iloc[:idx + 1])
    ma_pb, trend_pb = _pullback_to_key_lines(df.iloc[:idx + 1], df_trend.iloc[:idx + 1], df_ma.iloc[:idx + 1])
    sb = _is_sb1(today_row, yesterday_row, df_kdj.iloc[:idx + 1])
    all_bull = _recent_max_volume_is_bullish(df.iloc[:idx + 1])
    bull_vol = _bullish_volume_dominance(df.iloc[:idx + 1])
    first_pb = _first_pullback_after_cross(df.iloc[:idx + 1], df_trend.iloc[:idx + 1])
    gap_up = _has_gap_up_followed_by_big_bullish(df.iloc[:idx + 1], df_trend.iloc[:idx + 1])
    long_neg = _has_long_negative_short_volume(df.iloc[:idx + 1])

    trigger = ma_pb or trend_pb or first_pb or lvp or gap_up or long_neg

    if sb and all_bull and bull_vol:
        return "SB1条件"
    if ma_pb:
        return "回踩均线"
    if trend_pb:
        return "回踩趋势线"
    if first_pb:
        return "第一次回踩"
    if lvp:
        return "地量低价"
    if gap_up:
        return "跳空K线"
    if long_neg:
        return "长阴短柱"
    if decl_vol and consistent:
        return "缩量且涨幅一致"
    return "其他原因"
