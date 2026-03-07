from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


FACTOR_COLUMNS = [
    "low_volume_pullback_factor",
    "price_amplitude_factor",
    "staged_volume_burst_factor",
    "pullback_confirmation_factor",
    "long_bear_short_volume_factor",
    "daily_ma_bull_factor",
    "macd_dif_factor",
    "rsi_bull_factor",
    "key_k_support_factor",
    "price_sideways_10_factor",
    "price_sideways_factor",
    "j_decline_acceleration_factor",
]

PENALTY_COLUMNS = [
    "bearish_volume_penalty",
    "break_trend_penalty",
    "break_bull_bear_penalty",
    "bull_bear_above_trend_penalty",
    "bearish_candle_dominance_penalty",
    "extreme_bull_run_penalty",
    "high_overbought_penalty",
    "abnormal_amplitude_penalty",
    "volume_stagnation_penalty",
    "key_k_close_break_penalty",
    "key_k_low_break_penalty",
]

FACTOR_NAME_MAP = {
    "low_volume_pullback_factor": "缩量因子",
    "price_amplitude_factor": "价格波动幅度因子",
    "staged_volume_burst_factor": "阶段性放量因子",
    "pullback_confirmation_factor": "回踩趋势线/多空线确认因子",
    "long_bear_short_volume_factor": "长阴短柱因子",
    "daily_ma_bull_factor": "日线均线多头结构因子",
    "macd_dif_factor": "MACD DIF因子",
    "rsi_bull_factor": "RSI多头结构因子",
    "key_k_support_factor": "关键K支撑因子",
    "price_sideways_10_factor": "十日横盘因子",
    "price_sideways_factor": "二十日横盘因子",
    "j_decline_acceleration_factor": "J值快速下行因子",
    "bearish_volume_penalty": "放量阴线扣分",
    "break_trend_penalty": "跌破趋势线扣分",
    "break_bull_bear_penalty": "跌破多空线扣分",
    "bull_bear_above_trend_penalty": "多空线高于趋势线扣分",
    "bearish_candle_dominance_penalty": "三十日阴线显著偏多扣分",
    "extreme_bull_run_penalty": "三十日连续大阳线过热扣分",
    "high_overbought_penalty": "高位超买扣分",
    "abnormal_amplitude_penalty": "异常振幅扣分",
    "volume_stagnation_penalty": "放量滞涨扣分",
    "key_k_close_break_penalty": "跌破关键K收盘扣分",
    "key_k_low_break_penalty": "跌破关键K最低价重扣分",
}


@dataclass
class ResearchConfig:
    success_return_threshold: float = 0.0
    burst_window: int = 20
    top_quantile: float = 0.30


def _calc_trend(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["trend_line"] = out["close"].ewm(span=10, adjust=False).mean()
    out["trend_line"] = out["trend_line"].ewm(span=10, adjust=False).mean()
    out["MA14"] = out["close"].rolling(window=14).mean()
    out["MA28"] = out["close"].rolling(window=28).mean()
    out["MA57"] = out["close"].rolling(window=57).mean()
    out["MA114"] = out["close"].rolling(window=114).mean()
    out["bull_bear_line"] = (out["MA14"] + out["MA28"] + out["MA57"] + out["MA114"]) / 4.0
    return out


def _calc_kdj(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    low_9 = out["low"].rolling(window=9, min_periods=1).min()
    high_9 = out["high"].rolling(window=9, min_periods=1).max()
    rsv = (out["close"] - low_9) / (high_9 - low_9 + 1e-6) * 100.0
    out["K"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    out["D"] = out["K"].ewm(alpha=1 / 3, adjust=False).mean()
    out["J"] = 3.0 * out["K"] - 2.0 * out["D"]
    return out


def _calc_mas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for period in [10, 20, 30, 60]:
        out[f"MA{period}"] = out["close"].rolling(window=period).mean()
    return out


def _calc_macd(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    out["DIF"] = ema12 - ema26
    return out


def _calc_rsi(df: pd.DataFrame, period: int) -> pd.Series:
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100.0 - 100.0 / (1.0 + rs)


def _calc_key_k_levels(df: pd.DataFrame, window: int = 60) -> tuple[np.ndarray, np.ndarray]:
    bullish_volume = np.where((df["close"] > df["open"]).to_numpy(dtype=bool), df["volume"].to_numpy(dtype=float), -np.inf)
    closes = df["close"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    key_close = np.full(len(df), np.nan, dtype=float)
    key_low = np.full(len(df), np.nan, dtype=float)

    from collections import deque

    dq: deque[int] = deque()
    for idx in range(len(df)):
        left = idx - window + 1
        while dq and dq[0] < left:
            dq.popleft()
        current_volume = bullish_volume[idx]
        if np.isfinite(current_volume):
            while dq and bullish_volume[dq[-1]] <= current_volume:
                dq.pop()
            dq.append(idx)
        if dq:
            key_idx = dq[0]
            key_close[idx] = closes[key_idx]
            key_low[idx] = lows[key_idx]
    return key_close, key_low


def _calc_event_streak(event: pd.Series) -> pd.Series:
    streak = np.zeros(len(event), dtype=float)
    values = event.to_numpy(dtype=bool)
    for idx, flag in enumerate(values):
        streak[idx] = streak[idx - 1] + 1.0 if flag and idx > 0 else float(flag)
    return pd.Series(streak, index=event.index, dtype=float)


def prepare_factor_frame(df: pd.DataFrame, burst_window: int = 20) -> pd.DataFrame:
    out = _calc_trend(df)
    out = _calc_kdj(out)
    out = _calc_mas(out)
    out = _calc_macd(out)
    out["prev_close"] = out["close"].shift(1)
    out["return_pct"] = out["close"] / out["prev_close"] - 1.0
    out["amplitude_pct"] = (out["high"] - out["low"]) / out["prev_close"].replace(0, np.nan)
    out["is_bullish"] = out["close"] > out["open"]
    out["is_bearish"] = out["close"] < out["open"]
    bullish_count_30 = out["is_bullish"].rolling(window=30, min_periods=30).sum()
    bearish_count_30 = out["is_bearish"].rolling(window=30, min_periods=30).sum()
    key_k_close, key_k_low = _calc_key_k_levels(out, window=60)
    out["关键K收盘价"] = key_k_close
    out["关键K最低价"] = key_k_low
    out["volume_rank_60"] = out["volume"].rolling(window=60, min_periods=60).rank(method="min", ascending=False)
    out["large_volume_bull"] = (out["is_bullish"] & (out["volume_rank_60"] <= 3)).astype(bool)

    low_volume_strength = pd.Series(0.0, index=out.index, dtype=float)
    for lag in (1, 2, 3):
        trigger = out["large_volume_bull"].shift(lag).eq(True)
        trigger_volume = out["volume"].shift(lag)
        cond = out["is_bearish"] & trigger & (out["volume"] <= trigger_volume * 0.5)
        low_volume_strength = low_volume_strength + cond.astype(float)
    out["low_volume_pullback_factor"] = low_volume_strength.clip(upper=3.0) / 3.0

    board_main = out["board"].eq("MAIN")
    board_growth = out["board"].isin(["GEM", "STAR"])
    main_cond = board_main & out["return_pct"].gt(-0.035) & out["return_pct"].lt(0.02)
    growth_cond = board_growth & out["return_pct"].gt(-0.05) & out["return_pct"].lt(0.035)
    out["price_amplitude_factor"] = ((main_cond | growth_cond) & (out["volume"] < out["volume"].shift(1))).astype(float)

    burst_event = out["is_bullish"] & out["volume"].gt(out["volume"].shift(1) * 2.0)
    out["staged_volume_burst_count"] = burst_event.rolling(window=burst_window, min_periods=1).sum()
    out["staged_volume_burst_factor"] = (out["staged_volume_burst_count"].clip(upper=5.0) / 5.0).astype(float)

    prev_low = out["low"].shift(1)
    trend_touch = prev_low.between(out["trend_line"] * 0.99, out["trend_line"] * 1.01)
    dk_touch = prev_low.between(out["bull_bear_line"] * 0.99, out["bull_bear_line"] * 1.01)
    trend_confirm = trend_touch & out["close"].gt(out["trend_line"])
    dk_confirm = dk_touch & out["close"].gt(out["bull_bear_line"])
    out["pullback_confirmation_factor"] = ((trend_confirm.astype(float) + dk_confirm.astype(float)).clip(upper=2.0) / 2.0)

    out["drop_pct"] = out["return_pct"]
    min_drop_3 = out["drop_pct"].rolling(window=3, min_periods=3).min()
    min_vol_3 = out["volume"].rolling(window=3, min_periods=3).min()
    out["long_bear_short_volume_factor"] = (
        out["drop_pct"].eq(min_drop_3) & out["volume"].eq(min_vol_3) & out["drop_pct"].lt(0)
    ).astype(float)

    bullish_structure = out["MA10"].gt(out["MA20"]) & out["MA20"].gt(out["MA30"]) & out["MA30"].gt(out["MA60"])
    slopes_positive = pd.Series(True, index=out.index)
    for period in (10, 20, 30, 60):
        slopes_positive = slopes_positive & out[f"MA{period}"].gt(out[f"MA{period}"].shift(5))
    out["daily_ma_bull_factor"] = (bullish_structure & slopes_positive).astype(float)

    out["RSI14"] = _calc_rsi(out, 14)
    out["RSI28"] = _calc_rsi(out, 28)
    out["RSI57"] = _calc_rsi(out, 57)
    dif_strength = (out["DIF"] / out["close"].replace(0, np.nan) * 100.0).clip(lower=0.0)
    out["macd_dif_factor"] = (dif_strength / 2.0).clip(lower=0.0, upper=1.0).fillna(0.0)
    rsi_spread = (out["RSI14"] - out["RSI28"]).clip(lower=0.0) + (out["RSI28"] - out["RSI57"]).clip(lower=0.0)
    out["rsi_bull_factor"] = (rsi_spread / 20.0).clip(lower=0.0, upper=1.0).fillna(0.0)
    key_close_valid = out["关键K收盘价"].notna()
    out["key_k_support_factor"] = (
        key_close_valid & out["close"].ge(out["关键K收盘价"])
    ).astype(float)
    close_max_10 = out["close"].rolling(window=11, min_periods=11).max()
    close_min_10 = out["close"].rolling(window=11, min_periods=11).min()
    sideways_range_10 = ((close_max_10 - close_min_10) / close_min_10.replace(0, np.nan)).fillna(np.inf)
    out["price_sideways_10_factor"] = (
        ((0.03 - sideways_range_10) / 0.03)
        .clip(lower=0.0, upper=1.0)
        .fillna(0.0)
    )
    close_max_20 = out["close"].rolling(window=21, min_periods=21).max()
    close_min_20 = out["close"].rolling(window=21, min_periods=21).min()
    sideways_range = ((close_max_20 - close_min_20) / close_min_20.replace(0, np.nan)).fillna(np.inf)
    out["price_sideways_factor"] = (
        ((0.03 - sideways_range) / 0.03)
        .clip(lower=0.0, upper=1.0)
        .fillna(0.0)
    )
    j_avg_decline_4 = ((out["J"].shift(4) - out["J"]) / 4.0).fillna(0.0)
    out["j_decline_acceleration_factor"] = (
        (j_avg_decline_4 / 18.0)
        .clip(lower=0.0, upper=1.0)
        .fillna(0.0)
    )

    # Penalties are normalized to [0, 1] and combined later with fixed weights.
    bearish_volume_ratio = (out["volume"] / out["volume"].shift(1).replace(0, np.nan)).fillna(0.0)
    out["bearish_volume_penalty"] = (
        out["is_bearish"].astype(float)
        * ((bearish_volume_ratio - 1.5) / 1.5).clip(lower=0.0, upper=1.0)
    )

    trend_break_ratio = ((out["trend_line"] - out["close"]) / out["trend_line"].replace(0, np.nan)).clip(lower=0.0)
    intraday_trend_break = ((out["trend_line"] - out["low"]) / out["trend_line"].replace(0, np.nan) - 0.01).clip(lower=0.0)
    out["break_trend_penalty"] = pd.concat(
        [trend_break_ratio / 0.03, intraday_trend_break / 0.03],
        axis=1,
    ).max(axis=1).clip(lower=0.0, upper=1.0).fillna(0.0)

    bull_bear_break_ratio = ((out["bull_bear_line"] - out["close"]) / out["bull_bear_line"].replace(0, np.nan)).clip(lower=0.0)
    out["break_bull_bear_penalty"] = (bull_bear_break_ratio / 0.03).clip(lower=0.0, upper=1.0).fillna(0.0)
    out["bull_bear_above_trend_penalty"] = (
        ((out["bull_bear_line"] - out["trend_line"]) / out["trend_line"].replace(0, np.nan) / 0.03)
        .clip(lower=0.0, upper=1.0)
        .fillna(0.0)
    )
    dominance_ratio = bearish_count_30 / bullish_count_30.replace(0, np.nan)
    out["bearish_candle_dominance_penalty"] = (
        ((dominance_ratio - 1.5) / 1.0)
        .clip(lower=0.0, upper=1.0)
        .fillna(0.0)
    )
    extreme_bull_event = out["is_bullish"] & out["return_pct"].ge(0.08)
    extreme_bull_streak = _calc_event_streak(extreme_bull_event)
    rolling_extreme_bull_streak = extreme_bull_streak.rolling(window=30, min_periods=1).max()
    out["extreme_bull_run_penalty"] = (
        ((rolling_extreme_bull_streak - 2.0) / 3.0)
        .clip(lower=0.0, upper=1.0)
        .fillna(0.0)
    )

    overbought_j = ((out["J"] - 90.0) / 20.0).clip(lower=0.0, upper=1.0)
    overbought_rsi = ((out["RSI14"] - 75.0) / 15.0).clip(lower=0.0, upper=1.0).fillna(0.0)
    out["high_overbought_penalty"] = pd.concat([overbought_j, overbought_rsi], axis=1).max(axis=1).fillna(0.0)

    main_abnormal = out["board"].eq("MAIN") * ((out["amplitude_pct"] - 0.07) / 0.05).clip(lower=0.0)
    growth_abnormal = out["board"].isin(["GEM", "STAR"]) * ((out["amplitude_pct"] - 0.12) / 0.08).clip(lower=0.0)
    out["abnormal_amplitude_penalty"] = pd.concat([main_abnormal, growth_abnormal], axis=1).max(axis=1).clip(lower=0.0, upper=1.0).fillna(0.0)

    stagnation_volume_ratio = (out["volume"] / out["volume"].shift(1).replace(0, np.nan)).fillna(0.0)
    stagnation_price = (0.01 - out["return_pct"].clip(upper=0.01)) / 0.02
    out["volume_stagnation_penalty"] = (
        ((stagnation_volume_ratio - 1.2) / 0.8).clip(lower=0.0, upper=1.0)
        * stagnation_price.clip(lower=0.0, upper=1.0)
    ).fillna(0.0)
    out["key_k_close_break_penalty"] = (
        key_close_valid
        * ((out["关键K收盘价"] - out["close"]) / out["关键K收盘价"].replace(0, np.nan) / 0.05).clip(lower=0.0, upper=1.0)
    ).fillna(0.0)
    out["key_k_low_break_penalty"] = (
        out["关键K最低价"].notna()
        * ((out["关键K最低价"] - out["close"]) / out["关键K最低价"].replace(0, np.nan) / 0.03).clip(lower=0.0, upper=1.0)
    ).fillna(0.0)

    out["base_signal"] = out["J"].lt(0) & out["trend_line"].gt(out["bull_bear_line"])
    return out


def build_prepared_stock_data(stock_data: Dict[str, pd.DataFrame], burst_window: int) -> Dict[str, pd.DataFrame]:
    prepared: Dict[str, pd.DataFrame] = {}
    for code, raw_df in tqdm(stock_data.items(), desc="Preparing factor frames", unit="stock"):
        prepared[code] = prepare_factor_frame(raw_df, burst_window=burst_window).reset_index(drop=True)
    return prepared


def build_signal_candidates(prepared_stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    records: List[dict] = []
    for code, df in tqdm(prepared_stock_data.items(), desc="Collecting base signals", unit="stock"):
        signal_idx = np.flatnonzero(df["base_signal"].to_numpy(dtype=bool))
        signal_idx = signal_idx[signal_idx + 1 < len(df)]
        if signal_idx.size == 0:
            continue

        dates = df["date"].to_numpy()
        boards = df["board"].to_numpy()
        j_values = df["J"].to_numpy(dtype=float)
        trend_lines = df["trend_line"].to_numpy(dtype=float)
        bull_bear_lines = df["bull_bear_line"].to_numpy(dtype=float)
        burst_counts = df["staged_volume_burst_count"].to_numpy(dtype=float)
        factor_arrays = {col: df[col].to_numpy(dtype=float) for col in FACTOR_COLUMNS}

        for idx in signal_idx:
            row = {
                "code": code,
                "signal_idx": int(idx),
                "signal_date": dates[idx],
                "board": boards[idx],
                "j_value": float(j_values[idx]),
                "trend_line": float(trend_lines[idx]),
                "bull_bear_line": float(bull_bear_lines[idx]),
                "burst_count": float(burst_counts[idx]),
            }
            for col in FACTOR_COLUMNS:
                row[col] = float(factor_arrays[col][idx])
            for col in PENALTY_COLUMNS:
                row[col] = float(df.at[idx, col])
            records.append(row)

    if not records:
        return pd.DataFrame(
            columns=[
                "code",
                "signal_idx",
                "signal_date",
                "board",
                "j_value",
                "trend_line",
                "bull_bear_line",
                "burst_count",
                *FACTOR_COLUMNS,
                *PENALTY_COLUMNS,
            ]
        )

    return pd.DataFrame(records).sort_values(["signal_date", "code"]).reset_index(drop=True)
