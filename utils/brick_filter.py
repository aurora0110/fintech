from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from utils.market_risk_tags import add_risk_features, format_risk_note, latest_risk_snapshot


INPUT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/data/20260226/normal")
MIN_BARS = 160
EPS = 1e-12
TOP_N = 10
PCT_RANK_THRESHOLD = 0.50
MODE = "perfect"

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


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().reset_index(drop=True)
    x["ret1"] = x["close"].pct_change()
    x["trend_line"] = x["close"].ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()
    x["close_pullback_white"] = x["close"] < x["trend_line"] * 1.01
    x["close_above_white"] = x["close"] > x["trend_line"]

    x["ma14"] = x["close"].rolling(14).mean()
    x["ma28"] = x["close"].rolling(28).mean()
    x["ma57"] = x["close"].rolling(57).mean()
    x["ma114"] = x["close"].rolling(114).mean()
    x["long_line"] = (x["ma14"] + x["ma28"] + x["ma57"] + x["ma114"]) / 4.0

    x["vol_ma5_prev"] = x["volume"].shift(1).rolling(5).mean()
    x["signal_vs_ma5"] = safe_div(x["volume"], x["vol_ma5_prev"])
    x["signal_vs_ma5_valid"] = x["signal_vs_ma5"].between(1, 2.2, inclusive="both")

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

    x["close_slope_10"] = (
        x["close"]
        .rolling(10)
        .apply(lambda s: np.polyfit(np.arange(len(s)), s, 1)[0] if np.isfinite(s).all() else np.nan, raw=False)
    )
    x["not_sideways"] = np.abs(safe_div(x["close_slope_10"], x["close"].rolling(10).mean())) > 0.002

    x["up_leg_avg_vol"] = x["volume"].shift(4).rolling(3).mean()
    x["pullback_avg_vol"] = x["volume"].shift(1).rolling(3).mean()
    x["pullback_shrinking"] = x["pullback_avg_vol"] < x["up_leg_avg_vol"]

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
    x["signal_base"] = (
        (x["pattern_a"] | x["pattern_b"])
        & x["pullback_shrinking"].fillna(False)
        & x["signal_vs_ma5_valid"].fillna(False)
        & x["not_sideways"].fillna(False)
        & x["ret1"].notna()
    )

    x["ret10"] = x["close"].pct_change(10)
    x["ret20"] = x["close"].pct_change(20)
    x["trend_spread"] = safe_div(x["trend_line"] - x["long_line"], x["close"], default=np.nan)
    x["close_to_trend"] = safe_div(x["close"] - x["trend_line"], x["trend_line"], default=np.nan)
    x["close_to_long"] = safe_div(x["close"] - x["long_line"], x["long_line"], default=np.nan)

    delta = x["close"].diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    avg_up_14 = up.ewm(alpha=1 / 14, adjust=False).mean()
    avg_down_14 = down.ewm(alpha=1 / 14, adjust=False).mean()
    avg_up_28 = up.ewm(alpha=1 / 28, adjust=False).mean()
    avg_down_28 = down.ewm(alpha=1 / 28, adjust=False).mean()
    avg_up_57 = up.ewm(alpha=1 / 57, adjust=False).mean()
    avg_down_57 = down.ewm(alpha=1 / 57, adjust=False).mean()
    x["RSI14"] = 100 - 100 / (1 + safe_div(avg_up_14, avg_down_14.replace(0, np.nan)))
    x["RSI28"] = 100 - 100 / (1 + safe_div(avg_up_28, avg_down_28.replace(0, np.nan)))
    x["RSI57"] = 100 - 100 / (1 + safe_div(avg_up_57, avg_down_57.replace(0, np.nan)))

    low_9 = x["low"].rolling(9).min()
    high_9 = x["high"].rolling(9).max()
    rsv = safe_div(x["close"] - low_9, (high_9 - low_9).replace(0, np.nan)) * 100
    x["K"] = pd.Series(rsv, index=x.index).ewm(alpha=1 / 3, adjust=False).mean()
    x["D"] = x["K"].ewm(alpha=1 / 3, adjust=False).mean()
    x["J"] = 3 * x["K"] - 2 * x["D"]
    x["J_turn_up"] = x["J"] > x["J"].shift(1)

    rng = (x["high"] - x["low"]).replace(0, np.nan)
    x["body_abs"] = (x["close"] - x["open"]).abs()
    x["body_pct"] = safe_div(x["body_abs"], x["close"], default=np.nan)
    x["upper_shadow"] = (x["high"] - x[["open", "close"]].max(axis=1)).clip(lower=0)
    x["lower_shadow"] = (x[["open", "close"]].min(axis=1) - x["low"]).clip(lower=0)
    x["upper_shadow_pct"] = safe_div(x["upper_shadow"], rng, default=np.nan)
    x["lower_shadow_pct"] = safe_div(x["lower_shadow"], rng, default=np.nan)
    x["close_location"] = safe_div(x["close"] - x["low"], rng, default=np.nan)

    x["touch_trend"] = x["low"] <= x["trend_line"] * 1.015
    x["touch_long"] = x["low"] <= x["long_line"] * 1.015
    x["trend_riding"] = x["close_to_trend"].between(0.0, 0.08, inclusive="both")

    x["green_bar"] = x["close"] > x["open"]
    x["double_bull_bar"] = x["green_bar"] & (x["volume"] >= x["volume"].shift(1) * 2.0)
    x["prior_double_bull_20"] = x["double_bull_bar"].shift(1).rolling(20).max().fillna(0).astype(bool)
    x["double_bar_high"] = np.nan
    x["double_bar_low"] = np.nan
    x["double_bar_close"] = np.nan
    last_high = np.nan
    last_low = np.nan
    last_close = np.nan
    for idx, row in x.iterrows():
        if bool(row["double_bull_bar"]):
            last_high = float(row["high"])
            last_low = float(row["low"])
            last_close = float(row["close"])
        x.at[idx, "double_bar_high"] = last_high
        x.at[idx, "double_bar_low"] = last_low
        x.at[idx, "double_bar_close"] = last_close
    x["support_above_double_low"] = x["close"] >= x["double_bar_low"]
    x["support_above_double_close"] = x["close"] >= x["double_bar_close"]
    x["support_above_double_high"] = x["close"] >= x["double_bar_high"]
    x["dist_to_double_high"] = safe_div(x["close"] - x["double_bar_high"], x["double_bar_high"], default=np.nan)
    x["dist_to_double_close"] = safe_div(x["close"] - x["double_bar_close"], x["double_bar_close"], default=np.nan)
    x["dist_to_double_low"] = safe_div(x["close"] - x["double_bar_low"], x["double_bar_low"], default=np.nan)
    x["prior5_avg_close_to_trend"] = x["close_to_trend"].shift(1).rolling(5).mean()
    x["price_vol_trend_sync"] = (
        x["ret10"].gt(0.01)
        & x["close_to_long"].gt(0.05)
        & x["signal_vs_ma5"].between(0.65, 3.8, inclusive="both")
    )
    x["strong_trend_setup"] = (
        x["trend_line"].gt(x["long_line"])
        & x["trend_spread"].gt(0.025)
        & x["close_to_long"].gt(0.04)
        & x["ret10"].gt(0.01)
        & x["RSI14"].ge(55)
        & x["prior_double_bull_20"]
        & x["support_above_double_close"].fillna(False)
        & x["close_location"].ge(0.72)
        & x["upper_shadow_pct"].le(0.28)
        & x["signal_vs_ma5"].between(0.65, 3.8, inclusive="both")
    )
    return x


def build_signal_df(input_dir: Path, mode: str = MODE) -> pd.DataFrame:
    rows: List[dict] = []
    files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {".csv", ".txt"}])
    total = len(files)
    for idx, path in enumerate(files, 1):
        df = load_one_csv(str(path))
        if df is None or df.empty:
            continue
        code = str(df["code"].iloc[0])
        x = add_features(df)
        mask_a = x["pattern_a"] & (x["rebound_ratio"] >= (1.2 if mode == "legacy" else 0.8))
        mask_b = x["pattern_b"] & (x["rebound_ratio"] >= 1.0)
        legacy_mask = (
            x["signal_base"]
            & (x["ret1"] <= 0.08)
            & (mask_a | mask_b)
            & (x["trend_line"] > x["long_line"])
        )
        perfect_mask = (
            legacy_mask
            & x["trend_line"].gt(x["long_line"])
            & x["ret1"].between(-0.03, 0.11, inclusive="both")
        )
        mask = legacy_mask if mode == "legacy" else perfect_mask
        signal_idxs = np.flatnonzero(mask.to_numpy())
        for signal_idx in signal_idxs:
            close = float(x.at[int(signal_idx), "close"])
            trend_line = float(x.at[int(signal_idx), "trend_line"])
            long_line = float(x.at[int(signal_idx), "long_line"])
            ret1 = float(x.at[int(signal_idx), "ret1"])
            signal_vs_ma5 = float(x.at[int(signal_idx), "signal_vs_ma5"])
            rebound_ratio = float(x.at[int(signal_idx), "rebound_ratio"])
            pullback_avg_vol = float(x.at[int(signal_idx), "pullback_avg_vol"])
            up_leg_avg_vol = float(x.at[int(signal_idx), "up_leg_avg_vol"])
            pullback_shrink_ratio = pullback_avg_vol / up_leg_avg_vol if np.isfinite(up_leg_avg_vol) and up_leg_avg_vol > 0 else np.nan
            trend_spread_clip = max((trend_line - long_line) / close, 0.0) if np.isfinite(close) and close > 0 else 0.0
            trend_quality = triangle_quality(float(x.at[int(signal_idx), "trend_spread"]), 0.08, 0.06)
            support_quality = (
                1.0 if bool(x.at[int(signal_idx), "support_above_double_high"]) else
                0.75 if bool(x.at[int(signal_idx), "support_above_double_close"]) else
                0.45 if bool(x.at[int(signal_idx), "support_above_double_low"]) else 0.0
            )
            momentum_quality = 0.5 * triangle_quality(float(x.at[int(signal_idx), "RSI14"]), 62.0, 12.0) + 0.5 * triangle_quality(float(x.at[int(signal_idx), "ret20"]), 0.18, 0.15)
            candle_quality = 0.6 * triangle_quality(float(x.at[int(signal_idx), "close_location"]), 0.86, 0.18) + 0.4 * triangle_quality(float(x.at[int(signal_idx), "upper_shadow_pct"]), 0.08, 0.20)
            volume_quality = triangle_quality(float(x.at[int(signal_idx), "signal_vs_ma5"]), 1.25, 1.8)
            brick_quality = triangle_quality(min(float(x.at[int(signal_idx), "rebound_ratio"]) if pd.notna(x.at[int(signal_idx), "rebound_ratio"]) else 0.0, 8.0), 4.5, 4.0)
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
                    "ret1_quality": triangle_quality(ret1, 0.03, 0.03),
                    "signal_vs_ma5_quality": triangle_quality(signal_vs_ma5, 1.7, 0.5),
                    "shrink_quality": triangle_quality(pullback_shrink_ratio, 0.8, 0.3),
                    "trend_quality": trend_quality,
                    "support_quality": support_quality,
                    "momentum_quality": momentum_quality,
                    "candle_quality": candle_quality,
                    "volume_quality": volume_quality,
                    "brick_quality": brick_quality,
                    "trend_spread": float(x.at[int(signal_idx), "trend_spread"]) if pd.notna(x.at[int(signal_idx), "trend_spread"]) else np.nan,
                    "close_to_trend": float(x.at[int(signal_idx), "close_to_trend"]) if pd.notna(x.at[int(signal_idx), "close_to_trend"]) else np.nan,
                    "close_to_long": float(x.at[int(signal_idx), "close_to_long"]) if pd.notna(x.at[int(signal_idx), "close_to_long"]) else np.nan,
                    "RSI14": float(x.at[int(signal_idx), "RSI14"]) if pd.notna(x.at[int(signal_idx), "RSI14"]) else np.nan,
                    "ret10": float(x.at[int(signal_idx), "ret10"]) if pd.notna(x.at[int(signal_idx), "ret10"]) else np.nan,
                    "ret20": float(x.at[int(signal_idx), "ret20"]) if pd.notna(x.at[int(signal_idx), "ret20"]) else np.nan,
                    "upper_shadow_pct": float(x.at[int(signal_idx), "upper_shadow_pct"]) if pd.notna(x.at[int(signal_idx), "upper_shadow_pct"]) else np.nan,
                    "close_location": float(x.at[int(signal_idx), "close_location"]) if pd.notna(x.at[int(signal_idx), "close_location"]) else np.nan,
                    "support_above_double_close": bool(x.at[int(signal_idx), "support_above_double_close"]),
                    "support_above_double_high": bool(x.at[int(signal_idx), "support_above_double_high"]),
                    "prior_double_bull_20": bool(x.at[int(signal_idx), "prior_double_bull_20"]),
                    "strong_trend_setup": bool(x.at[int(signal_idx), "strong_trend_setup"]),
                    "pattern_a": bool(x.at[int(signal_idx), "pattern_a"]),
                    "pattern_b": bool(x.at[int(signal_idx), "pattern_b"]),
                }
            )
        if idx % 500 == 0 or idx == total:
            print(f"特征进度: {idx}/{total}")

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["date", "code"]).reset_index(drop=True)
    if mode == "legacy":
        out["rebound_rank"] = out.groupby("date")["rebound_ratio"].rank(pct=True)
        out["trend_rank"] = out.groupby("date")["trend_spread_clip"].rank(pct=True)
        out["shrink_rank"] = out.groupby("date")["shrink_quality"].rank(pct=True)
        out["sort_score"] = 0.50 * out["shrink_rank"] + 0.30 * out["rebound_rank"] + 0.20 * out["trend_rank"]
    else:
        out["trend_rank"] = out.groupby("date")["trend_quality"].rank(pct=True)
        out["support_rank"] = out.groupby("date")["support_quality"].rank(pct=True)
        out["momentum_rank"] = out.groupby("date")["momentum_quality"].rank(pct=True)
        out["candle_rank"] = out.groupby("date")["candle_quality"].rank(pct=True)
        out["volume_rank"] = out.groupby("date")["volume_quality"].rank(pct=True)
        out["brick_rank"] = out.groupby("date")["brick_quality"].rank(pct=True)
        out["sort_score"] = (
            0.28 * out["trend_rank"]
            + 0.22 * out["support_rank"]
            + 0.20 * out["momentum_rank"]
            + 0.15 * out["candle_rank"]
            + 0.10 * out["volume_rank"]
            + 0.05 * out["brick_rank"]
        )
    out["score_pct_rank"] = out.groupby("date")["sort_score"].rank(pct=True)
    out["daily_rank"] = out.sort_values(["date", "sort_score", "code"], ascending=[True, False, True]).groupby("date").cumcount() + 1
    return out


def apply_selection(signal_df: pd.DataFrame, mode: str = MODE) -> pd.DataFrame:
    if signal_df.empty:
        return signal_df
    if mode == "legacy":
        x = signal_df[signal_df["score_pct_rank"] >= PCT_RANK_THRESHOLD].copy()
    else:
        x = signal_df[signal_df["score_pct_rank"] >= 0.20].copy()
    x = x.sort_values(["date", "sort_score", "code"], ascending=[True, False, True])
    x["daily_rank"] = x.groupby("date").cumcount() + 1
    return x.groupby("date", group_keys=False).head(TOP_N).reset_index(drop=True)


def check(file_path, hold_list=None, mode: str = MODE, feature_cache=None):
    if feature_cache is not None:
        df = feature_cache.raw_df()
    else:
        df = load_one_csv(str(file_path))
    if df is None or df.empty:
        return [-1]
    x = feature_cache.brick_features() if feature_cache is not None else add_features(df)
    latest_idx = len(x) - 1
    if latest_idx < 0:
        return [-1]
    latest = x.iloc[latest_idx]
    mask_a = bool(latest["pattern_a"]) and float(latest["rebound_ratio"]) >= (1.2 if mode == "legacy" else 0.8)
    mask_b = bool(latest["pattern_b"]) and float(latest["rebound_ratio"]) >= 1.0
    legacy_ok = (
        bool(latest["signal_base"])
        and float(latest["ret1"]) <= 0.08
        and (mask_a or mask_b)
        and float(latest["trend_line"]) > float(latest["long_line"])
    )
    perfect_ok = (
        legacy_ok
        and float(latest["trend_line"]) > float(latest["long_line"])
        and (-0.03 <= float(latest["ret1"]) <= 0.11)
    )
    signal_ok = legacy_ok if mode == "legacy" else perfect_ok
    if not signal_ok:
        return [-1]
    pullback_avg_vol = float(latest["pullback_avg_vol"])
    up_leg_avg_vol = float(latest["up_leg_avg_vol"])
    pullback_shrink_ratio = pullback_avg_vol / up_leg_avg_vol if np.isfinite(up_leg_avg_vol) and up_leg_avg_vol > 0 else np.nan
    if mode == "legacy":
        sort_score = (
            0.50 * triangle_quality(pullback_shrink_ratio, 0.8, 0.3)
            + 0.30 * 1.0
            + 0.20 * 1.0
        )
    else:
        trend_quality = triangle_quality(float(latest["trend_spread"]), 0.08, 0.06)
        support_quality = 1.0 if bool(latest["support_above_double_high"]) else 0.75 if bool(latest["support_above_double_close"]) else 0.45
        momentum_quality = 0.5 * triangle_quality(float(latest["RSI14"]), 62.0, 12.0) + 0.5 * triangle_quality(float(latest["ret20"]), 0.18, 0.15)
        candle_quality = 0.6 * triangle_quality(float(latest["close_location"]), 0.86, 0.18) + 0.4 * triangle_quality(float(latest["upper_shadow_pct"]), 0.08, 0.20)
        volume_quality = triangle_quality(float(latest["signal_vs_ma5"]), 1.25, 1.8)
        brick_quality = triangle_quality(min(float(latest["rebound_ratio"]) if pd.notna(latest["rebound_ratio"]) else 0.0, 8.0), 4.5, 4.0)
        sort_score = (
            0.28 * trend_quality
            + 0.22 * support_quality
            + 0.20 * momentum_quality
            + 0.15 * candle_quality
            + 0.10 * volume_quality
            + 0.05 * brick_quality
        )
    stop_loss_price = round(float(latest["low"]) * 0.99, 3)
    if feature_cache is not None:
        risk_note = format_risk_note(feature_cache.risk_snapshot())
    else:
        risk_df = add_risk_features(df)
        risk_note = format_risk_note(latest_risk_snapshot(risk_df))
    note = "brick动量续冲"
    if risk_note:
        note = f"{note} | {risk_note}"
    return [1, stop_loss_price, float(latest["close"]), round(sort_score, 4), note]


def main() -> None:
    signal_df = build_signal_df(INPUT_DIR, mode=MODE)
    selected_df = apply_selection(signal_df, mode=MODE)
    if selected_df.empty:
        return
    latest_trade_date = pd.to_datetime(selected_df["date"]).max()
    latest_df = selected_df[pd.to_datetime(selected_df["date"]) == latest_trade_date].copy()
    print(latest_df[["date", "code", "daily_rank", "sort_score", "rebound_ratio", "signal_vs_ma5", "pullback_shrink_ratio"]].to_string(index=False))


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


if __name__ == "__main__":
    main()
