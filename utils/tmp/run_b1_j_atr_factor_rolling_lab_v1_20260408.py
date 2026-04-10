from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import b1filter, stoploss, technical_indicators


RESULTS_DIR = ROOT / "results"
DATA_ROOT = ROOT / "data"
LOT_SIZE = 100
INITIAL_CAPITAL = 1_000_000.0
TRADING_DAYS_PER_YEAR = 252
WINDOW_START = pd.Timestamp("2024-04-02")
WINDOW_END = pd.Timestamp("2026-04-02")
SMOKE_HORIZONS = (20, 60, 120)
FULL_HORIZONS = (3, 5, 7, 10, 20, 30, 60, 120)
BEST_EXIT_DRAWDOWN = 0.09
BEST_EXIT_STOP_MULTIPLIER = 0.95
BEST_EXIT_MAX_HOLD_DAYS = 120
MIN_SAMPLE_EXP1 = 300
MIN_COVERAGE_EXP1 = 80
MIN_SAMPLE_EXP2 = 150
PIN_DEFAULT_EXIT_HOLD_DAYS = 3

ABS_J_THRESHOLDS = (-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0)
J_RANK_WINDOWS = (10, 20, 30, 40, 60)
J_RANK_THRESHOLDS = (0.05, 0.10, 0.15, 0.20)
HYBRID_WINDOWS = (20, 30)
HYBRID_THRESHOLDS = (0.05, 0.10)
HYBRID_ABS_J = 0.0

ATR_LOWER_BOUNDS = (0.8, 1.0, 1.2, 1.4)
ATR_UPPER_BOUNDS = (1.2, 1.4, 1.6, 1.8, 2.0)
VOL_THRESHOLDS = (0.6, 0.7, 0.8, 0.9)
STRICT_VOL_THRESHOLDS = (0.7, 0.8)

EXPERIMENT3_FACTOR_FAMILIES = {
    "low_volume_low_price": (
        ("volrank20_q05_and_closerank20_q10", lambda x: (x["volume_pct_rank_20"] <= 0.05) & (x["close_pct_rank_20"] <= 0.10)),
        ("volrank30_q10_and_closerank30_q15", lambda x: (x["volume_pct_rank_30"] <= 0.10) & (x["close_pct_rank_30"] <= 0.15)),
        ("volrank20_q10_and_closerank20_q20", lambda x: (x["volume_pct_rank_20"] <= 0.10) & (x["close_pct_rank_20"] <= 0.20)),
    ),
    "ma_pullback": (
        ("low_to_ma20_abs_le_2pct", lambda x: x["low_to_ma20_abs"] <= 0.02),
        ("low_to_ma60_abs_le_2pct", lambda x: x["low_to_ma60_abs"] <= 0.02),
        ("min_low_to_ma20_ma60_abs_le_3pct", lambda x: x[["low_to_ma20_abs", "low_to_ma60_abs"]].min(axis=1) <= 0.03),
    ),
    "trend_pullback": (
        ("low_to_trend_abs_le_2pct", lambda x: x["low_to_trend_abs"] <= 0.02),
        ("low_to_long_abs_le_2pct", lambda x: x["low_to_long_abs"] <= 0.02),
        ("min_low_to_trend_long_abs_le_3pct", lambda x: x[["low_to_trend_abs", "low_to_long_abs"]].min(axis=1) <= 0.03),
    ),
    "declining_volume": (
        ("vol_ma5_le_0p8", lambda x: x["vol_ratio_ma5"] <= 0.8),
        ("vol_ma10_le_0p8", lambda x: x["vol_ratio_ma10"] <= 0.8),
        ("vol_prev_lt_1", lambda x: x["vol_ratio_prev"] < 1.0),
        ("vol_prev_lt_1_and_ma5_le_0p8", lambda x: (x["vol_ratio_prev"] < 1.0) & (x["vol_ratio_ma5"] <= 0.8)),
    ),
    "consistent_close_range": (
        ("abs_ret1_le_2pct", lambda x: x["abs_ret1"] <= 0.02),
        ("abs_ret1_le_2pct_and_band3_le_4pct", lambda x: (x["abs_ret1"] <= 0.02) & (x["return_band_3d"] <= 0.04)),
        ("abs_ret1_le_3pct_and_band3_le_5pct", lambda x: (x["abs_ret1"] <= 0.03) & (x["return_band_3d"] <= 0.05)),
    ),
    "bullish_volume": (
        ("bull_vol_ratio20_ge_1p2", lambda x: x["bull_vol_ratio_20"] >= 1.2),
        ("bull_vol_ratio30_ge_1p382", lambda x: x["bull_vol_ratio_30"] >= 1.382),
        ("max_vol_bullish60_and_bars_le_20", lambda x: x["max_vol_is_bullish_60"] & (x["bars_since_max_vol_60"] <= 20)),
    ),
}


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


def calc_max_drawdown(nav: pd.Series) -> float:
    if nav.empty:
        return np.nan
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def calc_sharpe(daily_ret: pd.Series) -> float:
    if len(daily_ret) <= 1:
        return np.nan
    std = float(daily_ret.std(ddof=1))
    if not np.isfinite(std) or std <= 1e-12:
        return np.nan
    return float(daily_ret.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def calc_cagr(final_nav: float, n_days: int) -> float:
    if not np.isfinite(final_nav) or final_nav <= 0 or n_days <= 0:
        return np.nan
    years = n_days / TRADING_DAYS_PER_YEAR
    if years <= 0:
        return np.nan
    return float((final_nav / INITIAL_CAPITAL) ** (1.0 / years) - 1.0)


def choose_data_dir() -> tuple[Path, dict]:
    dirs = sorted(
        [p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 8],
        reverse=True,
    )
    rows = []
    for p in dirs:
        raw_cnt = len(list(p.glob("*.txt")))
        normal_cnt = len(list((p / "normal").glob("*.txt"))) if (p / "normal").exists() else 0
        rows.append({"date_dir": p.name, "path": str(p), "raw_count": raw_cnt, "normal_count": normal_cnt})
    if not rows:
        raise RuntimeError("data 目录中没有可用的日期快照")
    chosen = rows[0]
    if len(rows) >= 2 and rows[1]["raw_count"] > 0:
        if chosen["raw_count"] < rows[1]["raw_count"] * 0.95:
            chosen = rows[1]
    return Path(chosen["path"]), {"snapshot_candidates": rows, "selected": chosen}


def list_files(data_dir: Path, max_files: int) -> list[Path]:
    files = sorted(p for p in data_dir.glob("*.txt") if p.is_file())
    if max_files > 0:
        files = files[:max_files]
    return files


def _stock_code(file_path: Path) -> str:
    return file_path.stem.split("#")[-1]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    out = numerator.astype(float) / denominator.astype(float)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["最高"].astype(float)
    low = df["最低"].astype(float)
    close = df["收盘"].astype(float)
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work = technical_indicators.calculate_trend(work)
    work = technical_indicators.calculate_kdj(work)
    work = technical_indicators.calculate_daily_ma(work)
    weekly_map = b1filter.map_weekly_screen_to_daily_df(work)
    work = work.merge(weekly_map, on="日期", how="left")
    work["weekly_ok"] = work["weekly_ok"].fillna(False).astype(bool)
    work["ATR14"] = _compute_atr(work)
    work["ATR14_prev"] = work["ATR14"].shift(1)
    work["body_atr"] = _safe_ratio(work["开盘"] - work["收盘"], work["ATR14_prev"])
    work["vol_ma5"] = work["成交量"].rolling(5, min_periods=5).mean()
    work["vol_ma10"] = work["成交量"].rolling(10, min_periods=10).mean()
    work["vol_ratio_ma5"] = _safe_ratio(work["成交量"], work["vol_ma5"])
    work["vol_ratio_ma10"] = _safe_ratio(work["成交量"], work["vol_ma10"])
    work["vol_ratio_prev"] = _safe_ratio(work["成交量"], work["成交量"].shift(1))
    work["ret1"] = _safe_ratio(work["收盘"], work["收盘"].shift(1)) - 1.0
    work["abs_ret1"] = work["ret1"].abs()
    work["volume_pct_rank_20"] = rolling_last_percentile(work["成交量"], 20)
    work["volume_pct_rank_30"] = rolling_last_percentile(work["成交量"], 30)
    work["close_pct_rank_20"] = rolling_last_percentile(work["收盘"], 20)
    work["close_pct_rank_30"] = rolling_last_percentile(work["收盘"], 30)
    work["low_to_ma20_pct"] = _safe_ratio(work["最低"] - work["MA20"], work["MA20"])
    work["low_to_ma60_pct"] = _safe_ratio(work["最低"] - work["MA60"], work["MA60"])
    work["low_to_trend_pct"] = _safe_ratio(work["最低"] - work["知行短期趋势线"], work["知行短期趋势线"])
    work["low_to_long_pct"] = _safe_ratio(work["最低"] - work["知行多空线"], work["知行多空线"])
    work["low_to_ma20_abs"] = work["low_to_ma20_pct"].abs()
    work["low_to_ma60_abs"] = work["low_to_ma60_pct"].abs()
    work["low_to_trend_abs"] = work["low_to_trend_pct"].abs()
    work["low_to_long_abs"] = work["low_to_long_pct"].abs()
    recent_max_ret = work["ret1"].rolling(3, min_periods=3).max()
    recent_min_ret = work["ret1"].rolling(3, min_periods=3).min()
    work["return_band_3d"] = recent_max_ret - recent_min_ret

    bullish_vol = np.where(work["收盘"] > work["开盘"], work["成交量"], 0.0)
    bearish_vol = np.where(work["收盘"] < work["开盘"], work["成交量"], 0.0)
    work["bull_vol_ratio_20"] = _safe_ratio(
        pd.Series(bullish_vol, index=work.index).rolling(20, min_periods=20).sum(),
        pd.Series(bearish_vol, index=work.index).rolling(20, min_periods=20).sum(),
    )
    work["bull_vol_ratio_30"] = _safe_ratio(
        pd.Series(bullish_vol, index=work.index).rolling(30, min_periods=30).sum(),
        pd.Series(bearish_vol, index=work.index).rolling(30, min_periods=30).sum(),
    )

    max_vol_bullish = []
    bars_since_max_vol = []
    vol = work["成交量"].astype(float).to_numpy()
    open_arr = work["开盘"].astype(float).to_numpy()
    close_arr = work["收盘"].astype(float).to_numpy()
    for i in range(len(work)):
        start = max(0, i - 59)
        window_vol = vol[start : i + 1]
        if len(window_vol) == 0 or not np.isfinite(window_vol).any():
            max_vol_bullish.append(False)
            bars_since_max_vol.append(np.nan)
            continue
        local_idx = int(np.nanargmax(window_vol))
        idx = start + local_idx
        max_vol_bullish.append(bool(np.isfinite(close_arr[idx]) and np.isfinite(open_arr[idx]) and close_arr[idx] > open_arr[idx]))
        bars_since_max_vol.append(float(i - idx))
    work["max_vol_is_bullish_60"] = pd.Series(max_vol_bullish, index=work.index).astype(bool)
    work["bars_since_max_vol_60"] = pd.Series(bars_since_max_vol, index=work.index)

    for w in J_RANK_WINDOWS:
        work[f"j_rank_{w}"] = rolling_last_percentile(work["J"], w)
    return work


def _build_base_rows_for_file(file_path_str: str, horizons: tuple[int, ...]) -> list[dict]:
    file_path = Path(file_path_str)
    df, load_error = stoploss.load_data(str(file_path))
    if load_error or df is None or len(df) < 160:
        return []

    numeric_cols = ["开盘", "最高", "最低", "收盘", "成交量"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["日期", "开盘", "最高", "最低", "收盘"]).copy()
    if len(df) < 160:
        return []

    feature_df = _build_feature_frame(df.copy()).reset_index(drop=True)
    code = _stock_code(file_path)
    rows: list[dict] = []

    for idx in range(len(feature_df) - 1):
        signal_date = pd.Timestamp(feature_df.at[idx, "日期"])
        if signal_date < WINDOW_START or signal_date > WINDOW_END:
            continue
        trend_line = float(feature_df.at[idx, "知行短期趋势线"]) if pd.notna(feature_df.at[idx, "知行短期趋势线"]) else np.nan
        long_line = float(feature_df.at[idx, "知行多空线"]) if pd.notna(feature_df.at[idx, "知行多空线"]) else np.nan
        if not np.isfinite(trend_line) or not np.isfinite(long_line) or trend_line <= long_line:
            continue

        entry_idx = idx + 1
        entry_date = pd.Timestamp(feature_df.at[entry_idx, "日期"])
        if entry_date > WINDOW_END:
            continue
        entry_open = float(feature_df.at[entry_idx, "开盘"]) if pd.notna(feature_df.at[entry_idx, "开盘"]) else np.nan
        signal_low = float(feature_df.at[idx, "最低"]) if pd.notna(feature_df.at[idx, "最低"]) else np.nan
        if not np.isfinite(entry_open) or entry_open <= 0 or not np.isfinite(signal_low) or signal_low <= 0:
            continue

        row = {
            "code": code,
            "signal_idx": idx,
            "signal_date": signal_date,
            "entry_idx": entry_idx,
            "entry_date": entry_date,
            "entry_open": entry_open,
            "signal_low": signal_low,
            "j_value": float(feature_df.at[idx, "J"]) if pd.notna(feature_df.at[idx, "J"]) else np.nan,
            "weekly_ok": bool(feature_df.at[idx, "weekly_ok"]),
            "body_atr": float(feature_df.at[idx, "body_atr"]) if pd.notna(feature_df.at[idx, "body_atr"]) else np.nan,
            "vol_ratio_ma5": float(feature_df.at[idx, "vol_ratio_ma5"]) if pd.notna(feature_df.at[idx, "vol_ratio_ma5"]) else np.nan,
            "vol_ratio_ma10": float(feature_df.at[idx, "vol_ratio_ma10"]) if pd.notna(feature_df.at[idx, "vol_ratio_ma10"]) else np.nan,
            "vol_ratio_prev": float(feature_df.at[idx, "vol_ratio_prev"]) if pd.notna(feature_df.at[idx, "vol_ratio_prev"]) else np.nan,
            "volume_pct_rank_20": float(feature_df.at[idx, "volume_pct_rank_20"]) if pd.notna(feature_df.at[idx, "volume_pct_rank_20"]) else np.nan,
            "volume_pct_rank_30": float(feature_df.at[idx, "volume_pct_rank_30"]) if pd.notna(feature_df.at[idx, "volume_pct_rank_30"]) else np.nan,
            "close_pct_rank_20": float(feature_df.at[idx, "close_pct_rank_20"]) if pd.notna(feature_df.at[idx, "close_pct_rank_20"]) else np.nan,
            "close_pct_rank_30": float(feature_df.at[idx, "close_pct_rank_30"]) if pd.notna(feature_df.at[idx, "close_pct_rank_30"]) else np.nan,
            "low_to_ma20_pct": float(feature_df.at[idx, "low_to_ma20_pct"]) if pd.notna(feature_df.at[idx, "low_to_ma20_pct"]) else np.nan,
            "low_to_ma60_pct": float(feature_df.at[idx, "low_to_ma60_pct"]) if pd.notna(feature_df.at[idx, "low_to_ma60_pct"]) else np.nan,
            "low_to_trend_pct": float(feature_df.at[idx, "low_to_trend_pct"]) if pd.notna(feature_df.at[idx, "low_to_trend_pct"]) else np.nan,
            "low_to_long_pct": float(feature_df.at[idx, "low_to_long_pct"]) if pd.notna(feature_df.at[idx, "low_to_long_pct"]) else np.nan,
            "low_to_ma20_abs": float(feature_df.at[idx, "low_to_ma20_abs"]) if pd.notna(feature_df.at[idx, "low_to_ma20_abs"]) else np.nan,
            "low_to_ma60_abs": float(feature_df.at[idx, "low_to_ma60_abs"]) if pd.notna(feature_df.at[idx, "low_to_ma60_abs"]) else np.nan,
            "low_to_trend_abs": float(feature_df.at[idx, "low_to_trend_abs"]) if pd.notna(feature_df.at[idx, "low_to_trend_abs"]) else np.nan,
            "low_to_long_abs": float(feature_df.at[idx, "low_to_long_abs"]) if pd.notna(feature_df.at[idx, "low_to_long_abs"]) else np.nan,
            "abs_ret1": float(feature_df.at[idx, "abs_ret1"]) if pd.notna(feature_df.at[idx, "abs_ret1"]) else np.nan,
            "return_band_3d": float(feature_df.at[idx, "return_band_3d"]) if pd.notna(feature_df.at[idx, "return_band_3d"]) else np.nan,
            "bull_vol_ratio_20": float(feature_df.at[idx, "bull_vol_ratio_20"]) if pd.notna(feature_df.at[idx, "bull_vol_ratio_20"]) else np.nan,
            "bull_vol_ratio_30": float(feature_df.at[idx, "bull_vol_ratio_30"]) if pd.notna(feature_df.at[idx, "bull_vol_ratio_30"]) else np.nan,
            "max_vol_is_bullish_60": bool(feature_df.at[idx, "max_vol_is_bullish_60"]),
            "bars_since_max_vol_60": float(feature_df.at[idx, "bars_since_max_vol_60"]) if pd.notna(feature_df.at[idx, "bars_since_max_vol_60"]) else np.nan,
        }
        for w in J_RANK_WINDOWS:
            col = f"j_rank_{w}"
            row[col] = float(feature_df.at[idx, col]) if pd.notna(feature_df.at[idx, col]) else np.nan

        # Compute B1 boolean factors only for dates that already passed the hard trend condition.
        df_slice = feature_df.iloc[: idx + 1]
        ma_slice = feature_df.iloc[: idx + 1]
        kdj_slice = feature_df.iloc[: idx + 1]
        today_row = feature_df.iloc[idx]
        yesterday_row = feature_df.iloc[idx - 1] if idx >= 1 else feature_df.iloc[idx]
        ma_pullback, trend_pullback = b1filter._pullback_to_key_lines(df_slice, df_slice, ma_slice)
        row.update(
            {
                "factor_low_volume_low_price": bool(b1filter._is_low_volume_low_price(df_slice)),
                "factor_ma_pullback": bool(ma_pullback),
                "factor_trend_pullback": bool(trend_pullback),
                "factor_first_pullback": bool(b1filter._first_pullback_after_cross(df_slice, df_slice)),
                "factor_sb1": bool(b1filter._is_sb1(today_row, yesterday_row, kdj_slice)),
                "factor_declining_volume": bool(float(today_row["成交量"]) < float(yesterday_row["成交量"])) if idx >= 1 else False,
                "factor_consistent_close_range": bool(
                    b1filter._within_pct_range(
                        float(today_row["收盘"]),
                        float(yesterday_row["收盘"]),
                        b1filter.PRICE_RANGE_DOWN,
                        b1filter.PRICE_RANGE_UP,
                    )
                )
                if idx >= 1
                else False,
                "factor_recent_max_volume_is_bullish": bool(b1filter._recent_max_volume_is_bullish(df_slice)),
                "factor_bullish_volume_dominance": bool(b1filter._bullish_volume_dominance(df_slice)),
                "factor_gap_up_followed_by_big_bullish": bool(b1filter._has_gap_up_followed_by_big_bullish(df_slice, df_slice)),
                "factor_long_negative_short_volume": bool(b1filter._has_long_negative_short_volume(df_slice)),
            }
        )

        for h in horizons:
            close_idx = entry_idx + h - 1
            if close_idx < len(feature_df):
                exit_close = float(feature_df.at[close_idx, "收盘"]) if pd.notna(feature_df.at[close_idx, "收盘"]) else np.nan
                row[f"ret_{h}d"] = exit_close / entry_open - 1.0 if np.isfinite(exit_close) else np.nan
            else:
                row[f"ret_{h}d"] = np.nan
        rows.append(row)
    return rows


def build_base_candidate_df(data_dir: Path, max_files: int, horizons: tuple[int, ...]) -> pd.DataFrame:
    files = list_files(data_dir, max_files=max_files)
    if not files:
        raise RuntimeError("没有可用股票文件")
    worker_count = min(max(cpu_count() - 1, 1), 10)
    with Pool(processes=worker_count) as pool:
        rows_nested = pool.starmap(_build_base_rows_for_file, [(str(p), horizons) for p in files], chunksize=8)
    rows = [row for chunk in rows_nested for row in chunk]
    if not rows:
        raise RuntimeError("base candidate rows 为空")
    df = pd.DataFrame(rows).sort_values(["signal_date", "code"]).reset_index(drop=True)
    return df


def build_price_map(data_dir: Path, codes: list[str]) -> dict[str, pd.DataFrame]:
    price_map: dict[str, pd.DataFrame] = {}
    for code in sorted(set(codes)):
        paths = list(data_dir.glob(f"*#{code}.txt"))
        if not paths:
            continue
        df, load_error = stoploss.load_data(str(paths[0]))
        if load_error or df is None or df.empty:
            continue
        df = df[(df["日期"] >= WINDOW_START) & (df["日期"] <= pd.Timestamp("2026-12-31"))].copy()
        if df.empty:
            continue
        df = df.sort_values("日期").reset_index(drop=True)
        price_map[code] = df[["日期", "开盘", "最高", "最低", "收盘"]].copy()
    return price_map


def make_exp1_rule_masks(base_df: pd.DataFrame) -> list[dict]:
    rules: list[dict] = []
    for threshold in ABS_J_THRESHOLDS:
        mask = pd.to_numeric(base_df["j_value"], errors="coerce") <= threshold
        rules.append({"family": "absolute_j", "rule_name": f"abs_j_le_{threshold:g}", "complexity_rank": 0, "mask": mask})

    for w in J_RANK_WINDOWS:
        col = f"j_rank_{w}"
        series = pd.to_numeric(base_df[col], errors="coerce")
        for q in J_RANK_THRESHOLDS:
            mask = series <= q
            rules.append({"family": "j_percentile", "rule_name": f"j_rank_{w}_le_{q:.2f}", "complexity_rank": 1, "mask": mask})

    abs_mask = pd.to_numeric(base_df["j_value"], errors="coerce") <= HYBRID_ABS_J
    for op in ("and", "or"):
        for w in HYBRID_WINDOWS:
            rank_series = pd.to_numeric(base_df[f"j_rank_{w}"], errors="coerce")
            for q in HYBRID_THRESHOLDS:
                rank_mask = rank_series <= q
                mask = abs_mask & rank_mask if op == "and" else abs_mask | rank_mask
                rules.append(
                    {
                        "family": f"hybrid_{op}",
                        "rule_name": f"hybrid_{op}_abs0_rank{w}_q{int(q*100):02d}",
                        "complexity_rank": 2,
                        "mask": mask,
                    }
                )
    return rules


def summarize_signal_layer(df: pd.DataFrame, horizons: tuple[int, ...]) -> dict:
    out = {
        "signal_count": int(len(df)),
        "coverage_days": int(df["signal_date"].nunique()) if not df.empty else 0,
    }
    for h in horizons:
        col = f"ret_{h}d"
        valid = pd.to_numeric(df[col], errors="coerce").dropna()
        out[f"avg_ret_{h}d"] = float(valid.mean()) if not valid.empty else np.nan
        out[f"win_rate_{h}d"] = float((valid > 0).mean()) if not valid.empty else np.nan
    return out


def simulate_trade_layer(signal_df: pd.DataFrame, price_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for rec in signal_df.itertuples(index=False):
        price_df = price_map.get(rec.code)
        if price_df is None or price_df.empty:
            continue
        date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(price_df["日期"])}
        signal_idx = date_to_idx.get(pd.Timestamp(rec.signal_date))
        entry_idx = date_to_idx.get(pd.Timestamp(rec.entry_date))
        if signal_idx is None or entry_idx is None:
            continue
        if entry_idx <= signal_idx:
            continue
        entry_open = float(rec.entry_open)
        signal_low = float(rec.signal_low)
        if not np.isfinite(entry_open) or entry_open <= 0 or not np.isfinite(signal_low) or signal_low <= 0:
            continue

        stop_price = signal_low * BEST_EXIT_STOP_MULTIPLIER
        rolling_high = float(price_df["最高"].iloc[entry_idx]) if pd.notna(price_df["最高"].iloc[entry_idx]) else entry_open
        exit_date = None
        exit_price = np.nan
        exit_reason = ""
        max_trade_drawdown = np.nan

        # 买入当日允许按计划做当日止损
        day0_low = float(price_df["最低"].iloc[entry_idx]) if pd.notna(price_df["最低"].iloc[entry_idx]) else np.nan
        if np.isfinite(day0_low):
            max_trade_drawdown = day0_low / entry_open - 1.0
        if np.isfinite(day0_low) and day0_low <= stop_price:
            exit_date = pd.Timestamp(price_df["日期"].iloc[entry_idx])
            exit_price = stop_price
            exit_reason = "stop_same_day"
            holding_days = 1
            rows.append(
                {
                    "code": rec.code,
                    "signal_date": pd.Timestamp(rec.signal_date),
                    "entry_date": pd.Timestamp(rec.entry_date),
                    "exit_date": exit_date,
                    "entry_price": entry_open,
                    "exit_price": exit_price,
                    "holding_return": exit_price / entry_open - 1.0,
                    "holding_days": holding_days,
                    "exit_reason": exit_reason,
                    "max_drawdown_during_trade": max_trade_drawdown,
                }
            )
            continue

        scheduled_exit_idx: Optional[int] = None
        scheduled_exit_reason = ""
        worst_low_ratio = max_trade_drawdown
        for i in range(entry_idx + 1, len(price_df)):
            high_price = float(price_df["最高"].iloc[i]) if pd.notna(price_df["最高"].iloc[i]) else np.nan
            low_price = float(price_df["最低"].iloc[i]) if pd.notna(price_df["最低"].iloc[i]) else np.nan
            if np.isfinite(high_price):
                rolling_high = max(rolling_high, high_price)
            if np.isfinite(low_price):
                low_ratio = low_price / entry_open - 1.0
                if not np.isfinite(worst_low_ratio) or low_ratio < worst_low_ratio:
                    worst_low_ratio = low_ratio

            if np.isfinite(low_price) and low_price <= stop_price:
                exit_date = pd.Timestamp(price_df["日期"].iloc[i])
                exit_price = stop_price
                exit_reason = "stop_same_day"
                holding_days = i - entry_idx + 1
                break

            if scheduled_exit_idx is None and np.isfinite(low_price) and np.isfinite(rolling_high):
                if low_price <= rolling_high * (1.0 - BEST_EXIT_DRAWDOWN):
                    if i + 1 < len(price_df):
                        scheduled_exit_idx = i + 1
                        scheduled_exit_reason = "drawdown_9pct_next_open"

            if scheduled_exit_idx is not None and i >= scheduled_exit_idx:
                exit_date = pd.Timestamp(price_df["日期"].iloc[scheduled_exit_idx])
                exit_price = float(price_df["开盘"].iloc[scheduled_exit_idx])
                exit_reason = scheduled_exit_reason
                holding_days = scheduled_exit_idx - entry_idx + 1
                break

            if i - entry_idx + 1 >= BEST_EXIT_MAX_HOLD_DAYS:
                if i + 1 < len(price_df):
                    exit_date = pd.Timestamp(price_df["日期"].iloc[i + 1])
                    exit_price = float(price_df["开盘"].iloc[i + 1])
                    exit_reason = "max_hold_next_open"
                    holding_days = i + 1 - entry_idx + 1
                break

        if exit_date is None or not np.isfinite(exit_price) or exit_price <= 0:
            continue

        rows.append(
            {
                "code": rec.code,
                "signal_date": pd.Timestamp(rec.signal_date),
                "entry_date": pd.Timestamp(rec.entry_date),
                "exit_date": exit_date,
                "entry_price": entry_open,
                "exit_price": exit_price,
                "holding_return": exit_price / entry_open - 1.0,
                "holding_days": holding_days,
                "exit_reason": exit_reason,
                "max_drawdown_during_trade": worst_low_ratio,
            }
        )
    return pd.DataFrame(rows)


def summarize_trade_layer(trade_df: pd.DataFrame) -> dict:
    if trade_df.empty:
        return {
            "trade_count": 0,
            "avg_trade_return": np.nan,
            "win_rate": np.nan,
            "avg_holding_days": np.nan,
            "avg_max_drawdown_during_trade": np.nan,
        }
    return {
        "trade_count": int(len(trade_df)),
        "avg_trade_return": float(trade_df["holding_return"].mean()),
        "win_rate": float((trade_df["holding_return"] > 0).mean()),
        "avg_holding_days": float(trade_df["holding_days"].mean()),
        "avg_max_drawdown_during_trade": float(trade_df["max_drawdown_during_trade"].mean()),
    }


def rank_exp1_results(summary_df: pd.DataFrame) -> pd.DataFrame:
    df = summary_df.copy()
    df["valid_main_filter"] = (df["signal_count"] >= MIN_SAMPLE_EXP1) & (df["coverage_days"] >= MIN_COVERAGE_EXP1)
    df["sort_trade"] = pd.to_numeric(df["avg_trade_return"], errors="coerce").fillna(-np.inf)
    df["sort_20d"] = pd.to_numeric(df["avg_ret_20d"], errors="coerce").fillna(-np.inf)
    df["sort_win20"] = pd.to_numeric(df["win_rate_20d"], errors="coerce").fillna(-np.inf)
    df = df.sort_values(
        ["valid_main_filter", "sort_trade", "sort_20d", "sort_win20", "complexity_rank"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    return df


def run_experiment1(base_df: pd.DataFrame, price_map: dict[str, pd.DataFrame], horizons: tuple[int, ...], result_dir: Path) -> tuple[pd.DataFrame, dict]:
    rules = make_exp1_rule_masks(base_df)
    signal_rows = []
    trade_rows = []
    trades_dir = result_dir / "exp1_trade_samples"
    trades_dir.mkdir(parents=True, exist_ok=True)
    for rule in rules:
        picked = base_df[rule["mask"].fillna(False)].copy()
        signal_summary = summarize_signal_layer(picked, horizons)
        trade_df = simulate_trade_layer(picked[["code", "signal_date", "entry_date", "entry_open", "signal_low"]], price_map)
        trade_summary = summarize_trade_layer(trade_df)
        row = {
            "family": rule["family"],
            "rule_name": rule["rule_name"],
            "complexity_rank": rule["complexity_rank"],
            **signal_summary,
            **trade_summary,
        }
        signal_rows.append(row)
        if not trade_df.empty:
            trade_df.to_csv(trades_dir / f"{rule['rule_name']}.csv", index=False, encoding="utf-8-sig")
        trade_rows.append({"rule_name": rule["rule_name"], "trade_count": int(len(trade_df))})

    signal_summary_df = pd.DataFrame(signal_rows)
    ranked_df = rank_exp1_results(signal_summary_df)
    ranked_df.to_csv(result_dir / "signal_summary.csv", index=False, encoding="utf-8-sig")
    ranked_df.to_csv(result_dir / "trade_summary.csv", index=False, encoding="utf-8-sig")

    best_row = ranked_df.iloc[0].to_dict()
    secondary_row = ranked_df.iloc[1].to_dict() if len(ranked_df) > 1 else None
    best_payload = {"best": best_row, "secondary": secondary_row}
    (result_dir / "best_j_config.json").write_text(json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return ranked_df, best_payload


def _build_exp2_masks(df: pd.DataFrame) -> list[dict]:
    masks = []
    body_atr = pd.to_numeric(df["body_atr"], errors="coerce")
    vol_ma5 = pd.to_numeric(df["vol_ratio_ma5"], errors="coerce")
    vol_ma10 = pd.to_numeric(df["vol_ratio_ma10"], errors="coerce")
    vol_prev = pd.to_numeric(df["vol_ratio_prev"], errors="coerce")
    for lb in ATR_LOWER_BOUNDS:
        for ub in ATR_UPPER_BOUNDS:
            if lb >= ub:
                continue
            atr_mask = (body_atr >= lb) & (body_atr < ub)
            for t in VOL_THRESHOLDS:
                masks.append({"definition": f"atr_{lb:.1f}_{ub:.1f}__ma5_le_{t:.1f}", "mask": atr_mask & (vol_ma5 <= t)})
                masks.append({"definition": f"atr_{lb:.1f}_{ub:.1f}__ma10_le_{t:.1f}", "mask": atr_mask & (vol_ma10 <= t)})
            masks.append({"definition": f"atr_{lb:.1f}_{ub:.1f}__prev_lt_1", "mask": atr_mask & (vol_prev < 1.0)})
            for t in STRICT_VOL_THRESHOLDS:
                masks.append(
                    {
                        "definition": f"atr_{lb:.1f}_{ub:.1f}__prev_lt_1_and_ma5_le_{t:.1f}",
                        "mask": atr_mask & (vol_prev < 1.0) & (vol_ma5 <= t),
                    }
                )
    return masks


def run_experiment2(best_j_df: pd.DataFrame, price_map: dict[str, pd.DataFrame], horizons: tuple[int, ...], result_dir: Path) -> tuple[pd.DataFrame, dict]:
    rows = []
    for item in _build_exp2_masks(best_j_df):
        picked = best_j_df[item["mask"].fillna(False)].copy()
        signal_summary = summarize_signal_layer(picked, horizons)
        trade_df = simulate_trade_layer(picked[["code", "signal_date", "entry_date", "entry_open", "signal_low"]], price_map)
        trade_summary = summarize_trade_layer(trade_df)
        rows.append({"definition": item["definition"], **signal_summary, **trade_summary})

    result_df = pd.DataFrame(rows)
    result_df["valid_main_filter"] = result_df["signal_count"] >= MIN_SAMPLE_EXP2
    result_df = result_df.sort_values(
        ["valid_main_filter", "avg_trade_return", "avg_ret_20d", "win_rate_20d", "signal_count"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    result_df.to_csv(result_dir / "atr_long_bear_short_vol_grid.csv", index=False, encoding="utf-8-sig")
    best_payload = {
        "best": result_df.iloc[0].to_dict() if not result_df.empty else None,
        "secondary": result_df.iloc[1].to_dict() if len(result_df) > 1 else None,
    }
    (result_dir / "best_long_bear_short_vol.json").write_text(json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return result_df, best_payload


def run_experiment3(best_j_df: pd.DataFrame, price_map: dict[str, pd.DataFrame], horizons: tuple[int, ...], result_dir: Path) -> tuple[pd.DataFrame, dict]:
    family_best = {}
    all_rows = []
    for family, variants in EXPERIMENT3_FACTOR_FAMILIES.items():
        rows = []
        for variant_name, fn in variants:
            mask = fn(best_j_df).fillna(False)
            picked = best_j_df[mask].copy()
            signal_summary = summarize_signal_layer(picked, horizons)
            trade_df = simulate_trade_layer(picked[["code", "signal_date", "entry_date", "entry_open", "signal_low"]], price_map)
            trade_summary = summarize_trade_layer(trade_df)
            rows.append({"family": family, "variant_name": variant_name, **signal_summary, **trade_summary})
        family_df = pd.DataFrame(rows).sort_values(
            ["avg_trade_return", "avg_ret_20d", "win_rate_20d", "signal_count"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
        family_df.to_csv(result_dir / f"{family}_factor_quant_summary.csv", index=False, encoding="utf-8-sig")
        all_rows.extend(family_df.to_dict(orient="records"))
        family_best[family] = family_df.iloc[0].to_dict() if not family_df.empty else None

    all_df = pd.DataFrame(all_rows).sort_values(
        ["avg_trade_return", "avg_ret_20d", "win_rate_20d", "signal_count"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    all_df.to_csv(result_dir / "factor_quant_ranking.csv", index=False, encoding="utf-8-sig")
    payload = {"family_best": family_best, "overall_top10": all_df.head(10).to_dict(orient="records")}
    (result_dir / "best_quant_factor_candidates.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return all_df, payload


def compute_pin_candidate_df(data_dir: Path, max_files: int) -> pd.DataFrame:
    rows = []
    for file_path in list_files(data_dir, max_files=max_files):
        df, load_error = stoploss.load_data(str(file_path))
        if load_error or df is None or len(df) < 40:
            continue
        df = technical_indicators.calculate_trend(df.copy())
        df = technical_indicators.calculate_kdj(df)
        weekly_map = b1filter.map_weekly_screen_to_daily_df(df)
        df = df.merge(weekly_map, on="日期", how="left")
        df["weekly_ok"] = df["weekly_ok"].fillna(False).astype(bool)
        vol_ma5 = pd.to_numeric(df["成交量"], errors="coerce").rolling(5, min_periods=5).mean()
        n1, n2 = 3, 21
        llv_l_n1 = df["最低"].rolling(window=n1).min()
        hhv_c_n1 = df["收盘"].rolling(window=n1).max()
        short_value = (df["收盘"] - llv_l_n1) / (hhv_c_n1 - llv_l_n1) * 100
        llv_l_n2 = df["最低"].rolling(window=n2).min()
        hhv_l_n2 = df["收盘"].rolling(window=n2).max()
        long_value = (df["收盘"] - llv_l_n2) / (hhv_l_n2 - llv_l_n2) * 100
        code = _stock_code(file_path)

        for idx in range(len(df) - 1):
            signal_date = pd.Timestamp(df.at[idx, "日期"])
            if signal_date < WINDOW_START or signal_date > WINDOW_END:
                continue
            entry_idx = idx + 1
            entry_date = pd.Timestamp(df.at[entry_idx, "日期"])
            if entry_date > WINDOW_END:
                continue
            if not bool(df.at[idx, "weekly_ok"]):
                continue
            trend_line = float(df.at[idx, "知行短期趋势线"]) if pd.notna(df.at[idx, "知行短期趋势线"]) else np.nan
            long_line = float(df.at[idx, "知行多空线"]) if pd.notna(df.at[idx, "知行多空线"]) else np.nan
            if not np.isfinite(trend_line) or not np.isfinite(long_line) or trend_line <= long_line:
                continue
            today_vol = float(df.at[idx, "成交量"]) if pd.notna(df.at[idx, "成交量"]) else np.nan
            ma5_v = float(vol_ma5.iloc[idx]) if pd.notna(vol_ma5.iloc[idx]) else np.nan
            if not np.isfinite(today_vol) or not np.isfinite(ma5_v) or not (today_vol < ma5_v):
                continue
            entry_open = float(df.at[entry_idx, "开盘"]) if pd.notna(df.at[entry_idx, "开盘"]) else np.nan
            if not np.isfinite(entry_open) or entry_open <= 0:
                continue
            row = {
                "code": code,
                "signal_date": signal_date,
                "entry_date": entry_date,
                "entry_open": entry_open,
                "short_value": float(short_value.iloc[idx]) if pd.notna(short_value.iloc[idx]) else np.nan,
                "long_value": float(long_value.iloc[idx]) if pd.notna(long_value.iloc[idx]) else np.nan,
            }
            for h in FULL_HORIZONS:
                close_idx = entry_idx + h - 1
                if close_idx < len(df):
                    exit_close = float(df.at[close_idx, "收盘"]) if pd.notna(df.at[close_idx, "收盘"]) else np.nan
                    row[f"ret_{h}d"] = exit_close / entry_open - 1.0 if np.isfinite(exit_close) else np.nan
                else:
                    row[f"ret_{h}d"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def simulate_pin_hold3(signal_df: pd.DataFrame, price_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for rec in signal_df.itertuples(index=False):
        price_df = price_map.get(rec.code)
        if price_df is None or price_df.empty:
            continue
        date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(price_df["日期"])}
        entry_idx = date_to_idx.get(pd.Timestamp(rec.entry_date))
        if entry_idx is None:
            continue
        exit_idx = entry_idx + PIN_DEFAULT_EXIT_HOLD_DAYS - 1
        if exit_idx >= len(price_df):
            continue
        exit_price = float(price_df["收盘"].iloc[exit_idx])
        if not np.isfinite(exit_price) or exit_price <= 0:
            continue
        rows.append(
            {
                "code": rec.code,
                "signal_date": pd.Timestamp(rec.signal_date),
                "entry_date": pd.Timestamp(rec.entry_date),
                "exit_date": pd.Timestamp(price_df["日期"].iloc[exit_idx]),
                "holding_return": exit_price / float(rec.entry_open) - 1.0,
                "holding_days": PIN_DEFAULT_EXIT_HOLD_DAYS,
            }
        )
    return pd.DataFrame(rows)


def summarize_window(signal_df: pd.DataFrame, trade_df: pd.DataFrame, signal_horizon: int) -> dict:
    signal_count = int(len(signal_df))
    signal_coverage_days = int(signal_df["signal_date"].nunique()) if not signal_df.empty else 0
    col = f"ret_{signal_horizon}d"
    signal_ret = pd.to_numeric(signal_df[col], errors="coerce").dropna() if col in signal_df.columns else pd.Series(dtype=float)
    return {
        "signal_count": signal_count,
        "signal_coverage_days": signal_coverage_days,
        "signal_avg_return": float(signal_ret.mean()) if not signal_ret.empty else np.nan,
        "signal_win_rate": float((signal_ret > 0).mean()) if not signal_ret.empty else np.nan,
        "trade_count": int(len(trade_df)),
        "trade_avg_return": float(trade_df["holding_return"].mean()) if not trade_df.empty else np.nan,
        "trade_win_rate": float((trade_df["holding_return"] > 0).mean()) if not trade_df.empty else np.nan,
    }


def run_experiment4(
    base_df: pd.DataFrame,
    price_map: dict[str, pd.DataFrame],
    exp1_ranked: pd.DataFrame,
    exp2_best: dict,
    exp3_payload: dict,
    pin_df: pd.DataFrame,
    result_dir: Path,
) -> dict:
    latest_date = WINDOW_END
    windows = (60, 120, 240)
    versions = []

    def exp1_mask(rule_name: str) -> pd.Series:
        for item in make_exp1_rule_masks(base_df):
            if item["rule_name"] == rule_name:
                return item["mask"].fillna(False)
        return pd.Series(False, index=base_df.index)

    baseline_rule = "hybrid_or_abs0_rank20_q10"
    versions.append({"family": "b1", "version_name": "b1_baseline_current", "signal_df": base_df[exp1_mask(baseline_rule)].copy(), "signal_horizon": 20, "trade_builder": "b1"})

    top_exp1_rows = exp1_ranked.head(2)
    for _, row in top_exp1_rows.iterrows():
        versions.append(
            {
                "family": "b1",
                "version_name": row["rule_name"],
                "signal_df": base_df[exp1_mask(row["rule_name"])].copy(),
                "signal_horizon": 20,
                "trade_builder": "b1",
            }
        )

    if exp2_best.get("best"):
        definition = exp2_best["best"]["definition"]
        mask = None
        best_j_rule = exp1_ranked.iloc[0]["rule_name"]
        best_j_df = base_df[exp1_mask(best_j_rule)].copy()
        for item in _build_exp2_masks(best_j_df):
            if item["definition"] == definition:
                mask = item["mask"].fillna(False)
                break
        if mask is not None:
            versions.append(
                {
                    "family": "b1",
                    "version_name": f"exp2_{definition}",
                    "signal_df": best_j_df[mask].copy(),
                    "signal_horizon": 20,
                    "trade_builder": "b1",
                }
            )

    best_j_rule = exp1_ranked.iloc[0]["rule_name"]
    best_j_df = base_df[exp1_mask(best_j_rule)].copy()
    for family, row in exp3_payload["family_best"].items():
        if not row:
            continue
        variant_name = row["variant_name"]
        for candidate_name, fn in EXPERIMENT3_FACTOR_FAMILIES[family]:
            if candidate_name == variant_name:
                mask = fn(best_j_df).fillna(False)
                versions.append(
                    {
                        "family": "b1",
                        "version_name": f"exp3_{family}_{variant_name}",
                        "signal_df": best_j_df[mask].copy(),
                        "signal_horizon": 20,
                        "trade_builder": "b1",
                    }
                )
                break

    versions.append(
        {
            "family": "pin",
            "version_name": "pin_85_30",
            "signal_df": pin_df[(pin_df["short_value"] <= 30) & (pin_df["long_value"] >= 85)].copy(),
            "signal_horizon": 3,
            "trade_builder": "pin",
        }
    )
    versions.append(
        {
            "family": "pin",
            "version_name": "pin_80_20",
            "signal_df": pin_df[(pin_df["short_value"] <= 20) & (pin_df["long_value"] >= 80)].copy(),
            "signal_horizon": 3,
            "trade_builder": "pin",
        }
    )

    scoreboard_rows = []
    for version in versions:
        signal_df = version["signal_df"].copy()
        if signal_df.empty:
            continue
        for window_len in windows:
            window_start = latest_date - pd.offsets.BDay(window_len - 1)
            signal_window = signal_df[(signal_df["signal_date"] >= window_start) & (signal_df["signal_date"] <= latest_date)].copy()
            if version["trade_builder"] == "b1":
                trade_window = simulate_trade_layer(signal_window[["code", "signal_date", "entry_date", "entry_open", "signal_low"]], price_map)
            else:
                trade_window = simulate_pin_hold3(signal_window[["code", "signal_date", "entry_date", "entry_open"]], price_map)
            metrics = summarize_window(signal_window, trade_window, version["signal_horizon"])
            scoreboard_rows.append(
                {
                    "family": version["family"],
                    "version_name": version["version_name"],
                    "window_days": window_len,
                    **metrics,
                }
            )

    scoreboard = pd.DataFrame(scoreboard_rows)
    scoreboard.to_csv(result_dir / "rolling_window_scoreboard.csv", index=False, encoding="utf-8-sig")

    active_payload = {"latest_date": str(latest_date.date()), "active_versions": []}
    baseline_lookup = {"b1": "b1_baseline_current", "pin": "pin_85_30"}
    for family, baseline_name in baseline_lookup.items():
        fam_df = scoreboard[scoreboard["family"] == family].copy()
        if fam_df.empty:
            continue
        baseline_120 = fam_df[(fam_df["version_name"] == baseline_name) & (fam_df["window_days"] == 120)]
        baseline_60 = fam_df[(fam_df["version_name"] == baseline_name) & (fam_df["window_days"] == 60)]
        if baseline_120.empty or baseline_60.empty:
            continue
        b120 = baseline_120.iloc[0]
        b60 = baseline_60.iloc[0]
        for version_name in sorted(fam_df["version_name"].unique()):
            if version_name == baseline_name:
                continue
            cur120_df = fam_df[(fam_df["version_name"] == version_name) & (fam_df["window_days"] == 120)]
            cur60_df = fam_df[(fam_df["version_name"] == version_name) & (fam_df["window_days"] == 60)]
            if cur120_df.empty or cur60_df.empty:
                continue
            cur120 = cur120_df.iloc[0]
            cur60 = cur60_df.iloc[0]
            still_effective = (
                np.isfinite(cur120["signal_avg_return"])
                and cur120["signal_avg_return"] > 0
                and np.isfinite(cur120["signal_win_rate"])
                and cur120["signal_win_rate"] > b120["signal_win_rate"]
                and np.isfinite(cur60["signal_avg_return"])
                and cur60["signal_avg_return"] > 0
                and cur120["signal_count"] >= 50
            )
            better_now = False
            if still_effective:
                avg_ret_better = (
                    np.isfinite(cur120["trade_avg_return"])
                    and np.isfinite(b120["trade_avg_return"])
                    and cur120["trade_avg_return"] > b120["trade_avg_return"] * 1.2
                )
                win_better = (
                    np.isfinite(cur120["trade_win_rate"])
                    and np.isfinite(b120["trade_win_rate"])
                    and cur120["trade_win_rate"] > b120["trade_win_rate"] + 0.05
                )
                count_better = (
                    cur120["signal_count"] > b120["signal_count"]
                    and (
                        not np.isfinite(b120["trade_avg_return"])
                        or not np.isfinite(cur120["trade_avg_return"])
                        or cur120["trade_avg_return"] >= b120["trade_avg_return"]
                    )
                )
                direction_agree = (
                    np.isfinite(cur60["trade_avg_return"])
                    and np.isfinite(b60["trade_avg_return"])
                    and cur60["trade_avg_return"] >= b60["trade_avg_return"]
                )
                better_now = direction_agree and (avg_ret_better or win_better or count_better)
            active_payload["active_versions"].append(
                {
                    "family": family,
                    "baseline": baseline_name,
                    "version_name": version_name,
                    "still_effective": bool(still_effective),
                    "started_working_now": bool(better_now),
                    "window120_trade_avg_return": cur120["trade_avg_return"],
                    "window120_trade_win_rate": cur120["trade_win_rate"],
                    "window120_signal_count": int(cur120["signal_count"]),
                }
            )

    (result_dir / "active_strategy_now.json").write_text(json.dumps(active_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_lines = [
        "# Pin Regime Switch Report",
        "",
        f"- Latest date: {latest_date.date()}",
        "- Default pin version: 85-30",
        "- Switch rule: 80-20 only switches in when both 120-day and 60-day windows beat 85-30.",
        "",
    ]
    pin_rows = [row for row in active_payload["active_versions"] if row["family"] == "pin"]
    if pin_rows:
        for row in pin_rows:
            report_lines.append(
                f"- {row['version_name']}: still_effective={row['still_effective']}, started_working_now={row['started_working_now']}, "
                f"window120_trade_avg_return={row['window120_trade_avg_return']}, signal_count={row['window120_signal_count']}"
            )
    (result_dir / "pin_regime_switch_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return active_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B1买点优化、ATR长阴短柱优化、关键因子量化与滚动窗口判定")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizons = SMOKE_HORIZONS if args.mode == "smoke" else FULL_HORIZONS
    max_files = args.max_files if args.max_files is not None else (400 if args.mode == "smoke" else 0)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR / f"b1_j_atr_factor_rolling_lab_v1_{args.mode}_{ts}"
    result_dir.mkdir(parents=True, exist_ok=True)

    data_dir, selection_meta = choose_data_dir()
    (result_dir / "data_selection.json").write_text(json.dumps(selection_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    base_df = build_base_candidate_df(data_dir, max_files=max_files, horizons=horizons)
    base_df.to_csv(result_dir / "base_candidates.csv", index=False, encoding="utf-8-sig")
    price_map = build_price_map(data_dir, base_df["code"].tolist())

    exp1_dir = result_dir / "exp1"
    exp1_dir.mkdir(parents=True, exist_ok=True)
    exp1_ranked, best_j_payload = run_experiment1(base_df, price_map, horizons, exp1_dir)

    best_rule_name = best_j_payload["best"]["rule_name"]
    best_mask = None
    for item in make_exp1_rule_masks(base_df):
        if item["rule_name"] == best_rule_name:
            best_mask = item["mask"].fillna(False)
            break
    if best_mask is None:
        raise RuntimeError("无法回溯实验1最佳J规则")
    best_j_df = base_df[best_mask].copy()
    best_j_df.to_csv(result_dir / "best_j_signals.csv", index=False, encoding="utf-8-sig")

    exp2_dir = result_dir / "exp2"
    exp2_dir.mkdir(parents=True, exist_ok=True)
    exp2_df, exp2_payload = run_experiment2(best_j_df, price_map, horizons, exp2_dir)

    exp3_dir = result_dir / "exp3"
    exp3_dir.mkdir(parents=True, exist_ok=True)
    exp3_df, exp3_payload = run_experiment3(best_j_df, price_map, horizons, exp3_dir)

    pin_df = compute_pin_candidate_df(data_dir, max_files=max_files)
    pin_df.to_csv(result_dir / "pin_candidates.csv", index=False, encoding="utf-8-sig")

    exp4_dir = result_dir / "exp4"
    exp4_dir.mkdir(parents=True, exist_ok=True)
    active_payload = run_experiment4(base_df, price_map, exp1_ranked, exp2_payload, exp3_payload, pin_df, exp4_dir)

    summary = {
        "mode": args.mode,
        "data_dir": str(data_dir),
        "max_files": max_files,
        "window_start": str(WINDOW_START.date()),
        "window_end": str(WINDOW_END.date()),
        "base_candidate_count": int(len(base_df)),
        "price_map_codes": int(len(price_map)),
        "exp1_best": best_j_payload["best"],
        "exp2_best": exp2_payload["best"],
        "exp3_family_best": exp3_payload["family_best"],
        "active_versions": active_payload["active_versions"],
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
