from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils import technical_indicators


DATA_DIR = os.environ.get("PIN_DATA_DIR", "/Users/lidongyang/Desktop/Qstrategy/data/forward_data")
OUTPUT_DIR = os.environ.get("PIN_OUTPUT_DIR", "/Users/lidongyang/Desktop/Qstrategy/results/pin_secondary_factor_experiment")
MAX_FILES = int(os.environ.get("PIN_MAX_FILES", "0"))
MIN_BARS = 160
EPS = 1e-12
EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")

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
    df["code"] = raw[code_col].astype(str).iloc[0] if code_col else os.path.splitext(os.path.basename(path))[0]
    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    df = df[(df["date"] < EXCLUDE_START) | (df["date"] > EXCLUDE_END)].copy()
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0) & (df["volume"] >= 0)].copy()
    if len(df) < MIN_BARS:
        return None
    return df


def safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full(np.shape(a), np.nan, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > EPS)
    out[mask] = a[mask] / b[mask]
    return out


def build_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().reset_index(drop=True)
    cn = pd.DataFrame(
        {
            "日期": x["date"],
            "开盘": x["open"],
            "最高": x["high"],
            "最低": x["low"],
            "收盘": x["close"],
            "成交量": x["volume"],
        }
    )
    cn = technical_indicators.calculate_trend(cn)

    x["trend_line"] = cn["知行短期趋势线"].astype(float)
    x["long_line"] = cn["知行多空线"].astype(float)
    x["trend_ok"] = x["trend_line"] > x["long_line"]

    short_n = 3
    long_n = 21
    short_llv = x["low"].rolling(short_n).min()
    short_hhv = x["close"].rolling(short_n).max()
    short_den = (short_hhv - short_llv).replace(0, np.nan)
    short_value = (x["close"] - short_llv) / short_den * 100
    long_llv = x["low"].rolling(long_n).min()
    long_hhv = x["close"].rolling(long_n).max()
    long_den = (long_hhv - long_llv).replace(0, np.nan)
    long_value = (x["close"] - long_llv) / long_den * 100
    x["pin_signal"] = (short_value <= 30) & (long_value >= 85)

    full_range = (x["high"] - x["low"]).replace(0, np.nan)
    real_body_low = np.minimum(x["open"], x["close"])
    x["close_position"] = safe_div(x["close"] - x["low"], full_range)
    x["lower_shadow_ratio"] = safe_div(real_body_low - x["low"], full_range)
    x["break_prev_low_recover"] = (x["low"] < x["low"].shift(1)) & (x["close"] > x["low"].shift(1))
    x["trend_slope_3"] = safe_div(x["trend_line"], x["trend_line"].shift(3)) - 1.0
    x["vol_ma5_prev"] = x["volume"].shift(1).rolling(5).mean()
    x["signal_vs_ma5"] = safe_div(x["volume"], x["vol_ma5_prev"])
    return x


def load_feature_map(data_dir: str) -> Dict[str, pd.DataFrame]:
    feature_map: Dict[str, pd.DataFrame] = {}
    files = [f for f in os.listdir(data_dir) if f.lower().endswith((".csv", ".txt"))]
    if MAX_FILES > 0:
        files = files[:MAX_FILES]
    for idx, file_name in enumerate(files, 1):
        df = load_one_csv(os.path.join(data_dir, file_name))
        if df is None:
            continue
        code = str(df["code"].iloc[0])
        feature_map[code] = build_feature_df(df)
        if idx % 500 == 0:
            print(f"特征进度: {idx}/{len(files)}")
    return feature_map


def simulate_trade(df: pd.DataFrame, signal_idx: int) -> Optional[dict]:
    entry_idx = signal_idx + 1
    exit_idx = signal_idx + 2
    if exit_idx >= len(df):
        return None
    entry_price = float(df.at[entry_idx, "open"])
    exit_price = float(df.at[exit_idx, "open"])
    if not np.isfinite(entry_price) or not np.isfinite(exit_price) or entry_price <= 0 or exit_price <= 0:
        return None
    ret = exit_price / entry_price - 1.0
    return {
        "signal_date": df.at[signal_idx, "date"],
        "entry_date": df.at[entry_idx, "date"],
        "exit_date": df.at[exit_idx, "date"],
        "code": df.at[signal_idx, "code"],
        "ret": ret,
        "success": ret > 0,
        "close_position": float(df.at[signal_idx, "close_position"]),
        "lower_shadow_ratio": float(df.at[signal_idx, "lower_shadow_ratio"]),
        "break_prev_low_recover": bool(df.at[signal_idx, "break_prev_low_recover"]),
        "trend_slope_3": float(df.at[signal_idx, "trend_slope_3"]),
        "signal_vs_ma5": float(df.at[signal_idx, "signal_vs_ma5"]),
    }


def build_trade_df(feature_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    trades = []
    for code, df in feature_map.items():
        idxs = np.flatnonzero((df["trend_ok"] & df["pin_signal"]).to_numpy())
        for signal_idx in idxs:
            trade = simulate_trade(df, int(signal_idx))
            if trade is not None:
                trades.append(trade)
    return pd.DataFrame(trades)


def bucket_numeric(series: pd.Series, bins: List[float], labels: List[str]) -> pd.Series:
    return pd.cut(series.astype(float), bins=bins, labels=labels, include_lowest=True, right=True)


def summarize_bucket(df: pd.DataFrame, factor_col: str, bucket_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby(bucket_col, dropna=False)
        .agg(
            sample_count=("ret", "size"),
            avg_trade_return=("ret", "mean"),
            success_rate=("success", "mean"),
        )
        .reset_index()
        .rename(columns={bucket_col: "bucket"})
    )
    grouped.insert(0, "factor", factor_col)
    return grouped


def main() -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    feature_map = load_feature_map(DATA_DIR)
    trade_df = build_trade_df(feature_map)
    if trade_df.empty:
        raise ValueError("基础单针策略无样本")

    x = trade_df.copy()
    x["close_position_bucket"] = bucket_numeric(
        x["close_position"],
        [-0.001, 0.2, 0.4, 0.6, 0.8, 1.0],
        ["<=0.2", "0.2~0.4", "0.4~0.6", "0.6~0.8", "0.8~1.0"],
    )
    x["lower_shadow_bucket"] = bucket_numeric(
        x["lower_shadow_ratio"],
        [-0.001, 0.1, 0.2, 0.35, 0.5, 1.0],
        ["<=0.1", "0.1~0.2", "0.2~0.35", "0.35~0.5", "0.5~1.0"],
    )
    x["trend_slope_3_bucket"] = bucket_numeric(
        x["trend_slope_3"],
        [-1.0, 0.0, 0.003, 0.008, 1.0],
        ["<=0", "0~0.3%", "0.3%~0.8%", ">0.8%"],
    )
    x["signal_vs_ma5_bucket"] = bucket_numeric(
        x["signal_vs_ma5"],
        [-0.001, 0.8, 1.2, 1.8, 2.5, 100.0],
        ["<=0.8", "0.8~1.2", "1.2~1.8", "1.8~2.5", ">2.5"],
    )
    x["break_prev_low_recover_bucket"] = x["break_prev_low_recover"].map({True: "true", False: "false"})

    summary_frames = [
        summarize_bucket(x, "close_position", "close_position_bucket"),
        summarize_bucket(x, "lower_shadow_ratio", "lower_shadow_bucket"),
        summarize_bucket(x, "break_prev_low_recover", "break_prev_low_recover_bucket"),
        summarize_bucket(x, "trend_slope_3", "trend_slope_3_bucket"),
        summarize_bucket(x, "signal_vs_ma5", "signal_vs_ma5_bucket"),
    ]
    summary_df = pd.concat(summary_frames, ignore_index=True)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "factor_bucket_summary.csv"), index=False, encoding="utf-8-sig")
    x.to_csv(os.path.join(OUTPUT_DIR, "pin_trades_with_secondary_factors.csv"), index=False, encoding="utf-8-sig")

    best_rows = []
    for factor, group in summary_df.groupby("factor"):
        g = group[group["sample_count"] >= 100].copy()
        if g.empty:
            g = group.copy()
        best = g.sort_values(["avg_trade_return", "success_rate"], ascending=[False, False]).iloc[0]
        best_rows.append(
            {
                "factor": factor,
                "best_bucket": best["bucket"],
                "sample_count": int(best["sample_count"]),
                "avg_trade_return": float(best["avg_trade_return"]),
                "success_rate": float(best["success_rate"]),
            }
        )
    summary = {
        "data_dir": DATA_DIR,
        "exclude_range": [str(EXCLUDE_START.date()), str(EXCLUDE_END.date())],
        "base_signal": ["趋势线 > 多空线", "符合单针条件"],
        "entry_exit": {
            "entry": "signal_date_next_open",
            "holding_days": 1,
            "exit": "entry_date_plus_1_open",
        },
        "trade_count": int(len(x)),
        "best_buckets": best_rows,
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
