from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import stoploss, technical_indicators


SIGNAL_FILE = ROOT / "results" / "brick_filter" / "all_signals_ranked.csv"
DATA_DIR = ROOT / "data" / "forward_data"
OUT_DIR = ROOT / "results" / "brick_negative_factor_ab_20260314"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")
MIN_BARS = 160
EPS = 1e-12


def safe_div(a, b):
    if a is None or b is None:
        return np.nan
    try:
        a_f = float(a)
        b_f = float(b)
    except Exception:
        return np.nan
    if not math.isfinite(a_f) or not math.isfinite(b_f) or abs(b_f) <= EPS:
        return np.nan
    return a_f / b_f


def build_stair_volume_flag(x: pd.DataFrame, lookback: int = 30) -> pd.Series:
    out = np.zeros(len(x), dtype=bool)
    closes = x["close"].to_numpy(dtype=float)
    opens = x["open"].to_numpy(dtype=float)
    vols = x["volume"].to_numpy(dtype=float)
    for idx in range(len(x)):
        left = max(0, idx - lookback + 1)
        hit = False
        for anchor in range(left, idx - 2):
            if not (closes[anchor] > opens[anchor]):
                continue
            hist_left = max(0, anchor - lookback + 1)
            anchor_hist = vols[hist_left : anchor + 1]
            if len(anchor_hist) < 5:
                continue
            if np.sum(anchor_hist >= vols[anchor]) > 3:
                continue
            for stair_len in (2, 3, 4):
                end = anchor + stair_len
                if end >= idx:
                    continue
                ok = True
                prev_vol = vols[anchor]
                for j in range(anchor + 1, end + 1):
                    if not (closes[j] < opens[j]):
                        ok = False
                        break
                    if not (vols[j] < prev_vol and vols[j] <= prev_vol * 0.95 and vols[j] >= prev_vol * 0.45):
                        ok = False
                        break
                    prev_vol = vols[j]
                if ok:
                    hit = True
                    break
            if hit:
                break
        out[idx] = hit
    return pd.Series(out, index=x.index)


def build_failed_breakout_flag(x: pd.DataFrame, lookback: int = 60) -> pd.Series:
    full_range = (x["high"] - x["low"]).replace(0, np.nan)
    body_high = np.maximum(x["open"], x["close"])
    upper_shadow_ratio = (x["high"] - body_high) / full_range
    prior_high = x["high"].shift(1).rolling(lookback, min_periods=20).max()
    return (
        (x["high"] >= prior_high * 0.995)
        & (x["close"] < prior_high * 0.995)
        & (upper_shadow_ratio >= (1.0 / 3.0))
    ).fillna(False)


def build_code_negative_features(file_path: Path) -> Optional[pd.DataFrame]:
    df, err = stoploss.load_data(str(file_path))
    if err or df is None or len(df) < MIN_BARS:
        return None
    df = df[(df["日期"] < EXCLUDE_START) | (df["日期"] > EXCLUDE_END)].copy()
    if len(df) < MIN_BARS:
        return None
    df = technical_indicators.calculate_trend(df)
    x = pd.DataFrame(
        {
            "date": pd.to_datetime(df["日期"]),
            "open": df["开盘"].astype(float),
            "high": df["最高"].astype(float),
            "low": df["最低"].astype(float),
            "close": df["收盘"].astype(float),
            "volume": df["成交量"].astype(float),
        }
    ).reset_index(drop=True)
    x["ret5"] = x["close"] / x["close"].shift(5) - 1.0
    x["too_fast_up_5d40"] = (x["ret5"] > 0.40).fillna(False)
    x["stair_volume_30d"] = build_stair_volume_flag(x, 30)
    x["failed_breakout_long_upper"] = build_failed_breakout_flag(x, 60)
    x["negative_any"] = x["stair_volume_30d"] | x["too_fast_up_5d40"] | x["failed_breakout_long_upper"]
    return x[["date", "stair_volume_30d", "too_fast_up_5d40", "failed_breakout_long_upper", "negative_any"]]


def attach_negative_features(signals: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for idx, code in enumerate(sorted(signals["code"].unique()), 1):
        file_path = DATA_DIR / f"{code}.txt"
        if not file_path.exists():
            continue
        feat = build_code_negative_features(file_path)
        if feat is None:
            continue
        sub = signals.loc[signals["code"] == code, ["date", "code"]].drop_duplicates()
        merged = sub.merge(feat, on="date", how="left")
        rows.append(merged)
        if idx % 200 == 0:
            print(f"负向特征进度: {idx}/{signals['code'].nunique()}")
    enrich = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return signals.merge(enrich, on=["date", "code"], how="left")


def select_daily_variant(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    stair_mask = df["stair_volume_30d"].fillna(False).astype(bool)
    fast_mask = df["too_fast_up_5d40"].fillna(False).astype(bool)
    breakout_mask = df["failed_breakout_long_upper"].fillna(False).astype(bool)
    negative_mask = df["negative_any"].fillna(False).astype(bool)

    if variant == "baseline":
        pool = df.copy()
    elif variant == "exclude_stair_volume":
        pool = df.loc[~stair_mask].copy()
    elif variant == "exclude_fast_rise":
        pool = df.loc[~fast_mask].copy()
    elif variant == "exclude_failed_breakout":
        pool = df.loc[~breakout_mask].copy()
    elif variant == "exclude_any_negative":
        pool = df.loc[~negative_mask].copy()
    else:
        raise ValueError(variant)

    out_rows: List[pd.DataFrame] = []
    for date, sub in pool.groupby("date", sort=True):
        sub = sub.sort_values(["sort_score", "code"], ascending=[False, True]).reset_index(drop=True)
        if sub.empty:
            continue
        cutoff = max(1, int(math.ceil(len(sub) * 0.5)))
        selected = sub.iloc[:cutoff].head(10).copy()
        selected["variant"] = variant
        selected["variant_daily_rank"] = np.arange(1, len(selected) + 1)
        out_rows.append(selected)
    return pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()


def simulate_trade_for_signal(row: pd.Series, code_df: pd.DataFrame) -> Optional[dict]:
    date_to_idx = {d: i for i, d in enumerate(code_df["date"])}
    signal_date = pd.Timestamp(row["date"])
    if signal_date not in date_to_idx:
        return None
    signal_idx = date_to_idx[signal_date]
    entry_idx = signal_idx + 1
    exit_idx_default = signal_idx + 4
    if entry_idx >= len(code_df) or exit_idx_default >= len(code_df):
        return None

    entry_open = float(code_df.at[entry_idx, "open"])
    if not np.isfinite(entry_open) or entry_open <= 0:
        return None

    tp_level = entry_open * 1.03
    entry_day_low = float(code_df.at[entry_idx, "low"])
    sl_level = entry_day_low * 0.99

    trigger = None
    planned_exit_idx = exit_idx_default

    for bar_idx in range(entry_idx, min(signal_idx + 4, len(code_df) - 1)):
        bar_high = float(code_df.at[bar_idx, "high"])
        bar_low = float(code_df.at[bar_idx, "low"])
        hit_tp = np.isfinite(bar_high) and bar_high >= tp_level
        hit_sl = np.isfinite(bar_low) and bar_low <= sl_level
        if hit_tp and hit_sl:
            trigger = "stop_loss"
            planned_exit_idx = bar_idx + 1
            break
        if hit_sl:
            trigger = "stop_loss"
            planned_exit_idx = bar_idx + 1
            break
        if hit_tp:
            trigger = "take_profit"
            planned_exit_idx = bar_idx + 1
            break

    if planned_exit_idx >= len(code_df):
        return None

    exit_open = float(code_df.at[planned_exit_idx, "open"])
    trade_ret = exit_open / entry_open - 1.0
    return {
        "date": signal_date,
        "code": row["code"],
        "variant": row["variant"],
        "entry_open": entry_open,
        "exit_open": exit_open,
        "return": trade_ret,
        "success": trade_ret > 0,
        "trigger": trigger or "time_exit",
    }


def summarize_trades(df: pd.DataFrame, variant: str) -> dict:
    sub = df.loc[df["variant"] == variant].copy()
    if sub.empty:
        return {
            "variant": variant,
            "trade_count": 0,
            "avg_return": np.nan,
            "success_rate": np.nan,
        }
    return {
        "variant": variant,
        "trade_count": int(len(sub)),
        "avg_return": float(sub["return"].mean()),
        "success_rate": float(sub["success"].mean()),
    }


def main():
    signals = pd.read_csv(SIGNAL_FILE, parse_dates=["date"])
    signals = signals[(signals["date"] < EXCLUDE_START) | (signals["date"] > EXCLUDE_END)].copy()
    signals = attach_negative_features(signals)
    signals.to_csv(OUT_DIR / "all_signals_with_negative.csv", index=False)

    selected_variants = []
    for variant in [
        "baseline",
        "exclude_stair_volume",
        "exclude_fast_rise",
        "exclude_failed_breakout",
        "exclude_any_negative",
    ]:
        selected = select_daily_variant(signals, variant)
        selected_variants.append(selected)
    selected_all = pd.concat(selected_variants, ignore_index=True)
    selected_all.to_csv(OUT_DIR / "selected_variants.csv", index=False)

    trade_rows: List[dict] = []
    for idx, code in enumerate(sorted(selected_all["code"].unique()), 1):
        file_path = DATA_DIR / f"{code}.txt"
        if not file_path.exists():
            continue
        code_df_raw, err = stoploss.load_data(str(file_path))
        if err or code_df_raw is None or len(code_df_raw) < MIN_BARS:
            continue
        code_df = pd.DataFrame(
            {
                "date": pd.to_datetime(code_df_raw["日期"]),
                "open": code_df_raw["开盘"].astype(float),
                "high": code_df_raw["最高"].astype(float),
                "low": code_df_raw["最低"].astype(float),
                "close": code_df_raw["收盘"].astype(float),
            }
        )
        code_df = code_df[(code_df["date"] < EXCLUDE_START) | (code_df["date"] > EXCLUDE_END)].reset_index(drop=True)
        for _, row in selected_all.loc[selected_all["code"] == code].iterrows():
            trade = simulate_trade_for_signal(row, code_df)
            if trade is not None:
                trade_rows.append(trade)
        if idx % 200 == 0:
            print(f"交易回放进度: {idx}/{selected_all['code'].nunique()}")

    trades = pd.DataFrame(trade_rows)
    trades.to_csv(OUT_DIR / "trade_results.csv", index=False)

    summary_rows = []
    for variant in [
        "baseline",
        "exclude_stair_volume",
        "exclude_fast_rise",
        "exclude_failed_breakout",
        "exclude_any_negative",
    ]:
        sel = selected_all.loc[selected_all["variant"] == variant]
        summ = summarize_trades(trades, variant)
        summ["selected_count"] = int(len(sel))
        summary_rows.append(summ)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False)

    summary = {
        "signal_count": int(len(signals)),
        "selected_total": int(len(selected_all)),
        "trade_count": int(len(trades)),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
