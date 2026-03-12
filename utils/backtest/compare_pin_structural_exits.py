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

from utils import stoploss, technical_indicators


DATA_DIR = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"
OUTPUT_DIR = "/Users/lidongyang/Desktop/Qstrategy/results/pin_structural_exits"
MIN_BARS = 160
TRADING_DAYS_PER_YEAR = 252
INITIAL_CAPITAL = 1_000_000.0
MAX_POSITIONS = 10
MAX_SINGLE_WEIGHT = 0.2
EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")
MAX_HOLD_DAYS = 3


def safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full(np.shape(a), np.nan, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-12)
    out[mask] = a[mask] / b[mask]
    return out


def build_feature_df(file_path: str) -> Optional[pd.DataFrame]:
    df, err = stoploss.load_data(file_path)
    if err or df is None or len(df) < MIN_BARS:
        return None
    df = df[(df["日期"] < EXCLUDE_START) | (df["日期"] > EXCLUDE_END)].copy()
    if len(df) < MIN_BARS:
        return None
    df = technical_indicators.calculate_trend(df)
    x = pd.DataFrame(
        {
            "date": df["日期"],
            "open": df["开盘"].astype(float),
            "high": df["最高"].astype(float),
            "low": df["最低"].astype(float),
            "close": df["收盘"].astype(float),
            "code": Path(file_path).stem,
            "trend_line": df["知行短期趋势线"].astype(float),
            "long_line": df["知行多空线"].astype(float),
        }
    ).reset_index(drop=True)

    trend_ok = x["trend_line"] > x["long_line"]
    short_llv = x["low"].rolling(3).min()
    short_hhv = x["close"].rolling(3).max()
    short_den = (short_hhv - short_llv).replace(0, np.nan)
    short_value = (x["close"] - short_llv) / short_den * 100
    long_llv = x["low"].rolling(21).min()
    long_hhv = x["close"].rolling(21).max()
    long_den = (long_hhv - long_llv).replace(0, np.nan)
    long_value = (x["close"] - long_llv) / long_den * 100
    pin_signal = (short_value <= 30) & (long_value >= 85)

    full_range = (x["high"] - x["low"]).replace(0, np.nan)
    body_low = np.minimum(x["open"], x["close"])
    x["lower_shadow_ratio"] = safe_div(body_low - x["low"], full_range)
    x["trend_slope_3"] = safe_div(x["trend_line"], x["trend_line"].shift(3)) - 1.0
    x["signal_mask"] = (
        trend_ok.fillna(False)
        & pin_signal.fillna(False)
        & (x["lower_shadow_ratio"] <= 0.05)
        & (x["trend_slope_3"] > 0.008)
    )
    x["sort_score"] = (
        (1.0 - x["lower_shadow_ratio"].clip(lower=0, upper=1)).fillna(0.0) * 0.55
        + x["trend_slope_3"].clip(lower=-0.05, upper=0.05).fillna(0.0) * 10 * 0.45
    )
    return x


def load_feature_map() -> Dict[str, pd.DataFrame]:
    feature_map: Dict[str, pd.DataFrame] = {}
    files = [p for p in Path(DATA_DIR).glob("*.txt")]
    for idx, file_path in enumerate(files, 1):
        df = build_feature_df(str(file_path))
        if df is not None:
            feature_map[str(df["code"].iloc[0])] = df
        if idx % 500 == 0:
            print(f"特征进度: {idx}/{len(files)}")
    return feature_map


def simulate_trade(df: pd.DataFrame, signal_idx: int, exit_name: str) -> Optional[dict]:
    entry_idx = signal_idx + 1
    scheduled_exit_idx = signal_idx + 1 + MAX_HOLD_DAYS
    if scheduled_exit_idx >= len(df):
        return None
    entry_price = float(df.at[entry_idx, "open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    signal_close = float(df.at[signal_idx, "close"])
    signal_high = float(df.at[signal_idx, "high"])
    signal_low = float(df.at[signal_idx, "low"])
    exit_idx = scheduled_exit_idx
    exit_price = float(df.at[exit_idx, "open"])
    exit_reason = "time_exit"

    for j in range(entry_idx + 1, scheduled_exit_idx + 1):
        next_idx = min(j + 1, len(df) - 1)
        if next_idx <= entry_idx:
            continue

        if exit_name == "day1_not_above_signal_close":
            if j == entry_idx + 1 and float(df.at[j, "close"]) <= signal_close:
                px = float(df.at[next_idx, "open"])
                if np.isfinite(px) and px > 0:
                    exit_idx = next_idx
                    exit_price = px
                    exit_reason = exit_name
                    break

        elif exit_name == "day2_no_break_signal_high":
            if j == entry_idx + 2 and float(df.at[j, "high"]) <= signal_high:
                px = float(df.at[next_idx, "open"])
                if np.isfinite(px) and px > 0:
                    exit_idx = next_idx
                    exit_price = px
                    exit_reason = exit_name
                    break

        elif exit_name == "close_below_signal_low":
            if float(df.at[j, "close"]) < signal_low:
                px = float(df.at[next_idx, "open"])
                if np.isfinite(px) and px > 0:
                    exit_idx = next_idx
                    exit_price = px
                    exit_reason = exit_name
                    break

        elif exit_name == "close_below_trend_line":
            if float(df.at[j, "close"]) < float(df.at[j, "trend_line"]):
                px = float(df.at[next_idx, "open"])
                if np.isfinite(px) and px > 0:
                    exit_idx = next_idx
                    exit_price = px
                    exit_reason = exit_name
                    break

        elif exit_name == "close_below_long_line":
            if float(df.at[j, "close"]) < float(df.at[j, "long_line"]):
                px = float(df.at[next_idx, "open"])
                if np.isfinite(px) and px > 0:
                    exit_idx = next_idx
                    exit_price = px
                    exit_reason = exit_name
                    break

        elif exit_name == "baseline_hold_3":
            pass

    ret = exit_price / entry_price - 1.0
    return {
        "signal_date": df.at[signal_idx, "date"],
        "entry_date": df.at[entry_idx, "date"],
        "exit_date": df.at[exit_idx, "date"],
        "code": df.at[signal_idx, "code"],
        "ret": ret,
        "success": ret > 0,
        "sort_score": float(df.at[signal_idx, "sort_score"]),
        "exit_reason": exit_reason,
    }


def build_trade_df(feature_map: Dict[str, pd.DataFrame], exit_name: str) -> pd.DataFrame:
    trades: List[dict] = []
    for _, df in feature_map.items():
        for signal_idx in np.flatnonzero(df["signal_mask"].to_numpy()):
            trade = simulate_trade(df, int(signal_idx), exit_name)
            if trade is not None:
                trades.append(trade)
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame(trades).sort_values(["signal_date", "sort_score", "code"], ascending=[True, False, True]).reset_index(drop=True)


def build_portfolio_curve(trade_df: pd.DataFrame) -> pd.DataFrame:
    if trade_df.empty:
        return pd.DataFrame(columns=["signal_date", "portfolio_ret", "equity"])
    rows = []
    equity = INITIAL_CAPITAL
    x = trade_df.copy()
    x["signal_date"] = pd.to_datetime(x["signal_date"])
    for signal_date, group in x.groupby("signal_date", sort=True):
        g = group.head(MAX_POSITIONS).copy()
        score = g["sort_score"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
        if score.sum() <= 0:
            weights = np.repeat(1 / len(g), len(g))
        else:
            weights = (score / score.sum()).clip(upper=MAX_SINGLE_WEIGHT).to_numpy()
            weights = weights / weights.sum()
        basket_ret = float(np.sum(g["ret"].to_numpy() * weights))
        equity *= 1.0 + basket_ret
        rows.append({"signal_date": signal_date, "portfolio_ret": basket_ret, "equity": equity})
    return pd.DataFrame(rows)


def max_consecutive_failures(flags: List[bool]) -> int:
    cur = 0
    worst = 0
    for flag in flags:
        if flag:
            cur = 0
        else:
            cur += 1
            worst = max(worst, cur)
    return worst


def summarize(name: str, trade_df: pd.DataFrame, portfolio_df: pd.DataFrame) -> dict:
    if portfolio_df.empty:
        annual = np.nan
        max_dd = np.nan
        final_equity = np.nan
        days = 0
    else:
        eq = portfolio_df["equity"].astype(float)
        max_dd = float((eq / eq.cummax() - 1.0).min())
        final_equity = float(eq.iloc[-1])
        days = len(portfolio_df)
        annual = float((final_equity / INITIAL_CAPITAL) ** (TRADING_DAYS_PER_YEAR / days) - 1) if final_equity > 0 and days > 0 else np.nan
    return {
        "name": name,
        "sample_count": int(len(trade_df)),
        "avg_trade_return": float(trade_df["ret"].mean()) if not trade_df.empty else np.nan,
        "success_rate": float(trade_df["success"].mean()) if not trade_df.empty else np.nan,
        "max_consecutive_failures": int(max_consecutive_failures(trade_df["success"].tolist())) if not trade_df.empty else np.nan,
        "annual_return": annual,
        "max_drawdown": max_dd,
        "equity_days": int(days),
        "final_equity": final_equity,
    }


def main() -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    feature_map = load_feature_map()
    exit_names = [
        "baseline_hold_3",
        "day1_not_above_signal_close",
        "day2_no_break_signal_high",
        "close_below_signal_low",
        "close_below_trend_line",
        "close_below_long_line",
    ]
    rows = []
    for exit_name in exit_names:
        trade_df = build_trade_df(feature_map, exit_name)
        portfolio_df = build_portfolio_curve(trade_df)
        rows.append(summarize(exit_name, trade_df, portfolio_df))
        print(f"退出策略完成: {exit_name}")
    result_df = pd.DataFrame(rows).sort_values(["annual_return", "max_drawdown"], ascending=[False, False]).reset_index(drop=True)
    result_df.to_csv(os.path.join(OUTPUT_DIR, "comparison.csv"), index=False, encoding="utf-8-sig")
    summary = {"max_hold_days": MAX_HOLD_DAYS, "results": result_df.to_dict(orient="records")}
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
