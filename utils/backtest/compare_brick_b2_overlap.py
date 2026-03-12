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

from utils.backtest.run_b2_startup_experiment import load_one_csv as load_b2_csv
from utils.backtest.run_b2_startup_experiment import build_feature_df as build_b2_features


DATA_DIR = os.environ.get("OVERLAP_DATA_DIR", "/Users/lidongyang/Desktop/Qstrategy/data/forward_data")
OUTPUT_DIR = os.environ.get("OVERLAP_OUTPUT_DIR", "/Users/lidongyang/Desktop/Qstrategy/results/brick_b2_overlap")
MAX_FILES = int(os.environ.get("OVERLAP_MAX_FILES", "0"))
INITIAL_CAPITAL = 1_000_000.0
TRADING_DAYS_PER_YEAR = 252
MAX_POSITIONS = 10
MAX_SINGLE_WEIGHT = 0.2
PCT_RANK_THRESHOLD = 0.50


def safe_div(a, b, default=np.nan):
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    out = np.full(np.shape(a_arr), default, dtype=float)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr) & (np.abs(b_arr) > 1e-12)
    out[mask] = a_arr[mask] / b_arr[mask]
    return out


def tdx_sma(series: pd.Series, n: int, m: int) -> pd.Series:
    return series.ewm(alpha=m / n, adjust=False).mean()


def calc_green_streak(green_flag: np.ndarray) -> np.ndarray:
    out = np.zeros(len(green_flag), dtype=np.int32)
    for i in range(1, len(green_flag)):
        out[i] = out[i - 1] + 1 if green_flag[i] else 0
    return out


def build_brick_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().reset_index(drop=True)
    x["ret1"] = x["close"].pct_change()
    x["trend_line"] = x["close"].ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()
    x["ma14"] = x["close"].rolling(14).mean()
    x["ma28"] = x["close"].rolling(28).mean()
    x["ma57"] = x["close"].rolling(57).mean()
    x["ma114"] = x["close"].rolling(114).mean()
    x["long_line"] = (x["ma14"] + x["ma28"] + x["ma57"] + x["ma114"]) / 4.0

    # 主板/创业板/科创板分板块涨幅上限
    code = str(x["code"].iloc[0])
    if code.startswith("SH#688"):
        gain_limit = 0.08
    elif code.startswith("SZ#300"):
        gain_limit = 0.08
    else:
        gain_limit = 0.07
    x["gain_limit"] = gain_limit

    x["close_pullback_white"] = x["close"] < x["trend_line"] * 1.01
    x["close_above_white"] = x["close"] > x["trend_line"]

    x["vol_ma5_prev"] = x["volume"].shift(1).rolling(5).mean()
    x["signal_vs_ma5"] = safe_div(x["volume"], x["vol_ma5_prev"])
    x["signal_vs_ma5_valid"] = pd.Series(x["signal_vs_ma5"], index=x.index).between(1.3, 2.2, inclusive="both")

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
        x["close"].rolling(10).apply(
            lambda s: np.polyfit(np.arange(len(s)), s, 1)[0] if np.isfinite(s).all() else np.nan,
            raw=False,
        )
    )
    x["not_sideways"] = np.abs(safe_div(x["close_slope_10"], x["close"].rolling(10).mean())) > 0.002

    x["up_leg_avg_vol"] = x["volume"].shift(4).rolling(3).mean()
    x["pullback_avg_vol"] = x["volume"].shift(1).rolling(3).mean()
    x["pullback_shrinking"] = x["pullback_avg_vol"] < x["up_leg_avg_vol"]
    x["pullback_shrink_ratio"] = safe_div(x["pullback_avg_vol"], x["up_leg_avg_vol"])

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
    mask_a = x["pattern_a"] & (x["rebound_ratio"] >= 1.2)
    mask_b = x["pattern_b"] & (x["rebound_ratio"] >= 1.0)
    x["brick_signal"] = (
        (mask_a | mask_b)
        & x["pullback_shrinking"].fillna(False)
        & x["signal_vs_ma5_valid"].fillna(False)
        & x["not_sideways"].fillna(False)
        & (x["ret1"] <= x["gain_limit"])
        & (x["trend_line"] > x["long_line"])
    )
    trend_spread = np.maximum((x["trend_line"] - x["long_line"]) / x["close"], 0.0)
    rebound_rank_raw = pd.Series(x["rebound_ratio"], index=x.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    shrink_quality = 1.0 - np.minimum(np.abs(pd.Series(x["pullback_shrink_ratio"], index=x.index) - 0.8) / 0.3, 1.0)
    x["brick_sort_score"] = 0.50 * shrink_quality.fillna(0.0) + 0.30 * rebound_rank_raw + 0.20 * pd.Series(trend_spread, index=x.index).fillna(0.0)
    return x


def load_feature_map(data_dir: str) -> Dict[str, pd.DataFrame]:
    feature_map: Dict[str, pd.DataFrame] = {}
    files = [f for f in os.listdir(data_dir) if f.lower().endswith((".csv", ".txt"))]
    if MAX_FILES > 0:
        files = files[:MAX_FILES]
    total = len(files)
    for idx, file_name in enumerate(files, 1):
        df = load_b2_csv(os.path.join(data_dir, file_name))
        if df is None:
            continue
        code = str(df["code"].iloc[0])
        brick_df = build_brick_features(df)
        b2_df = build_b2_features(df)
        merged = brick_df.copy()
        for col in ["基础信号", "双线附近启动", "sort_score", "信号最低点"]:
            merged[f"b2_{col}"] = b2_df[col]
        merged["b2_signal"] = b2_df["基础信号"] & b2_df["双线附近启动"]
        merged["b2_sort_score"] = b2_df["sort_score"]
        merged["b2_stop_low"] = b2_df["信号最低点"]
        feature_map[code] = merged
        if idx % 500 == 0 or idx == total:
            print(f"特征进度: {idx}/{total}")
    return feature_map


def simulate_trade(df: pd.DataFrame, signal_idx: int, group_name: str) -> Optional[dict]:
    entry_idx = signal_idx + 1
    exit_idx = signal_idx + 4
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
        "entry_price": entry_price,
        "exit_price": exit_price,
        "ret": ret,
        "holding_days": 3,
        "success": ret > 0,
        "sort_score": float(df.at[signal_idx, "brick_sort_score"] if group_name != "只满足B2" else df.at[signal_idx, "b2_sort_score"]),
    }


def build_portfolio_curve(trade_df: pd.DataFrame) -> pd.DataFrame:
    if trade_df.empty:
        return pd.DataFrame(columns=["signal_date", "portfolio_ret", "equity"])
    rows = []
    equity = INITIAL_CAPITAL
    trade_df = trade_df.copy()
    trade_df["signal_date"] = pd.to_datetime(trade_df["signal_date"])
    for signal_date, group in trade_df.groupby("signal_date", sort=True):
        g = group.copy().sort_values(["sort_score", "code"], ascending=[False, True])
        if len(g) > MAX_POSITIONS:
            g["pct_rank"] = g["sort_score"].rank(pct=True)
            g = g[g["pct_rank"] >= PCT_RANK_THRESHOLD].sort_values(["sort_score", "code"], ascending=[False, True]).head(MAX_POSITIONS)
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


def compute_equity_metrics(portfolio_df: pd.DataFrame) -> dict:
    if portfolio_df.empty:
        return {"annual_return": np.nan, "max_drawdown": np.nan, "equity_days": 0, "final_equity": np.nan}
    eq = portfolio_df["equity"].astype(float)
    running_max = eq.cummax()
    drawdown = eq / running_max - 1.0
    final_equity = float(eq.iloc[-1])
    days = len(portfolio_df)
    annual_return = (final_equity / INITIAL_CAPITAL) ** (TRADING_DAYS_PER_YEAR / days) - 1 if final_equity > 0 and days > 0 else np.nan
    return {
        "annual_return": float(annual_return),
        "max_drawdown": float(drawdown.min()),
        "equity_days": int(days),
        "final_equity": final_equity,
    }


def max_consecutive_failures(success_flags: List[bool]) -> int:
    current = 0
    worst = 0
    for flag in success_flags:
        if flag:
            current = 0
        else:
            current += 1
            worst = max(worst, current)
    return worst


def summarize(group_name: str, trade_df: pd.DataFrame, portfolio_df: pd.DataFrame) -> dict:
    row = {
        "group_name": group_name,
        "sample_count": int(len(trade_df)) if not trade_df.empty else 0,
        "avg_trade_return": float(trade_df["ret"].mean()) if not trade_df.empty else np.nan,
        "success_rate": float(trade_df["success"].mean()) if not trade_df.empty else np.nan,
        "max_consecutive_failures": int(max_consecutive_failures(trade_df["success"].tolist())) if not trade_df.empty else np.nan,
    }
    row.update(compute_equity_metrics(portfolio_df))
    return row


def main() -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    feature_map = load_feature_map(DATA_DIR)
    groups = ["只满足砖型", "只满足B2", "同时满足砖型和B2"]
    rows = []
    overlap_daily_rows = []

    for group_name in groups:
        trades = []
        for code, df in feature_map.items():
            brick_mask = df["brick_signal"].fillna(False)
            b2_mask = df["b2_signal"].fillna(False)
            if group_name == "只满足砖型":
                mask = brick_mask & ~b2_mask
            elif group_name == "只满足B2":
                mask = b2_mask & ~brick_mask
            else:
                mask = brick_mask & b2_mask
            idxs = np.flatnonzero(mask.to_numpy())
            for signal_idx in idxs:
                trade = simulate_trade(df, int(signal_idx), group_name)
                if trade is None:
                    continue
                trade["code"] = code
                trades.append(trade)
                overlap_daily_rows.append({"date": df.at[int(signal_idx), "date"], "code": code, "group_name": group_name})
        trade_df = pd.DataFrame(trades)
        portfolio_df = build_portfolio_curve(trade_df)
        rows.append(summarize(group_name, trade_df, portfolio_df))
        print(f"组别进度: {group_name}")

    result_df = pd.DataFrame(rows).sort_values(["annual_return", "max_drawdown"], ascending=[False, False]).reset_index(drop=True)
    result_df.to_csv(os.path.join(OUTPUT_DIR, "comparison.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(overlap_daily_rows).to_csv(os.path.join(OUTPUT_DIR, "signal_groups.csv"), index=False, encoding="utf-8-sig")
    summary = {
        "data_dir": DATA_DIR,
        "entry_exit": {
            "entry": "T+1_open",
            "holding_days": 3,
            "exit": "T+4_open",
        },
        "brick_signal": [
            "趋势线>多空线",
            "信号日量/前5日均量在1.3~2.2",
            "主板7%/创业板8%/科创板8%涨幅上限",
            "3绿1红(反包1.2)或3绿1红1绿1红",
        ],
        "b2_signal": [
            "双线启动",
            "当日涨幅>=4%",
            "上影线长度<=实体长度*0.3",
            "J<80 且J向上启动",
            "趋势线>多空线",
            "当日量>昨日量且>5日均量",
        ],
        "result_groups": result_df.to_dict(orient="records"),
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
