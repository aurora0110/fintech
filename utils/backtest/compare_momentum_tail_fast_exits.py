from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


BASE_SCRIPT = Path("/Users/lidongyang/Desktop/Qstrategy/utils/backtest/run_momentum_tail_experiment.py")
OUTPUT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/results/momentum_tail_fast_exit_ab")


def load_base_module():
    spec = importlib.util.spec_from_file_location("momentum_tail_base_fast_exit", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def add_long_line(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["ma14"] = x["close"].rolling(14).mean()
    x["ma28"] = x["close"].rolling(28).mean()
    x["ma57"] = x["close"].rolling(57).mean()
    x["ma114"] = x["close"].rolling(114).mean()
    x["long_line"] = (x["ma14"] + x["ma28"] + x["ma57"] + x["ma114"]) / 4.0
    return x


def simulate_trade_with_fast_exit(df: pd.DataFrame, signal_idx: int, combo, fast_exit_mode: str) -> Optional[dict]:
    n = len(df)
    entry_idx = signal_idx + 1
    if entry_idx >= n:
        return None
    entry_price = float(df.at[entry_idx, "open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    signal_low = float(df.at[signal_idx, "low"])
    signal_high = float(df.at[signal_idx, "high"])
    entry_day_low = float(df.at[entry_idx, "low"])
    sl_price = base.stop_loss_price(signal_low, combo.stop_mode)
    target_price = entry_price * (1.0 + combo.take_profit)
    scheduled_exit_idx = min(entry_idx + 3 + 1, n - 1)
    exit_idx = scheduled_exit_idx
    exit_price = float(df.at[exit_idx, "open"])
    exit_reason = "time_exit_next_open"

    for j in range(entry_idx + 1, min(entry_idx + 3, n - 1) + 1):
        next_idx = min(j + 1, n - 1)
        if sl_price is not None and float(df.at[j, "low"]) <= sl_price and next_idx > entry_idx:
            px = float(df.at[next_idx, "open"])
            if np.isfinite(px) and px > 0:
                exit_idx = next_idx
                exit_price = px
                exit_reason = "stop_loss_next_open"
                break
        if float(df.at[j, "high"]) >= target_price and next_idx > entry_idx:
            px = float(df.at[next_idx, "open"])
            if np.isfinite(px) and px > 0:
                exit_idx = next_idx
                exit_price = px
                exit_reason = "take_profit_next_open"
                break

        if fast_exit_mode == "day1_not_up_next_open" and j == entry_idx + 1:
            if float(df.at[j, "close"]) <= entry_price and next_idx > entry_idx:
                px = float(df.at[next_idx, "open"])
                if np.isfinite(px) and px > 0:
                    exit_idx = next_idx
                    exit_price = px
                    exit_reason = "fast_exit_day1_not_up"
                    break

        if fast_exit_mode == "day2_no_new_high_next_open" and j == entry_idx + 2:
            highest_high = float(df.loc[entry_idx + 1 : j, "high"].max())
            if highest_high <= signal_high and next_idx > entry_idx:
                px = float(df.at[next_idx, "open"])
                if np.isfinite(px) and px > 0:
                    exit_idx = next_idx
                    exit_price = px
                    exit_reason = "fast_exit_day2_no_new_high"
                    break

        if fast_exit_mode == "close_below_entry_day_low_next_open":
            if float(df.at[j, "close"]) < entry_day_low and next_idx > entry_idx:
                px = float(df.at[next_idx, "open"])
                if np.isfinite(px) and px > 0:
                    exit_idx = next_idx
                    exit_price = px
                    exit_reason = "fast_exit_close_below_entry_day_low"
                    break

    ret = exit_price / entry_price - 1.0
    return {
        "signal_date": df.at[signal_idx, "date"],
        "entry_date": df.at[entry_idx, "date"],
        "exit_date": df.at[exit_idx, "date"],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "ret": ret,
        "holding_days": int(exit_idx - entry_idx),
        "success": ret > 0,
        "exit_reason": exit_reason,
        "signal_low": signal_low,
        "pattern_a": bool(df.at[signal_idx, "pattern_a"]),
        "pattern_b": bool(df.at[signal_idx, "pattern_b"]),
    }


def build_trade_df(feature_map: Dict[str, pd.DataFrame], combo, fast_exit_mode: str) -> pd.DataFrame:
    trades: List[dict] = []
    for code, raw_df in feature_map.items():
        df = add_long_line(raw_df)
        mask_a = df["pattern_a"] & (df["rebound_ratio"] >= combo.rebound_threshold)
        mask_b = df["pattern_b"] & (df["rebound_ratio"] >= 1.0)
        mask = (
            df["signal_base"]
            & (df["ret1"] <= combo.gain_limit)
            & (mask_a | mask_b)
            & (df["trend_line"] > df["long_line"])
        )
        for signal_idx in np.flatnonzero(mask.to_numpy()):
            trade = simulate_trade_with_fast_exit(df, int(signal_idx), combo, fast_exit_mode)
            if trade is None:
                continue
            trade["code"] = code
            trade["sort_score"] = float(df.at[int(signal_idx), "rebound_ratio"] / max(abs(float(df.at[int(signal_idx), "ret1"])), 0.01))
            trades.append(trade)
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame(trades).sort_values(["signal_date", "code"]).reset_index(drop=True)


base = load_base_module()


def run_fast_exit_experiment() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_map = base.load_feature_map(base.DATA_DIR)
    combo = base.Combo(rebound_threshold=1.2, gain_limit=0.08, take_profit=0.03, stop_mode="entry_low_x_0.99")
    scenarios = [
        "baseline",
        "day1_not_up_next_open",
        "day2_no_new_high_next_open",
        "close_below_entry_day_low_next_open",
    ]
    rows = []
    for scenario_name in scenarios:
        trade_df = build_trade_df(feature_map, combo, scenario_name)
        portfolio_df = base.build_portfolio_curve(trade_df)
        summary = base.summarize_combo(combo, trade_df, portfolio_df)
        summary["scenario_name"] = scenario_name
        rows.append(summary)
    result_df = pd.DataFrame(rows).sort_values(["annual_return", "max_drawdown"], ascending=[False, False]).reset_index(drop=True)
    summary = {
        "tested_combo": {
            "rebound_threshold": combo.rebound_threshold,
            "gain_limit": combo.gain_limit,
            "take_profit": combo.take_profit,
            "stop_mode": combo.stop_mode,
            "require_trend_above_long": True,
            "signal_vs_ma5_filter": "1.3~2.2",
        },
        "fast_exit_design": {
            "baseline": "固定持有3天，到期次日开盘卖出",
            "day1_not_up_next_open": "买入后第1天收盘不高于买入价，则次日开盘卖出",
            "day2_no_new_high_next_open": "买入后前2天都没有突破信号日最高价，则次日开盘卖出",
            "close_below_entry_day_low_next_open": "买入后任一天收盘跌破买入日最低价，则次日开盘卖出",
        },
        "results": result_df.to_dict(orient="records"),
    }
    result_df.to_csv(OUTPUT_DIR / "comparison.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_fast_exit_experiment(), ensure_ascii=False, indent=2))
