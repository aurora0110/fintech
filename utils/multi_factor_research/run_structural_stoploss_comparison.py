from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from utils.multi_factor_research.data_processor import load_stock_directory
from utils.multi_factor_research.research_metrics import summarize_trade_metrics
from utils.multi_factor_research.run_four_layer_scorecard_backtest import (
    build_scorecard_signals,
    summarize_score_groups,
)
from utils.multi_factor_research.factor_calculator import build_prepared_stock_data


def _simulate_structural_exit(
    df: pd.DataFrame,
    signal_idx: int,
    stop_rule: str,
    max_holding_days: int,
) -> dict | None:
    if signal_idx + 1 >= len(df):
        return None

    entry_idx = signal_idx + 1
    entry_price = float(df.at[entry_idx, "open"])
    if pd.isna(entry_price) or entry_price <= 0:
        return None

    end_idx = min(len(df) - 1, entry_idx + max_holding_days - 1)
    exit_idx = end_idx
    exit_reason = f"hold_{max_holding_days}d"

    if stop_rule == "double_break_bull_bear":
        for idx in range(entry_idx, end_idx + 1):
            if idx < 1:
                continue
            cond = bool(df.at[idx, "close"] < df.at[idx, "bull_bear_line"]) and bool(
                df.at[idx - 1, "close"] < df.at[idx - 1, "bull_bear_line"]
            )
            if cond:
                exit_idx = idx
                exit_reason = "double_break_bull_bear_stop"
                break
    elif stop_rule == "break_key_k_low":
        for idx in range(entry_idx, end_idx + 1):
            key_low = df.at[idx, "关键K最低价"]
            if pd.notna(key_low) and bool(df.at[idx, "close"] < key_low):
                exit_idx = idx
                exit_reason = "break_key_k_low_stop"
                break
    elif stop_rule == "break_trend_and_bull_bear":
        for idx in range(entry_idx, end_idx + 1):
            cond = bool(df.at[idx, "close"] < df.at[idx, "trend_line"]) and bool(
                df.at[idx, "close"] < df.at[idx, "bull_bear_line"]
            )
            if cond:
                exit_idx = idx
                exit_reason = "break_trend_and_bull_bear_stop"
                break
    else:
        raise ValueError(f"Unsupported stop rule: {stop_rule}")

    exit_price = float(df.at[exit_idx, "close"])
    if pd.isna(exit_price) or exit_price <= 0:
        return None

    return {
        "entry_date": df.at[entry_idx, "date"],
        "exit_date": df.at[exit_idx, "date"],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "return_pct": exit_price / entry_price - 1.0,
        "exit_reason": exit_reason,
        "holding_days": int(exit_idx - entry_idx + 1),
        "success": float(exit_price > entry_price),
    }


def simulate_structural_stoplosses(
    prepared_stock_data: Dict[str, pd.DataFrame],
    signal_candidates: pd.DataFrame,
    stop_rule: str,
    max_holding_days: int,
) -> pd.DataFrame:
    records: List[dict] = []
    grouped = signal_candidates.groupby("code", sort=False)
    for code, code_rows in grouped:
        df = prepared_stock_data[code]
        for row in code_rows.itertuples(index=False):
            trade = _simulate_structural_exit(df, int(row.signal_idx), stop_rule=stop_rule, max_holding_days=max_holding_days)
            if trade is None:
                continue
            record = row._asdict()
            record.update(trade)
            record["exit_model"] = stop_rule
            records.append(record)
    return pd.DataFrame(records) if records else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare structural stop-loss rules on four-layer scorecard signals")
    parser.add_argument("data_dir", nargs="?", default="/Users/lidongyang/Desktop/Qstrategy/data/forward_data")
    parser.add_argument("--burst-window", type=int, default=20)
    parser.add_argument("--top-quantile", type=float, default=0.30)
    parser.add_argument("--max-holding-days", type=int, default=30)
    parser.add_argument("--output-dir", default="results/structural_stoploss_comparison")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stock_data = load_stock_directory(args.data_dir)
    prepared_stock_data = build_prepared_stock_data(stock_data, burst_window=args.burst_window)
    scored_stock_data, signal_candidates = build_scorecard_signals(prepared_stock_data)
    signal_candidates.to_csv(output_dir / "signal_candidates.csv", index=False, encoding="utf-8-sig")

    stop_rules = {
        "double_break_bull_bear": "连续两天收盘跌破多空线",
        "break_key_k_low": "收盘跌破关键K最低价",
        "break_trend_and_bull_bear": "收盘跌破趋势线且跌破多空线",
    }

    summaries = []
    for stop_rule, label in stop_rules.items():
        dataset = simulate_structural_stoplosses(
            prepared_stock_data=scored_stock_data,
            signal_candidates=signal_candidates,
            stop_rule=stop_rule,
            max_holding_days=args.max_holding_days,
        )
        if dataset.empty:
            continue
        score_summary = summarize_score_groups(dataset, args.top_quantile)
        metrics = summarize_trade_metrics(dataset)
        run_dir = output_dir / stop_rule
        run_dir.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(run_dir / "trades.csv", index=False, encoding="utf-8-sig")
        summary = {
            "stop_rule": stop_rule,
            "stop_rule_cn": label,
            "signal_count": int(len(dataset)),
            "metrics": metrics,
            "score_summary": score_summary,
            "stop_trigger_rate": float(dataset["exit_reason"].ne(f"hold_{args.max_holding_days}d").mean()),
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries.append(summary)

    final_summary = {
        "signal_candidate_count": int(len(signal_candidates)),
        "max_holding_days": args.max_holding_days,
        "runs": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
