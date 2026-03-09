from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from core.metrics import compute_metrics
from utils.multi_factor_research.data_processor import load_stock_directory
from utils.multi_factor_research.research_metrics import summarize_trade_metrics
from strategies.common import base_prepare


def prepare_stock(df: pd.DataFrame) -> pd.DataFrame:
    out = base_prepare(df.copy())
    out["cross_up"] = out["short_trend"].gt(out["long_trend"]) & out["short_trend"].shift(1).le(out["long_trend"].shift(1))
    out["cross_down"] = out["short_trend"].lt(out["long_trend"]) & out["short_trend"].shift(1).ge(out["long_trend"].shift(1))
    prior_cross_up = out["cross_up"].shift(1).rolling(window=30, min_periods=1).max().fillna(False).astype(bool)
    out["first_cross_up_30"] = out["cross_up"] & ~prior_cross_up
    return out


def simulate_trades(stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    records: List[dict] = []

    for code, raw_df in stock_data.items():
        df = prepare_stock(raw_df.reset_index(drop=True))
        idx = 0
        while idx < len(df) - 1:
            if not bool(df.at[idx, "first_cross_up_30"]):
                idx += 1
                continue

            entry_idx = idx + 1
            entry_price = float(df.at[entry_idx, "open"])
            if pd.isna(entry_price) or entry_price <= 0:
                idx += 1
                continue

            sell_signal_idx = None
            for j in range(entry_idx, len(df)):
                if bool(df.at[j, "cross_down"]):
                    sell_signal_idx = j
                    break

            if sell_signal_idx is None:
                exit_idx = len(df) - 1
                exit_price = float(df.at[exit_idx, "close"])
                exit_reason = "end_of_data"
            else:
                if sell_signal_idx + 1 < len(df):
                    exit_idx = sell_signal_idx + 1
                    exit_price = float(df.at[exit_idx, "open"])
                else:
                    exit_idx = sell_signal_idx
                    exit_price = float(df.at[exit_idx, "close"])
                exit_reason = "cross_down"

            if pd.isna(exit_price) or exit_price <= 0:
                idx += 1
                continue

            records.append(
                {
                    "code": code,
                    "signal_date": df.at[idx, "date"],
                    "entry_date": df.at[entry_idx, "date"],
                    "exit_date": df.at[exit_idx, "date"],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return_pct": exit_price / entry_price - 1.0,
                    "holding_days": int(exit_idx - entry_idx + 1),
                    "exit_reason": exit_reason,
                    "success": float(exit_price > entry_price),
                }
            )
            idx = exit_idx + 1

    return pd.DataFrame(records).sort_values(["entry_date", "code"]).reset_index(drop=True) if records else pd.DataFrame()


def build_trade_equity(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    equity = (1.0 + trades["return_pct"].clip(lower=-0.999999)).cumprod()
    equity.index = pd.to_datetime(trades["exit_date"])
    return equity


def main() -> None:
    parser = argparse.ArgumentParser(description="Trend cross strategy backtest")
    parser.add_argument("data_dir", nargs="?", default="/Users/lidongyang/Desktop/Qstrategy/data/forward_data")
    parser.add_argument("--output-dir", default="results/trend_cross_strategy")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stock_data = load_stock_directory(args.data_dir)
    trades = simulate_trades(stock_data)
    trades.to_csv(output_dir / "trades.csv", index=False, encoding="utf-8-sig")

    trade_metrics = summarize_trade_metrics(trades) if not trades.empty else {}
    equity_metrics = compute_metrics(build_trade_equity(trades)) if not trades.empty else {}

    summary = {
        "strategy": "30日内第一次趋势线上穿多空线买入，买入后第一次趋势线下穿多空线卖出",
        "trade_count": int(len(trades)),
        "trade_metrics": trade_metrics,
        "equity_metrics": equity_metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
