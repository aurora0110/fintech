from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from core.data_loader import load_price_directory
from core.engine import BacktestEngine, BaseSignalStrategy, EngineConfig
from core.models import Signal


BRICK_FILTER_PATH = Path("/Users/lidongyang/Desktop/Qstrategy/utils/brick_filter.py")


def load_brick_filter():
    spec = importlib.util.spec_from_file_location("brick_filter", BRICK_FILTER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@dataclass
class BrickFilterConfig:
    mode: str
    initial_capital: float = 1_000_000.0
    max_positions: int = 10
    hold_days: int = 3
    stop_buffer: float = 0.99


class BrickFilterStrategy(BaseSignalStrategy):
    def __init__(self, config: BrickFilterConfig, selected_df: pd.DataFrame):
        self.config = config
        self.name = f"BRICK_FILTER_{config.mode.upper()}"
        self.selected_df = selected_df.copy()

    def prepare(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        return stock_data

    def generate_signals(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[pd.Timestamp, List[Signal]]:
        signal_map: Dict[pd.Timestamp, List[Signal]] = {}
        if self.selected_df.empty:
            return signal_map
        for row in self.selected_df.itertuples(index=False):
            dt = pd.Timestamp(row.date)
            signal_map.setdefault(
                dt,
                [],
            ).append(
                Signal(
                    date=dt,
                    code=row.code,
                    score=float(row.sort_score),
                    reason=f"brick_filter_{self.config.mode}",
                    metadata={
                        "signal_low": float(row.signal_low),
                        "sort_score": float(row.sort_score),
                    },
                )
            )
        return signal_map

    def build_position(self, signal: Signal, exec_row: pd.Series) -> Dict[str, float]:
        signal_low = float(signal.metadata.get("signal_low", exec_row["low"]))
        return {
            "stop_price": signal_low * self.config.stop_buffer,
            "max_holding_days": self.config.hold_days,
        }


def export_trades(trades: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["code", "entry_date", "exit_date", "entry_price", "exit_price", "shares", "pnl", "return_pct", "reason"],
        )
        writer.writeheader()
        for trade in trades:
            writer.writerow(
                {
                    "code": trade.code,
                    "entry_date": trade.entry_date.strftime("%Y-%m-%d"),
                    "exit_date": trade.exit_date.strftime("%Y-%m-%d"),
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "shares": trade.shares,
                    "pnl": trade.pnl,
                    "return_pct": trade.return_pct,
                    "reason": trade.reason,
                }
            )


def run_single_mode(brick_filter, data_dir: str, mode: str, output_dir: Path) -> dict:
    signal_df = brick_filter.build_signal_df(Path(data_dir), mode=mode)
    stock_data, all_dates = load_price_directory(data_dir)
    signal_df = signal_df[signal_df["code"].isin(set(stock_data))].copy()
    selected_df = brick_filter.apply_selection(signal_df, mode=mode)
    config = BrickFilterConfig(mode=mode)
    strategy = BrickFilterStrategy(config, selected_df)
    engine = BacktestEngine(EngineConfig(initial_capital=config.initial_capital, max_positions=config.max_positions))
    result = engine.run(strategy, stock_data, all_dates)

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_df.to_csv(output_dir / f"selected_signals_{mode}.csv", index=False)
    export_trades(result.trades, output_dir / f"trades_{mode}.csv")
    result.equity_curve.rename("equity").to_csv(output_dir / f"equity_{mode}.csv", encoding="utf-8")

    returns = [trade.return_pct for trade in result.trades]
    summary = {
        "mode": mode,
        "signal_count": int(len(signal_df)),
        "selected_signal_count": int(len(selected_df)),
        "trade_count": int(len(result.trades)),
        "win_rate": float(sum(1 for value in returns if value > 0) / len(returns)) if returns else 0.0,
        "avg_trade_return": float(np.mean(returns)) if returns else 0.0,
        "metrics": result.metrics,
        "diagnostics": result.diagnostics,
    }
    (output_dir / f"summary_{mode}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest legacy vs perfect brick_filter")
    parser.add_argument("data_dir", nargs="?", default="/Users/lidongyang/Desktop/Qstrategy/data/20260311/normal")
    parser.add_argument("--output-dir", default="/Users/lidongyang/Desktop/Qstrategy/results/brick_filter_backtest_20260312")
    parser.add_argument("--mode", choices=["legacy", "perfect", "both"], default="both")
    args = parser.parse_args()

    brick_filter = load_brick_filter()
    output_dir = Path(args.output_dir)

    summaries = []
    modes = ("legacy", "perfect") if args.mode == "both" else (args.mode,)
    for mode in modes:
        summaries.append(run_single_mode(brick_filter, args.data_dir, mode, output_dir))

    summary_df = pd.DataFrame(
        [
            {
                "mode": item["mode"],
                "signal_count": item["signal_count"],
                "selected_signal_count": item["selected_signal_count"],
                "trade_count": item["trade_count"],
                "win_rate": item["win_rate"],
                "avg_trade_return": item["avg_trade_return"],
                "final_multiple": item["metrics"]["final_multiple"],
                "annual_return": item["metrics"]["annual_return"],
                "max_drawdown": item["metrics"]["max_drawdown"],
                "sharpe": item["metrics"]["sharpe"],
            }
            for item in summaries
        ]
    )
    summary_df.to_csv(output_dir / "summary_table.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
