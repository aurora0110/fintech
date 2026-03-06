from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from core.data_loader import load_price_directory
from core.engine import BacktestEngine, EngineConfig
from strategies.unified_strategies import B1Strategy, B2Strategy, B3Strategy, BrickStrategy, PinStrategy


def build_strategy(name: str, args: argparse.Namespace):
    key = name.upper()
    if key == "B1":
        return B1Strategy(use_bullish_ma=args.use_bullish_ma)
    if key == "B2":
        return B2Strategy(volume_multiplier=args.volume_multiplier)
    if key == "B3":
        return B3Strategy()
    if key == "PIN":
        return PinStrategy()
    if key == "BRICK":
        return BrickStrategy()
    raise ValueError(f"Unsupported strategy: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified A-share backtest runner")
    parser.add_argument("strategy", choices=["B1", "B2", "B3", "PIN", "BRICK"])
    parser.add_argument("data_dir")
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--initial-capital", type=float, default=1_000_000)
    parser.add_argument("--output", default="results/unified_backtest_summary.json")
    parser.add_argument("--trades-output", default="")
    parser.add_argument("--equity-output", default="")
    parser.add_argument("--signals-output", default="")
    parser.add_argument("--use-bullish-ma", action="store_true")
    parser.add_argument("--volume-multiplier", type=float, default=2.0)
    args = parser.parse_args()

    stock_data, all_dates = load_price_directory(args.data_dir)
    strategy = build_strategy(args.strategy, args)
    engine = BacktestEngine(
        EngineConfig(
            initial_capital=args.initial_capital,
            max_positions=args.max_positions,
        )
    )
    result = engine.run(strategy, stock_data, all_dates)

    payload = {
        "strategy": strategy.name,
        "metrics": result.metrics,
        "diagnostics": result.diagnostics,
        "trade_count": len(result.trades),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.trades_output:
        trades_path = Path(args.trades_output)
        trades_path.parent.mkdir(parents=True, exist_ok=True)
        with trades_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["code", "entry_date", "exit_date", "entry_price", "exit_price", "shares", "pnl", "return_pct", "reason"],
            )
            writer.writeheader()
            for trade in result.trades:
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

    if args.equity_output:
        equity_path = Path(args.equity_output)
        equity_path.parent.mkdir(parents=True, exist_ok=True)
        result.equity_curve.rename("equity").to_csv(equity_path, encoding="utf-8")

    if args.signals_output:
        signals_path = Path(args.signals_output)
        signals_path.parent.mkdir(parents=True, exist_ok=True)
        with signals_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["signal_date", "code", "score", "reason"])
            writer.writeheader()
            for signal in result.signals:
                writer.writerow(
                    {
                        "signal_date": signal.date.strftime("%Y-%m-%d"),
                        "code": signal.code,
                        "score": signal.score,
                        "reason": signal.reason,
                    }
                )

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
