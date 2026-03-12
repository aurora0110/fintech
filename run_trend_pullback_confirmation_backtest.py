from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from core.data_loader import load_price_directory
from core.engine import BacktestEngine, BaseSignalStrategy, EngineConfig
from core.models import Signal
from strategies.common import calc_trend_cn


DISASTER_START = pd.Timestamp("2015-06-12")
DISASTER_END = pd.Timestamp("2024-09-18")


def add_prev_close(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["prev_close"] = out["close"].shift(1)
    return out


def add_pin_shape(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    low_3 = out["low"].rolling(window=3).min()
    close_high_3 = out["close"].rolling(window=3).max()
    low_21 = out["low"].rolling(window=21).min()
    close_high_21 = out["close"].rolling(window=21).max()
    short_denom = (close_high_3 - low_3).replace(0.0, np.nan)
    long_denom = (close_high_21 - low_21).replace(0.0, np.nan)
    out["pin_short"] = ((out["close"] - low_3) / short_denom) * 100.0
    out["pin_long"] = ((out["close"] - low_21) / long_denom) * 100.0
    out["pin_shape"] = out["pin_short"].le(30.0) & out["pin_long"].ge(85.0)
    return out


@dataclass
class PullbackConfig:
    name: str = "trend_pullback_confirm"
    initial_capital: float = 1_000_000.0
    max_positions: int = 10
    strong_5d_threshold: float = 0.03
    signal_day_drop_threshold: float = -0.02
    trend_slope_3_threshold: float = 0.0
    long_slope_5_threshold: float = 0.0
    touch_tolerance: float = 0.03
    close_above_long_tolerance: float = -0.03
    confirm_volume_ratio_max: float = 2.0
    max_stall_days: int = 7
    take_profit_pin_long: float = 100.0
    stop_pin_long: float = 80.0
    gem_score_bonus: float = 0.01
    recent_candidate_days: int = 4
    confirm_close_to_trend_tolerance: float = -0.005


class TrendPullbackConfirmationStrategy(BaseSignalStrategy):
    def __init__(self, config: PullbackConfig):
        self.config = config
        self.name = config.name

    def prepare(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        prepared: Dict[str, pd.DataFrame] = {}
        for code, raw_df in stock_data.items():
            df = add_prev_close(add_pin_shape(raw_df))
            df["short_trend"], df["long_trend"] = calc_trend_cn(df["close"])
            df["return_5d"] = df["close"].pct_change(5)
            df["signal_day_return"] = df["close"] / df["prev_close"] - 1.0
            df["volume_ratio_prev"] = df["volume"] / df["volume"].shift(1).replace(0.0, np.nan)
            df["trend_slope_3"] = df["short_trend"] / df["short_trend"].shift(3) - 1.0
            df["long_slope_5"] = df["long_trend"] / df["long_trend"].shift(5) - 1.0
            df["distance_to_trend"] = df["close"] / df["short_trend"] - 1.0
            df["distance_to_long"] = df["close"] / df["long_trend"] - 1.0

            df["trend_ok"] = (
                df["short_trend"].gt(df["long_trend"])
                & df["trend_slope_3"].gt(self.config.trend_slope_3_threshold)
                & df["long_slope_5"].gt(self.config.long_slope_5_threshold)
                & df["return_5d"].gt(self.config.strong_5d_threshold)
            )
            df["pullback_touch"] = (
                df["low"].le(df["short_trend"] * (1.0 + self.config.touch_tolerance))
                | df["low"].le(df["long_trend"] * (1.0 + self.config.touch_tolerance))
            )
            df["pullback_structure_ok"] = (
                df["pin_shape"]
                & df["signal_day_return"].le(self.config.signal_day_drop_threshold)
                & df["close"].ge(df["long_trend"] * (1.0 + self.config.close_above_long_tolerance))
            )
            df["pullback_candidate"] = df["trend_ok"] & df["pullback_touch"] & df["pullback_structure_ok"]
            recent_candidate = pd.Series(False, index=df.index)
            for i in range(1, self.config.recent_candidate_days + 1):
                recent_candidate = recent_candidate | df["pullback_candidate"].shift(i).fillna(False)
            df["recent_candidate"] = recent_candidate
            df["confirmation_bar"] = (
                df["trend_ok"]
                & df["recent_candidate"]
                & df["close"].gt(df["open"])
                & df["close"].ge(df["short_trend"] * (1.0 + self.config.confirm_close_to_trend_tolerance))
                & df["close"].gt(df["prev_close"])
                & df["volume_ratio_prev"].le(self.config.confirm_volume_ratio_max)
            )
            prepared[code] = df
        return prepared

    def generate_signals(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[pd.Timestamp, List[Signal]]:
        signal_map: Dict[pd.Timestamp, List[Signal]] = {}
        for code, df in stock_data.items():
            for dt in df.index[df["confirmation_bar"].fillna(False)]:
                row = df.loc[dt]
                score = float(row["return_5d"] - abs(row["distance_to_trend"]))
                if row.get("board") == "GEM":
                    score += self.config.gem_score_bonus
                signal = Signal(
                    date=dt,
                    code=code,
                    score=score,
                    reason="trend_pullback_confirmation",
                    metadata={},
                )
                signal_map.setdefault(dt, []).append(signal)
        return signal_map

    def should_exit(
        self,
        current_date: pd.Timestamp,
        position,
        row: pd.Series,
        stock_df: pd.DataFrame,
    ) -> str | None:
        hold_days = stock_df.index.get_loc(current_date) - stock_df.index.get_loc(position.entry_date)
        pin_long = row.get("pin_long")
        if pd.notna(pin_long) and float(pin_long) >= self.config.take_profit_pin_long:
            return "pin_long_100"
        if pd.notna(pin_long) and float(pin_long) < self.config.stop_pin_long:
            return "pin_long_below_80"
        if row["close"] < row["short_trend"]:
            return "close_below_trend"
        if hold_days >= self.config.max_stall_days and row["close"] <= position.entry_price:
            return "time_fail"
        return None


def maybe_exclude_disaster(
    stock_data: Dict[str, pd.DataFrame],
    exclude_disaster: bool,
) -> tuple[Dict[str, pd.DataFrame], List[pd.Timestamp]]:
    if not exclude_disaster:
        all_dates = sorted({dt for df in stock_data.values() for dt in df.index})
        return stock_data, all_dates

    filtered: Dict[str, pd.DataFrame] = {}
    all_dates = set()
    for code, df in stock_data.items():
        kept = df.loc[(df.index < DISASTER_START) | (df.index > DISASTER_END)].copy()
        if len(kept) < 30:
            continue
        filtered[code] = kept
        all_dates.update(kept.index.tolist())
    return filtered, sorted(all_dates)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal backtest for trend pullback confirmation strategy")
    parser.add_argument("data_dir", nargs="?", default="/Users/lidongyang/Desktop/Qstrategy/data/20260311/normal")
    parser.add_argument("--exclude-disaster", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="/Users/lidongyang/Desktop/Qstrategy/results/trend_pullback_confirmation_backtest_20260311",
    )
    args = parser.parse_args()

    config = PullbackConfig()
    stock_data, _ = load_price_directory(args.data_dir)
    stock_data, all_dates = maybe_exclude_disaster(stock_data, args.exclude_disaster)

    engine = BacktestEngine(
        EngineConfig(initial_capital=config.initial_capital, max_positions=config.max_positions)
    )
    strategy = TrendPullbackConfirmationStrategy(config)
    result = engine.run(strategy, stock_data, all_dates)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_trades(result.trades, output_dir / "trades.csv")
    result.equity_curve.rename("equity").to_csv(output_dir / "equity.csv", encoding="utf-8")

    win_rate = float(sum(1 for trade in result.trades if trade.return_pct > 0) / len(result.trades)) if result.trades else 0.0
    avg_trade_return = float(np.mean([trade.return_pct for trade in result.trades])) if result.trades else 0.0
    payload = {
        "strategy": config.name,
        "exclude_disaster": args.exclude_disaster,
        "disaster_period": [DISASTER_START.strftime("%Y-%m-%d"), DISASTER_END.strftime("%Y-%m-%d")],
        "config": {
            "strong_5d_threshold": config.strong_5d_threshold,
            "signal_day_drop_threshold": config.signal_day_drop_threshold,
            "touch_tolerance": config.touch_tolerance,
            "recent_candidate_days": config.recent_candidate_days,
            "confirm_close_to_trend_tolerance": config.confirm_close_to_trend_tolerance,
            "confirm_volume_ratio_max": config.confirm_volume_ratio_max,
            "max_stall_days": config.max_stall_days,
            "take_profit_pin_long": config.take_profit_pin_long,
            "stop_pin_long": config.stop_pin_long,
        },
        "signal_count": len(result.signals),
        "trade_count": len(result.trades),
        "win_rate": win_rate,
        "avg_trade_return": avg_trade_return,
        "metrics": result.metrics,
        "diagnostics": result.diagnostics,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
