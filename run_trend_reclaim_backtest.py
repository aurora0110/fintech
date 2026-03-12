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
from strategies.common import calc_kdj, calc_trend_cn


DISASTER_START = pd.Timestamp("2015-06-12")
DISASTER_END = pd.Timestamp("2024-09-18")


@dataclass
class ReclaimConfig:
    name: str = "trend_reclaim"
    initial_capital: float = 1_000_000.0
    max_positions: int = 10
    strong_5d_threshold: float = 0.03
    pullback_day_drop_threshold: float = -0.015
    pullback_close_floor_vs_long: float = 0.985
    touch_tolerance: float = 0.02
    candidate_window_days: int = 3
    confirm_volume_ratio_max: float = 1.8
    stop_loss_buffer: float = 0.98
    partial_take_profit_days: int = 3
    partial_take_profit_pct: float = 0.02
    failure_exit_days: int = 3


class TrendReclaimStrategy(BaseSignalStrategy):
    def __init__(self, config: ReclaimConfig):
        self.config = config
        self.name = config.name

    def prepare(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        prepared: Dict[str, pd.DataFrame] = {}
        for code, raw_df in stock_data.items():
            df = raw_df.copy()
            df["prev_close"] = df["close"].shift(1)
            df["short_trend"], df["long_trend"] = calc_trend_cn(df["close"])
            df["K"], df["D"], df["J"] = calc_kdj(df["high"], df["low"], df["close"])
            df["return_5d"] = df["close"].pct_change(5)
            df["signal_day_return"] = df["close"] / df["prev_close"] - 1.0
            df["volume_ratio_prev"] = df["volume"] / df["volume"].shift(1).replace(0.0, np.nan)
            df["trend_slope_3"] = df["short_trend"] / df["short_trend"].shift(3) - 1.0
            df["long_slope_5"] = df["long_trend"] / df["long_trend"].shift(5) - 1.0

            df["trend_ok"] = (
                df["short_trend"].gt(df["long_trend"])
                & df["trend_slope_3"].gt(0.0)
                & df["long_slope_5"].gt(0.0)
                & df["return_5d"].gt(self.config.strong_5d_threshold)
            )
            df["touch_trend"] = (
                df["low"].le(df["short_trend"] * (1.0 + self.config.touch_tolerance))
                | df["low"].le(df["long_trend"] * (1.0 + self.config.touch_tolerance))
            )
            df["pullback_candidate"] = (
                df["trend_ok"]
                & df["touch_trend"]
                & df["signal_day_return"].le(self.config.pullback_day_drop_threshold)
                & df["close"].ge(df["long_trend"] * self.config.pullback_close_floor_vs_long)
            )

            recent_candidate = pd.Series(False, index=df.index, dtype=bool)
            candidate_low = pd.Series(np.nan, index=df.index, dtype=float)
            for i in range(1, self.config.candidate_window_days + 1):
                shifted = df["pullback_candidate"].shift(i, fill_value=False)
                recent_candidate = recent_candidate | shifted
                shifted_low = df["low"].shift(i)
                candidate_low = candidate_low.where(~shifted, shifted_low)
            df["recent_candidate"] = recent_candidate
            df["candidate_low"] = candidate_low.ffill().where(recent_candidate)
            df["j_turn_up"] = df["J"].gt(df["J"].shift(1)) & df["J"].shift(1).le(df["J"].shift(2))
            df["j_turn_strength"] = df["J"] - df["J"].shift(1)
            df["reclaim_trend"] = (
                df["close"].ge(df["short_trend"])
                & df["open"].le(df["short_trend"] * 1.01)
                & df["close"].gt(df["prev_close"])
                & df["close"].gt(df["open"])
            )
            df["confirmation_bar"] = (
                df["trend_ok"]
                & df["recent_candidate"]
                & df["reclaim_trend"]
                & df["j_turn_up"]
                & df["volume_ratio_prev"].le(self.config.confirm_volume_ratio_max)
            )
            prepared[code] = df
        return prepared

    def generate_signals(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[pd.Timestamp, List[Signal]]:
        signal_map: Dict[pd.Timestamp, List[Signal]] = {}
        for code, df in stock_data.items():
            dates = df.index[df["confirmation_bar"].fillna(False)]
            for dt in dates:
                row = df.loc[dt]
                score = float(row["return_5d"] + (float(row["j_turn_strength"]) if pd.notna(row["j_turn_strength"]) else 0.0))
                signal_map.setdefault(dt, []).append(
                    Signal(
                        date=dt,
                        code=code,
                        score=score,
                        reason="trend_reclaim",
                        metadata={
                            "signal_low": float(row["candidate_low"]) if pd.notna(row["candidate_low"]) else float(row["low"]),
                        },
                    )
                )
        return signal_map

    def build_position(self, signal: Signal, exec_row: pd.Series) -> Dict[str, float]:
        signal_low = float(signal.metadata.get("signal_low", exec_row["low"]))
        return {"stop_price": signal_low * self.config.stop_loss_buffer}

    def should_exit(
        self,
        current_date: pd.Timestamp,
        position,
        row: pd.Series,
        stock_df: pd.DataFrame,
    ) -> str | None:
        hold_days = stock_df.index.get_loc(current_date) - stock_df.index.get_loc(position.entry_date)
        if position.stop_price is not None and row["close"] < position.stop_price:
            return "stop_loss"

        partial_taken = bool(position.metadata.get("partial_taken", False))
        if not partial_taken:
            gain = row["close"] / position.entry_price - 1.0
            if hold_days <= self.config.partial_take_profit_days and gain >= self.config.partial_take_profit_pct:
                position.metadata["partial_taken"] = True
                position.metadata["down_close_count"] = 0
                return "partial_take_profit"
            if hold_days >= self.config.failure_exit_days:
                return "failure_exit"
            return None

        prev_close = row.get("prev_close")
        if pd.notna(prev_close) and row["close"] < prev_close:
            position.metadata["down_close_count"] = int(position.metadata.get("down_close_count", 0)) + 1
        else:
            position.metadata["down_close_count"] = 0
        if int(position.metadata.get("down_close_count", 0)) >= 2:
            return "post_take_profit_down2"
        return None


class TrendReclaimEngine(BacktestEngine):
    def run(self, strategy: BaseSignalStrategy, stock_data: Dict[str, pd.DataFrame], all_dates: List[pd.Timestamp]):
        prepared = strategy.prepare(stock_data)
        raw_signal_map = strategy.generate_signals(prepared)
        signal_map: Dict[pd.Timestamp, List[Signal]] = {}
        flat_signals: List[Signal] = []
        date_to_idx = {dt: idx for idx, dt in enumerate(all_dates)}
        for signal_date, signals in raw_signal_map.items():
            idx = date_to_idx.get(signal_date)
            if idx is None or idx + 1 >= len(all_dates):
                continue
            exec_date = all_dates[idx + 1]
            signal_map.setdefault(exec_date, []).extend(signals)
            flat_signals.extend(signals)

        cash = float(self.config.initial_capital)
        positions: Dict[str, object] = {}
        trades: List[object] = []
        equity_points: List[float] = []
        equity_index: List[pd.Timestamp] = []
        pending_exit: Dict[pd.Timestamp, Dict[str, str]] = {}
        rejected_buys = 0
        rejected_sells = 0
        filled_buys = 0
        filled_sells = 0
        blocked_suspended_buys = 0
        blocked_suspended_sells = 0

        from core.market_rules import is_limit_down, is_limit_up
        from core.metrics import compute_metrics
        from core.models import BacktestResult, Position, Trade

        for current_date in all_dates:
            next_positions: Dict[str, Position] = {}
            for code, position in positions.items():
                df = prepared[code]
                if current_date not in df.index:
                    next_positions[code] = position
                    continue
                row = df.loc[current_date]
                if bool(row.get("is_suspended", False)):
                    blocked_suspended_sells += 1
                    next_positions[code] = position
                    continue
                scheduled_reason = pending_exit.get(current_date, {}).get(code)
                if scheduled_reason is not None:
                    prev_close = float(row.get("prev_close", row["close"]))
                    limit_pct = float(row.get("limit_pct", 0.10))
                    if is_limit_down(float(row["open"]), prev_close, limit_pct):
                        rejected_sells += 1
                        idx = date_to_idx.get(current_date)
                        if idx is not None and idx + 1 < len(all_dates):
                            pending_exit.setdefault(all_dates[idx + 1], {})[code] = scheduled_reason
                        next_positions[code] = position
                    else:
                        sell_price = float(row["open"]) * (1.0 - self.config.slippage_rate)
                        exit_ratio = 0.5 if scheduled_reason == "partial_take_profit" else 1.0
                        sell_shares = int(position.shares * exit_ratio / self.config.min_lot) * self.config.min_lot
                        if exit_ratio == 1.0 or sell_shares <= 0:
                            sell_shares = position.shares
                        gross_cash = sell_shares * sell_price
                        fee = gross_cash * self.config.commission_rate
                        tax = gross_cash * self.config.stamp_duty_rate
                        cash += gross_cash - fee - tax
                        pnl = (sell_price - position.entry_price) * sell_shares - fee - tax
                        trades.append(
                            Trade(
                                code=code,
                                entry_date=position.entry_date,
                                exit_date=current_date,
                                entry_price=position.entry_price,
                                exit_price=sell_price,
                                shares=sell_shares,
                                pnl=pnl,
                                return_pct=(sell_price - position.entry_price) / position.entry_price,
                                reason=scheduled_reason,
                            )
                        )
                        filled_sells += 1
                        remaining_shares = position.shares - sell_shares
                        if remaining_shares > 0:
                            position.shares = remaining_shares
                            position.metadata["last_close"] = float(row["close"])
                            next_positions[code] = position
                    continue

                exit_reason = strategy.should_exit(current_date, position, row, df)
                position.metadata["last_close"] = float(row["close"])
                if exit_reason is not None:
                    idx = date_to_idx.get(current_date)
                    if idx is not None and idx + 1 < len(all_dates):
                        pending_exit.setdefault(all_dates[idx + 1], {})[code] = exit_reason
                next_positions[code] = position

            positions = next_positions

            equity_before_buys = cash
            for code, position in positions.items():
                df = prepared[code]
                mark_price = float(df.loc[current_date, "close"]) if current_date in df.index else float(position.metadata.get("last_close", position.entry_price))
                equity_before_buys += position.shares * mark_price

            raw_candidates = strategy.order_candidates(current_date, signal_map.get(current_date, []))
            available_slots = max(self.config.max_positions - len(positions), 0)
            existing = set(positions)
            executable_candidates: List[tuple[Signal, pd.Series]] = []
            for signal in raw_candidates:
                if available_slots <= 0 or signal.code in existing:
                    continue
                df = prepared[signal.code]
                if current_date not in df.index:
                    continue
                row = df.loc[current_date]
                if bool(row.get("is_suspended", False)):
                    blocked_suspended_buys += 1
                    continue
                prev_close = float(row.get("prev_close", row["close"]))
                limit_pct = float(row.get("limit_pct", 0.10))
                if is_limit_up(float(row["open"]), prev_close, limit_pct):
                    rejected_buys += 1
                    continue
                executable_candidates.append((signal, row))
                existing.add(signal.code)
                available_slots -= 1

            existing = set(positions)
            available_slots = max(self.config.max_positions - len(positions), 0)
            candidate_count = len(executable_candidates)
            for signal, row in executable_candidates:
                if available_slots <= 0 or signal.code in existing:
                    continue
                entry_price = float(row["open"]) * (1.0 + self.config.slippage_rate)
                allocation = strategy.position_allocation(
                    current_date=current_date,
                    signal=signal,
                    exec_row=row,
                    cash=cash,
                    equity=equity_before_buys,
                    candidate_count=candidate_count,
                    available_slots=available_slots,
                    positions=positions,
                    trades=trades,
                    all_dates=all_dates,
                )
                if allocation is None or allocation <= 0:
                    rejected_buys += 1
                    continue
                shares = int(allocation / entry_price / self.config.min_lot) * self.config.min_lot
                if shares <= 0:
                    rejected_buys += 1
                    continue
                gross_cost = shares * entry_price
                fee = gross_cost * self.config.commission_rate
                total_cost = gross_cost + fee
                if total_cost > cash:
                    rejected_buys += 1
                    continue
                cash -= total_cost
                position_kwargs = strategy.build_position(signal, row)
                positions[signal.code] = Position(
                    code=signal.code,
                    entry_date=current_date,
                    entry_price=entry_price,
                    shares=shares,
                    stop_price=position_kwargs.get("stop_price"),
                    take_profit_price=position_kwargs.get("take_profit_price"),
                    signal=signal,
                    metadata={
                        "last_close": float(row["close"]),
                        **{k: v for k, v in position_kwargs.items() if k not in {"stop_price", "take_profit_price"}},
                    },
                )
                existing.add(signal.code)
                available_slots -= 1
                filled_buys += 1

            equity = cash
            for code, position in positions.items():
                df = prepared[code]
                mark_price = float(df.loc[current_date, "close"]) if current_date in df.index else float(position.metadata.get("last_close", position.entry_price))
                equity += position.shares * mark_price
            equity_points.append(equity)
            equity_index.append(current_date)

        equity_curve = pd.Series(equity_points, index=equity_index, name="equity")
        daily_returns = equity_curve.pct_change().fillna(0.0)
        metrics = compute_metrics(equity_curve)
        diagnostics = {
            "signal_count": float(len(flat_signals)),
            "filled_buys": float(filled_buys),
            "rejected_buys": float(rejected_buys),
            "filled_sells": float(filled_sells),
            "rejected_sells": float(rejected_sells),
            "blocked_suspended_buys": float(blocked_suspended_buys),
            "blocked_suspended_sells": float(blocked_suspended_sells),
            "trade_count": float(len(trades)),
        }
        return BacktestResult(
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            trades=trades,
            signals=flat_signals,
            metrics=metrics,
            diagnostics=diagnostics,
        )


def maybe_exclude_disaster(stock_data: Dict[str, pd.DataFrame], exclude_disaster: bool):
    if not exclude_disaster:
        return stock_data, sorted({dt for df in stock_data.values() for dt in df.index})
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
    parser = argparse.ArgumentParser(description="Formal backtest for trend reclaim strategy")
    parser.add_argument("data_dir", nargs="?", default="/Users/lidongyang/Desktop/Qstrategy/data/20260311/normal")
    parser.add_argument("--exclude-disaster", action="store_true")
    parser.add_argument("--output-dir", default="/Users/lidongyang/Desktop/Qstrategy/results/trend_reclaim_backtest_20260311")
    args = parser.parse_args()

    config = ReclaimConfig()
    stock_data, _ = load_price_directory(args.data_dir)
    stock_data, all_dates = maybe_exclude_disaster(stock_data, args.exclude_disaster)

    engine = TrendReclaimEngine(EngineConfig(initial_capital=config.initial_capital, max_positions=config.max_positions))
    strategy = TrendReclaimStrategy(config)
    result = engine.run(strategy, stock_data, all_dates)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_trades(result.trades, output_dir / "trades.csv")
    result.equity_curve.rename("equity").to_csv(output_dir / "equity.csv", encoding="utf-8")

    returns = [trade.return_pct for trade in result.trades]
    payload = {
        "strategy": config.name,
        "exclude_disaster": args.exclude_disaster,
        "disaster_period": [DISASTER_START.strftime("%Y-%m-%d"), DISASTER_END.strftime("%Y-%m-%d")],
        "config": {
            "strong_5d_threshold": config.strong_5d_threshold,
            "pullback_day_drop_threshold": config.pullback_day_drop_threshold,
            "pullback_close_floor_vs_long": config.pullback_close_floor_vs_long,
            "touch_tolerance": config.touch_tolerance,
            "candidate_window_days": config.candidate_window_days,
            "confirm_volume_ratio_max": config.confirm_volume_ratio_max,
            "stop_loss_buffer": config.stop_loss_buffer,
            "partial_take_profit_days": config.partial_take_profit_days,
            "partial_take_profit_pct": config.partial_take_profit_pct,
            "failure_exit_days": config.failure_exit_days,
        },
        "signal_count": len(result.signals),
        "trade_count": len(result.trades),
        "win_rate": float(sum(1 for value in returns if value > 0) / len(returns)) if returns else 0.0,
        "avg_trade_return": float(np.mean(returns)) if returns else 0.0,
        "metrics": result.metrics,
        "diagnostics": result.diagnostics,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
