from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from core.market_rules import is_limit_down, is_limit_up
from core.metrics import compute_metrics
from core.models import BacktestResult, Position, Signal, Trade


@dataclass
class EngineConfig:
    initial_capital: float = 1_000_000.0
    max_positions: int = 10
    commission_rate: float = 0.0003
    slippage_rate: float = 0.001
    stamp_duty_rate: float = 0.001
    min_lot: int = 100


class BaseSignalStrategy:
    name: str = "base"

    def prepare(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError

    def generate_signals(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[pd.Timestamp, List[Signal]]:
        raise NotImplementedError

    def build_position(self, signal: Signal, exec_row: pd.Series) -> Dict[str, float]:
        return {}

    def should_exit(
        self,
        current_date: pd.Timestamp,
        position: Position,
        row: pd.Series,
        stock_df: pd.DataFrame,
    ) -> Optional[str]:
        hold_days = stock_df.index.get_loc(current_date) - stock_df.index.get_loc(position.entry_date)
        if position.stop_price is not None and row["close"] <= position.stop_price:
            return "stop_loss"
        if position.take_profit_price is not None and row["close"] >= position.take_profit_price:
            return "take_profit"
        max_hold = position.metadata.get("max_holding_days")
        if max_hold is not None and hold_days >= max_hold:
            return f"hold_{int(max_hold)}d"
        return None


class BacktestEngine:
    def __init__(self, config: EngineConfig):
        self.config = config

    def run(self, strategy: BaseSignalStrategy, stock_data: Dict[str, pd.DataFrame], all_dates: List[pd.Timestamp]) -> BacktestResult:
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
        positions: Dict[str, Position] = {}
        trades: List[Trade] = []
        equity_points: List[float] = []
        equity_index: List[pd.Timestamp] = []
        pending_exit: Dict[pd.Timestamp, Dict[str, str]] = {}
        rejected_buys = 0
        rejected_sells = 0
        filled_buys = 0
        filled_sells = 0
        blocked_suspended_buys = 0
        blocked_suspended_sells = 0

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
                            next_date = all_dates[idx + 1]
                            pending_exit.setdefault(next_date, {})[code] = scheduled_reason
                        next_positions[code] = position
                    else:
                        sell_price = float(row["open"]) * (1.0 - self.config.slippage_rate)
                        gross_cash = position.shares * sell_price
                        fee = gross_cash * self.config.commission_rate
                        tax = gross_cash * self.config.stamp_duty_rate
                        cash += gross_cash - fee - tax
                        pnl = (sell_price - position.entry_price) * position.shares - fee - tax
                        trades.append(
                            Trade(
                                code=code,
                                entry_date=position.entry_date,
                                exit_date=current_date,
                                entry_price=position.entry_price,
                                exit_price=sell_price,
                                shares=position.shares,
                                pnl=pnl,
                                return_pct=(sell_price - position.entry_price) / position.entry_price,
                                reason=scheduled_reason,
                            )
                        )
                        filled_sells += 1
                    continue

                exit_reason = strategy.should_exit(current_date, position, row, df)
                position.metadata["last_close"] = float(row["close"])
                if exit_reason is not None:
                    idx = date_to_idx.get(current_date)
                    if idx is not None and idx + 1 < len(all_dates):
                        next_date = all_dates[idx + 1]
                        pending_exit.setdefault(next_date, {})[code] = exit_reason
                next_positions[code] = position

            positions = next_positions

            candidates = sorted(signal_map.get(current_date, []), key=lambda item: item.score, reverse=True)
            available_slots = max(self.config.max_positions - len(positions), 0)
            existing = set(positions)

            for signal in candidates:
                if available_slots <= 0:
                    break
                if signal.code in existing:
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

                entry_price = float(row["open"]) * (1.0 + self.config.slippage_rate)
                allocation = cash / available_slots if available_slots > 0 else 0.0
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
                if current_date in df.index:
                    mark_price = float(df.loc[current_date, "close"])
                else:
                    mark_price = float(position.metadata.get("last_close", position.entry_price))
                equity += position.shares * mark_price

            equity_index.append(current_date)
            equity_points.append(equity)

        equity_curve = pd.Series(equity_points, index=pd.DatetimeIndex(equity_index), dtype=float)
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
