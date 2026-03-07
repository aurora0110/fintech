from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


@dataclass(frozen=True)
class FixedTakeProfitExit:
    take_profit_pct: float
    max_holding_days: int = 30
    name: str = "fixed_take_profit"


@dataclass(frozen=True)
class FixedDaysExit:
    holding_days: int
    name: str = "fixed_holding_days"


@dataclass(frozen=True)
class TieredExit:
    max_holding_days: int = 60
    name: str = "tiered_take_profit"


def exit_config_label(config: object) -> str:
    if isinstance(config, FixedTakeProfitExit):
        return f"固定涨幅止盈_{int(config.take_profit_pct * 100)}pct"
    if isinstance(config, FixedDaysExit):
        return f"固定持有_{config.holding_days}天"
    if isinstance(config, TieredExit):
        return "分批顺序止盈"
    return "unknown"


def _build_stock_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "date": df["date"].to_numpy(),
        "open": df["open"].to_numpy(dtype=float),
        "high": df["high"].to_numpy(dtype=float),
        "low": df["low"].to_numpy(dtype=float),
        "close": df["close"].to_numpy(dtype=float),
        "trend_line": df["trend_line"].to_numpy(dtype=float),
        "J": df["J"].to_numpy(dtype=float),
    }


def _fixed_take_profit_trade(arrays: dict[str, np.ndarray], signal_idx: int, config: FixedTakeProfitExit) -> dict | None:
    if signal_idx + 1 >= len(arrays["open"]):
        return None
    entry_idx = signal_idx + 1
    entry_price = float(arrays["open"][entry_idx])
    entry_low = float(arrays["low"][entry_idx])
    if pd.isna(entry_price) or pd.isna(entry_low) or entry_price <= 0:
        return None

    stop_price = entry_low * 0.9
    take_profit_price = entry_price * (1.0 + config.take_profit_pct)
    end_idx = min(len(arrays["open"]) - 1, entry_idx + config.max_holding_days - 1)
    exit_price = float(arrays["close"][end_idx])
    exit_date = arrays["date"][end_idx]
    exit_reason = f"hold_{config.max_holding_days}d"

    if end_idx > entry_idx:
        lows = arrays["low"][entry_idx + 1 : end_idx + 1]
        highs = arrays["high"][entry_idx + 1 : end_idx + 1]
        stop_hits = np.flatnonzero(lows <= stop_price)
        tp_hits = np.flatnonzero(highs >= take_profit_price)
        first_stop = int(stop_hits[0]) if stop_hits.size > 0 else None
        first_tp = int(tp_hits[0]) if tp_hits.size > 0 else None
        if first_stop is not None and (first_tp is None or first_stop <= first_tp):
            hit_idx = entry_idx + 1 + first_stop
            exit_price = stop_price
            exit_date = arrays["date"][hit_idx]
            exit_reason = "stop_loss"
        elif first_tp is not None:
            hit_idx = entry_idx + 1 + first_tp
            exit_price = take_profit_price
            exit_date = arrays["date"][hit_idx]
            exit_reason = "take_profit"

    return {
        "entry_date": arrays["date"][entry_idx],
        "exit_date": exit_date,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "stop_price": stop_price,
        "take_profit_price": take_profit_price,
        "return_pct": exit_price / entry_price - 1.0,
        "exit_reason": exit_reason,
    }


def _fixed_days_trade(arrays: dict[str, np.ndarray], signal_idx: int, config: FixedDaysExit) -> dict | None:
    if signal_idx + 1 >= len(arrays["open"]):
        return None
    entry_idx = signal_idx + 1
    entry_price = float(arrays["open"][entry_idx])
    entry_low = float(arrays["low"][entry_idx])
    if pd.isna(entry_price) or pd.isna(entry_low) or entry_price <= 0:
        return None

    stop_price = entry_low * 0.9
    end_idx = min(len(arrays["open"]) - 1, entry_idx + config.holding_days - 1)
    exit_price = float(arrays["close"][end_idx])
    exit_date = arrays["date"][end_idx]
    exit_reason = f"hold_{config.holding_days}d"

    if end_idx > entry_idx:
        lows = arrays["low"][entry_idx + 1 : end_idx + 1]
        stop_hits = np.flatnonzero(lows <= stop_price)
        if stop_hits.size > 0:
            hit_idx = entry_idx + 1 + int(stop_hits[0])
            exit_price = stop_price
            exit_date = arrays["date"][hit_idx]
            exit_reason = "stop_loss"

    return {
        "entry_date": arrays["date"][entry_idx],
        "exit_date": exit_date,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "stop_price": stop_price,
        "take_profit_price": np.nan,
        "return_pct": exit_price / entry_price - 1.0,
        "exit_reason": exit_reason,
    }


def _tiered_trade(arrays: dict[str, np.ndarray], signal_idx: int, config: TieredExit) -> dict | None:
    if signal_idx + 1 >= len(arrays["open"]):
        return None
    entry_idx = signal_idx + 1
    entry_price = float(arrays["open"][entry_idx])
    entry_low = float(arrays["low"][entry_idx])
    if pd.isna(entry_price) or pd.isna(entry_low) or entry_price <= 0:
        return None

    stop_price = entry_low * 0.9
    remaining = 1.0
    realized_value = 0.0
    executed_steps: List[str] = []
    dd_enabled = False
    consecutive_dd = 0
    final_reason = f"hold_{config.max_holding_days}d"

    step1_done = False
    step2_done = False
    step3_done = False
    step4_done = False
    step5_done = False

    end_idx = min(len(arrays["open"]) - 1, entry_idx + config.max_holding_days - 1)
    exit_date = arrays["date"][end_idx]
    take_profit_price = entry_price * 1.08

    for idx in range(entry_idx + 1, end_idx + 1):
        low = float(arrays["low"][idx])
        high = float(arrays["high"][idx])
        close = float(arrays["close"][idx])
        prev_low = float(arrays["low"][idx - 1])
        trend_line = float(arrays["trend_line"][idx])
        j_value = float(arrays["J"][idx])

        if low <= stop_price and remaining > 0:
            realized_value += remaining * stop_price
            remaining = 0.0
            exit_date = arrays["date"][idx]
            final_reason = "stop_loss"
            break

        def sell_chunk(price: float, reason: str, sell_all: bool = False) -> None:
            nonlocal realized_value, remaining, stop_price, dd_enabled
            chunk = remaining if sell_all else min(remaining, 0.2)
            if chunk <= 0:
                return
            realized_value += chunk * price
            remaining -= chunk
            stop_price = max(stop_price, low * 0.95)
            if reason == "step2_gain_8pct":
                dd_enabled = True
            executed_steps.append(reason)

        if not step1_done and j_value > 100 and remaining > 0:
            sell_chunk(close, "step1_j_gt_100")
            step1_done = True

        if not step2_done and high >= take_profit_price and remaining > 0:
            sell_chunk(take_profit_price, "step2_gain_8pct")
            step2_done = True

        if step1_done and step2_done and not step3_done and high > trend_line * 1.15 and remaining > 0:
            sell_chunk(close, "step3_high_above_trend_15pct")
            step3_done = True
        elif step1_done and step2_done and step3_done and not step4_done and high > trend_line * 1.20 and remaining > 0:
            sell_chunk(close, "step4_high_above_trend_20pct")
            step4_done = True
        elif step1_done and step2_done and step4_done and not step5_done and high > trend_line * 1.25 and remaining > 0:
            sell_chunk(close, "step5_high_above_trend_25pct", sell_all=True)
            step5_done = True

        if dd_enabled and remaining > 0:
            if close < prev_low:
                consecutive_dd += 1
            else:
                consecutive_dd = 0
            if consecutive_dd >= 3:
                realized_value += remaining * close
                remaining = 0.0
                exit_date = arrays["date"][idx]
                final_reason = "dd_stop_loss"
                break

        if remaining <= 0:
            exit_date = arrays["date"][idx]
            final_reason = executed_steps[-1] if executed_steps else "tiered_take_profit"
            break

    if remaining > 0:
        final_close = float(arrays["close"][end_idx])
        realized_value += remaining * final_close
        exit_date = arrays["date"][end_idx]
        final_reason = f"hold_{config.max_holding_days}d"

    return_pct = realized_value / entry_price - 1.0
    return {
        "entry_date": arrays["date"][entry_idx],
        "exit_date": exit_date,
        "entry_price": entry_price,
        "exit_price": realized_value,
        "stop_price": stop_price,
        "take_profit_price": take_profit_price,
        "return_pct": return_pct,
        "exit_reason": final_reason,
        "remaining_position": remaining,
        "tiered_steps": "|".join(executed_steps),
    }


def simulate_signal_exits(
    prepared_stock_data: Dict[str, pd.DataFrame],
    signal_candidates: pd.DataFrame,
    exit_config: object,
    success_return_threshold: float,
) -> pd.DataFrame:
    records: List[dict] = []
    label = exit_config_label(exit_config)
    grouped = signal_candidates.groupby("code", sort=False)
    total_codes = signal_candidates["code"].nunique()
    for code, code_rows in tqdm(grouped, total=total_codes, desc=f"Simulating {label}", unit="stock"):
        arrays = _build_stock_arrays(prepared_stock_data[code])
        for row in code_rows.itertuples(index=False):
            signal_idx = int(row.signal_idx)
            if isinstance(exit_config, FixedTakeProfitExit):
                trade = _fixed_take_profit_trade(arrays, signal_idx, exit_config)
            elif isinstance(exit_config, FixedDaysExit):
                trade = _fixed_days_trade(arrays, signal_idx, exit_config)
            elif isinstance(exit_config, TieredExit):
                trade = _tiered_trade(arrays, signal_idx, exit_config)
            else:
                raise ValueError(f"Unsupported exit config: {exit_config}")

            if trade is None:
                continue

            record = row._asdict()
            record.update(trade)
            record["success"] = float(record["return_pct"] > success_return_threshold)
            record["exit_model"] = label
            records.append(record)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)
