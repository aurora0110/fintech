from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


ExitReason = Literal[
    "TAKE_PROFIT_GAP_OPEN",
    "TAKE_PROFIT_INTRADAY",
    "STOP_LOSS_GAP_OPEN",
    "STOP_LOSS_INTRADAY",
    "TRAIL_GAP_OPEN",
    "TRAIL_INTRADAY",
    "CLOSE_TRAIL_NEXT_OPEN",
    "FIXED_CLOSE",
    "NEXT_VALID_OPEN",
    "NO_EXIT",
]


@dataclass(frozen=True)
class ExitFill:
    is_settled: bool
    exit_idx: int | None
    exit_date: pd.Timestamp | pd.NaT
    exit_price: float
    exit_reason: ExitReason | str
    trigger_idx: int | None = None
    trigger_date: pd.Timestamp | pd.NaT = pd.NaT
    trigger_price: float = np.nan
    delayed_exit_days: int = 0
    gap_adjusted: bool = False


def valid_price(value: object) -> bool:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(price) and price > 0)


def valid_volume(value: object) -> bool:
    try:
        volume = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(volume) and volume > 0)


def row_date(df: pd.DataFrame, idx: int) -> pd.Timestamp:
    return pd.Timestamp(df.at[idx, "date"]) if "date" in df.columns else pd.NaT


def next_valid_open(
    df: pd.DataFrame,
    start_idx: int,
    *,
    max_idx: int | None = None,
    open_col: str = "open",
    volume_col: str = "volume",
) -> ExitFill:
    last_idx = len(df) - 1 if max_idx is None else min(int(max_idx), len(df) - 1)
    for idx in range(max(int(start_idx), 0), last_idx + 1):
        open_price = df.at[idx, open_col]
        volume_ok = volume_col not in df.columns or valid_volume(df.at[idx, volume_col])
        if valid_price(open_price) and volume_ok:
            return ExitFill(
                is_settled=True,
                exit_idx=idx,
                exit_date=row_date(df, idx),
                exit_price=float(open_price),
                exit_reason="NEXT_VALID_OPEN",
                delayed_exit_days=max(0, idx - int(start_idx)),
            )
    return ExitFill(False, None, pd.NaT, np.nan, "NO_EXIT", delayed_exit_days=0)


def fixed_close_exit(df: pd.DataFrame, signal_idx: int, hold_days: int, *, close_col: str = "close") -> ExitFill:
    exit_idx = int(signal_idx) + int(hold_days)
    if exit_idx >= len(df) or not valid_price(df.at[exit_idx, close_col]):
        return ExitFill(False, None, pd.NaT, np.nan, "NO_EXIT")
    return ExitFill(
        is_settled=True,
        exit_idx=exit_idx,
        exit_date=row_date(df, exit_idx),
        exit_price=float(df.at[exit_idx, close_col]),
        exit_reason="FIXED_CLOSE",
        trigger_idx=exit_idx,
        trigger_date=row_date(df, exit_idx),
        trigger_price=float(df.at[exit_idx, close_col]),
    )


def take_profit_gap_corrected(
    row: pd.Series,
    take_profit_price: float,
    *,
    open_col: str = "open",
    high_col: str = "high",
) -> ExitFill | None:
    open_price = float(row.get(open_col, np.nan))
    high_price = float(row.get(high_col, np.nan))
    idx = int(row.name) if isinstance(row.name, (int, np.integer)) else None
    date = pd.Timestamp(row.get("date", pd.NaT))
    if valid_price(open_price) and open_price >= take_profit_price:
        return ExitFill(True, idx, date, open_price, "TAKE_PROFIT_GAP_OPEN", idx, date, take_profit_price, gap_adjusted=True)
    if np.isfinite(high_price) and high_price >= take_profit_price:
        return ExitFill(True, idx, date, float(take_profit_price), "TAKE_PROFIT_INTRADAY", idx, date, take_profit_price)
    return None


def stop_loss_gap_corrected(
    row: pd.Series,
    stop_loss_price: float,
    *,
    open_col: str = "open",
    low_col: str = "low",
) -> ExitFill | None:
    open_price = float(row.get(open_col, np.nan))
    low_price = float(row.get(low_col, np.nan))
    idx = int(row.name) if isinstance(row.name, (int, np.integer)) else None
    date = pd.Timestamp(row.get("date", pd.NaT))
    if valid_price(open_price) and open_price <= stop_loss_price:
        return ExitFill(True, idx, date, open_price, "STOP_LOSS_GAP_OPEN", idx, date, stop_loss_price, gap_adjusted=True)
    if np.isfinite(low_price) and low_price <= stop_loss_price:
        return ExitFill(True, idx, date, float(stop_loss_price), "STOP_LOSS_INTRADAY", idx, date, stop_loss_price)
    return None


def corrected_intraday_trail_exit(
    df: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    max_exit_idx: int,
    *,
    trail_start: float,
    trail_drawdown: float,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
) -> ExitFill:
    holding_high = float(entry_price)
    active = False
    activated_idx: int | None = None
    last_idx = min(int(max_exit_idx), len(df) - 1)
    for idx in range(int(entry_idx), last_idx + 1):
        row = df.loc[idx]
        open_price = float(row.get(open_col, np.nan))
        high_price = float(row.get(high_col, np.nan))
        low_price = float(row.get(low_col, np.nan))
        if active:
            trail_price = holding_high * (1.0 - float(trail_drawdown))
            if valid_price(open_price) and open_price <= trail_price:
                return ExitFill(True, idx, row_date(df, idx), open_price, "TRAIL_GAP_OPEN", idx, row_date(df, idx), trail_price, gap_adjusted=True)
            if np.isfinite(low_price) and low_price <= trail_price:
                return ExitFill(True, idx, row_date(df, idx), trail_price, "TRAIL_INTRADAY", idx, row_date(df, idx), trail_price)
        if np.isfinite(high_price):
            holding_high = max(holding_high, high_price)
        if (not active) and holding_high >= float(entry_price) * (1.0 + float(trail_start)):
            active = True
            activated_idx = idx
    return ExitFill(False, None, pd.NaT, np.nan, "NO_EXIT", trigger_idx=activated_idx)


def close_trail_next_open_exit(
    df: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    max_exit_idx: int,
    *,
    trail_start: float,
    trail_drawdown: float,
    close_col: str = "close",
) -> ExitFill:
    highest_close = float(entry_price)
    active = False
    activated_idx: int | None = None
    last_idx = min(int(max_exit_idx), len(df) - 1)
    for idx in range(int(entry_idx), last_idx + 1):
        close_price = float(df.at[idx, close_col])
        if active and np.isfinite(close_price):
            trail_price = highest_close * (1.0 - float(trail_drawdown))
            if close_price <= trail_price:
                out = next_valid_open(df, idx + 1, max_idx=last_idx)
                if out.is_settled:
                    return ExitFill(
                        True,
                        out.exit_idx,
                        out.exit_date,
                        out.exit_price,
                        "CLOSE_TRAIL_NEXT_OPEN",
                        idx,
                        row_date(df, idx),
                        trail_price,
                        delayed_exit_days=out.delayed_exit_days,
                    )
                return out
        if np.isfinite(close_price):
            highest_close = max(highest_close, close_price)
        if (not active) and highest_close >= float(entry_price) * (1.0 + float(trail_start)):
            active = True
            activated_idx = idx
    return ExitFill(False, None, pd.NaT, np.nan, "NO_EXIT", trigger_idx=activated_idx)
