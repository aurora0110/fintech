from __future__ import annotations

from typing import Sequence

import pandas as pd


def normalize_trade_dates(trade_dates: Sequence[str]) -> list[pd.Timestamp]:
    values = pd.to_datetime(list(trade_dates), errors="coerce")
    return sorted({dt.normalize() for dt in values if not pd.isna(dt)})


def get_trade_date_offset(trade_dates: Sequence[str], current_date: str, offset: int):
    normalized = normalize_trade_dates(trade_dates)
    current = pd.to_datetime(current_date, errors="coerce")
    if pd.isna(current):
        return None
    current = current.normalize()
    if current not in normalized:
        return None
    idx = normalized.index(current) + offset
    if idx < 0 or idx >= len(normalized):
        return None
    return normalized[idx]


def get_next_trade_date(trade_dates: Sequence[str], current_date: str, n: int = 1):
    return get_trade_date_offset(trade_dates, current_date, n)


def get_prev_trade_date(trade_dates: Sequence[str], current_date: str, n: int = 1):
    return get_trade_date_offset(trade_dates, current_date, -n)

