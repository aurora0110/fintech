from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class Signal:
    date: pd.Timestamp
    code: str
    score: float
    reason: str
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class Position:
    code: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    stop_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    signal: Optional[Signal] = None
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class Trade:
    code: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    return_pct: float
    reason: str


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    daily_returns: pd.Series
    trades: List[Trade]
    signals: List[Signal]
    metrics: Dict[str, float]
    diagnostics: Dict[str, float]
