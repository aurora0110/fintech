from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class DataConfig:
    data_dir: str
    exclude_ranges: List[tuple[str, str]] = field(
        default_factory=lambda: [("2015-06-01", "2024-09-30")]
    )
    min_history_bars: int = 30


@dataclass
class StrategyConfig:
    strategy_family: str
    candidate_pool: str
    confirmer: Optional[str] = None
    confirmer_mode: str = "score"
    confirmer_weight: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RankerConfig:
    name: str
    mode: str = "score"
    top_n: int = 3
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExitConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccountPolicyConfig:
    initial_capital: float = 1_000_000.0
    max_positions: int = 10
    position_sizing: str = "equal"
    max_holding_days: int = 60
    allow_same_day_overlap: bool = False
    cooldown_days: int = 0
    pause_after_stop_loss_count: int = 3
    pause_days: int = 5
    default_stop_loss_pct: float = 0.10
    halve_stop_loss_below_long_line: bool = True


@dataclass
class PipelineConfig:
    name: str
    data: DataConfig
    strategy: StrategyConfig
    ranker: RankerConfig
    exit: ExitConfig
    account: AccountPolicyConfig
    output_dir: Optional[Path] = None


@dataclass
class CandidateRecord:
    code: str
    signal_date: pd.Timestamp
    entry_date: Optional[pd.Timestamp]
    strategy_family: str
    candidate_pool: str
    signal_type: str
    base_score: float = 0.0
    confirmer_score: float = 0.0
    rank_score: float = 0.0
    note: str = ""
    features: Dict[str, Any] = field(default_factory=dict)
