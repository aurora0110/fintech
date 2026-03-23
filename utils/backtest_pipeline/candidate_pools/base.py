from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from utils.backtest_pipeline.types import CandidateRecord, StrategyConfig


@dataclass
class CandidatePoolContext:
    data_dir: str
    stock_data: Dict[str, pd.DataFrame]
    all_dates: List[pd.Timestamp]
    strategy_config: StrategyConfig


class BaseCandidatePool:
    strategy_family: str = "base"
    name: str = "base"
    requires_market_data: bool = True

    def generate(self, context: CandidatePoolContext) -> pd.DataFrame:
        raise NotImplementedError
