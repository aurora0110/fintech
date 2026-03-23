from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class ConfirmerContext:
    candidate_df: pd.DataFrame
    stock_data: Dict[str, pd.DataFrame]
    params: Dict[str, object]


class BaseConfirmer:
    strategy_family: str = "base"
    name: str = "base"

    def apply(self, context: ConfirmerContext) -> pd.DataFrame:
        raise NotImplementedError
