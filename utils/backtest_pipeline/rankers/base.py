from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class RankerContext:
    candidate_df: pd.DataFrame
    params: Dict[str, object]


class BaseRanker:
    strategy_family: str = "generic"
    name: str = "base"

    def score(self, context: RankerContext) -> pd.DataFrame:
        raise NotImplementedError
