from __future__ import annotations

import pandas as pd

from utils.backtest_pipeline.confirmers.base import BaseConfirmer, ConfirmerContext


class B3FollowThroughConfirmer(BaseConfirmer):
    """B3：小涨、小振幅、缩量承接等确认项。"""

    strategy_family = "b3"
    name = "b3.follow_through_quality"

    def apply(self, context: ConfirmerContext) -> pd.DataFrame:
        df = context.candidate_df.copy()
        if "confirmer_score" not in df.columns:
            df["confirmer_score"] = 0.0
        return df
