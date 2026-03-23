from __future__ import annotations

import pandas as pd

from utils.backtest_pipeline.confirmers.base import BaseConfirmer, ConfirmerContext


class BrickTurnQualityConfirmer(BaseConfirmer):
    strategy_family = "brick"
    name = "brick.turn_quality"

    def apply(self, context: ConfirmerContext) -> pd.DataFrame:
        df = context.candidate_df.copy()
        if "confirmer_score" not in df.columns:
            df["confirmer_score"] = 0.0
        return df
