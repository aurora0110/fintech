from __future__ import annotations

import pandas as pd

from utils.backtest_pipeline.confirmers.base import BaseConfirmer, ConfirmerContext


class B2StartupQualityConfirmer(BaseConfirmer):
    strategy_family = "b2"
    name = "b2.startup_quality"

    def apply(self, context: ConfirmerContext) -> pd.DataFrame:
        df = context.candidate_df.copy()
        if "confirmer_score" not in df.columns:
            df["confirmer_score"] = 0.0
        return df
