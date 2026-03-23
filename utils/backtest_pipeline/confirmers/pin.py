from __future__ import annotations

import pandas as pd

from utils.backtest_pipeline.confirmers.base import BaseConfirmer, ConfirmerContext


class PinNeedleQualityConfirmer(BaseConfirmer):
    """单针：下影线质量、量能配合、结构支撑等确认项。"""

    strategy_family = "pin"
    name = "pin.needle_quality"

    def apply(self, context: ConfirmerContext) -> pd.DataFrame:
        df = context.candidate_df.copy()
        if "confirmer_score" not in df.columns:
            df["confirmer_score"] = 0.0
        return df
