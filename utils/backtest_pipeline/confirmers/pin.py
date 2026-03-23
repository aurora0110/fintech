from __future__ import annotations

import numpy as np
import pandas as pd

from utils.backtest_pipeline.confirmers.base import BaseConfirmer, ConfirmerContext


def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _bool_col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name].fillna(False).astype(bool)
    return pd.Series(False, index=df.index, dtype=bool)


class PinNeedleQualityConfirmer(BaseConfirmer):
    """单针：下影线质量、量能配合、结构支撑等确认项。"""

    strategy_family = "pin"
    name = "pin.needle_quality"

    def apply(self, context: ConfirmerContext) -> pd.DataFrame:
        df = context.candidate_df.copy()
        subtype_text = df.get("matched_subtypes", pd.Series("", index=df.index)).fillna("")
        a_bonus = subtype_text.str.contains("A型", regex=False).astype(float)
        b_bonus = subtype_text.str.contains("B型", regex=False).astype(float)
        c_bonus = subtype_text.str.contains("C型", regex=False).astype(float)
        trend_lead = _col(df, "trend_line_lead", 0.0).clip(lower=0.0)
        vol_shrink = (1.0 - np.minimum(_col(df, "signal_vs_ma20", 10.0), 2.0) / 2.0).clip(0.0, 1.0)
        ret_pullback = (1.0 - np.minimum(np.maximum(_col(df, "ret3", 0.0), -0.10).clip(lower=-0.10) + 0.10, 0.10) / 0.10).clip(0.0, 1.0)
        structure_bonus = (
            _bool_col(df, "along_trend_up").astype(float) * 0.10
            + _bool_col(df, "n_up_any").astype(float) * 0.12
            + _bool_col(df, "keyk_support_active").astype(float) * 0.18
        )
        df["confirmer_score"] = (
            0.20 * a_bonus
            + 0.10 * b_bonus
            + 0.20 * c_bonus
            + 0.20 * trend_lead
            + 0.15 * vol_shrink
            + 0.05 * ret_pullback
            + structure_bonus
        )
        return df
