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


class B2StartupQualityConfirmer(BaseConfirmer):
    strategy_family = "b2"
    name = "b2.startup_quality"

    def apply(self, context: ConfirmerContext) -> pd.DataFrame:
        df = context.candidate_df.copy()
        close_pos = _col(df, "close_position", 0.0).clip(0.0, 1.0)
        vol_quality = (1.0 - np.minimum(np.abs(_col(df, "signal_vs_ma5", 0.0) - 1.8) / 0.8, 1.0)).clip(0.0, 1.0)
        trend_lead = _col(df, "trend_line_lead", 0.0).clip(lower=0.0)
        j_quality = (1.0 - np.minimum(_col(df, "j_value", 200.0) / 80.0, 1.0)).clip(0.0, 1.0)
        type_bonus = (
            _bool_col(df, "type1").astype(float) * 0.06
            + _bool_col(df, "type4").astype(float) * 0.10
        )
        df["confirmer_score"] = (
            0.35 * close_pos
            + 0.20 * vol_quality
            + 0.20 * trend_lead
            + 0.15 * j_quality
            + type_bonus
        )
        return df
