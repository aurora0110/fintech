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


class B3FollowThroughConfirmer(BaseConfirmer):
    """B3：小涨、小振幅、缩量承接等确认项。"""

    strategy_family = "b3"
    name = "b3.follow_through_quality"

    def apply(self, context: ConfirmerContext) -> pd.DataFrame:
        df = context.candidate_df.copy()
        ret_quality = (1.0 - np.minimum(np.abs(_col(df, "ret1", 1.0)) / 0.02, 1.0)).clip(0.0, 1.0)
        amp_quality = (1.0 - np.minimum(_col(df, "amplitude", 1.0) / 0.05, 1.0)).clip(0.0, 1.0)
        shrink_quality = (1.0 - np.minimum(np.maximum(_col(df, "vol_vs_prev", 10.0) - 0.70, 0.0) / 0.30, 1.0)).clip(0.0, 1.0)
        prev_b2_bonus = _bool_col(df, "prev_b2_any").astype(float)
        weekly_bonus = pd.Series(0.0, index=df.index, dtype=float)
        if "weekly_reason" in df.columns:
            weekly_bonus += df["weekly_reason"].fillna("").str.contains("周线强势", regex=False).astype(float) * 0.10
            weekly_bonus += df["weekly_reason"].fillna("").str.contains("周线进碗", regex=False).astype(float) * 0.05
        df["confirmer_score"] = (
            0.30 * _col(df, "b3_score", 0.0)
            + 0.20 * ret_quality
            + 0.20 * amp_quality
            + 0.20 * shrink_quality
            + 0.05 * prev_b2_bonus
            + weekly_bonus
        )
        return df
