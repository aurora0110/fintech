from __future__ import annotations

import pandas as pd

from utils.backtest_pipeline.confirmers.base import BaseConfirmer, ConfirmerContext


def _series_or_default(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


class B1SemanticBonusConfirmer(BaseConfirmer):
    """关键K / 缩半量 / 倍量柱默认只做加分项。"""

    strategy_family = "b1"
    name = "b1.semantic_bonus"

    def apply(self, context: ConfirmerContext) -> pd.DataFrame:
        df = context.candidate_df.copy()
        bonus = (
            df.get("key_k_support", False).fillna(False).astype(int)
            + df.get("half_volume", False).fillna(False).astype(int)
            + df.get("double_bull_exist_60", False).fillna(False).astype(int)
        ).astype(float)
        txt_bonus = _series_or_default(df, "txt_confirm_bonus", 0.0)
        semantic_bonus = _series_or_default(df, "buy_semantic_score", 0.0)
        df["confirmer_score"] = bonus + txt_bonus + 0.1 * semantic_bonus
        return df
