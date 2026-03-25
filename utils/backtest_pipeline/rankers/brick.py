from __future__ import annotations

import pandas as pd

from utils.backtest_pipeline.rankers.base import BaseRanker, RankerContext


def _series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


class BrickSimilarityChampionRanker(BaseRanker):
    strategy_family = "brick"
    name = "ranker.brick_similarity_champion"

    def score(self, context: RankerContext) -> pd.DataFrame:
        df = context.candidate_df.copy()
        score_col = str(context.params.get("score_col", "sim_score"))
        pool_bonus = _series(df, "pool_bonus", 0.0)
        df["rank_score"] = _series(df, score_col, 0.0) + pool_bonus
        return df


class BrickFactorScoreRanker(BaseRanker):
    strategy_family = "brick"
    name = "ranker.brick_factor_score"

    def score(self, context: RankerContext) -> pd.DataFrame:
        df = context.candidate_df.copy()
        score_col = str(context.params.get("score_col", "factor_score"))
        pool_bonus = _series(df, "pool_bonus", 0.0)
        df["rank_score"] = _series(df, score_col, 0.0) + pool_bonus
        return df


class BrickSimilarityPlusFactorRanker(BaseRanker):
    strategy_family = "brick"
    name = "ranker.brick_similarity_plus_factor"

    def score(self, context: RankerContext) -> pd.DataFrame:
        df = context.candidate_df.copy()
        sim_col = str(context.params.get("sim_col", "sim_score"))
        factor_col = str(context.params.get("factor_col", "factor_score"))
        sim_weight = float(context.params.get("sim_weight", 0.8))
        factor_weight = float(context.params.get("factor_weight", 0.2))
        pool_bonus = _series(df, "pool_bonus", 0.0)
        df["rank_score"] = (
            _series(df, sim_col, 0.0) * sim_weight
            + _series(df, factor_col, 0.0) * factor_weight
            + pool_bonus
        )
        return df


class BrickSimilarityPlusMlRanker(BaseRanker):
    strategy_family = "brick"
    name = "ranker.brick_similarity_plus_ml"

    def score(self, context: RankerContext) -> pd.DataFrame:
        df = context.candidate_df.copy()
        score_col = str(context.params.get("score_col", "ml_score"))
        pool_bonus = _series(df, "pool_bonus", 0.0)
        df["rank_score"] = _series(df, score_col, 0.0) + pool_bonus
        return df


class BrickFullFusionRanker(BaseRanker):
    strategy_family = "brick"
    name = "ranker.brick_full_fusion"

    def score(self, context: RankerContext) -> pd.DataFrame:
        df = context.candidate_df.copy()
        sim_weight = float(context.params.get("sim_weight", 0.5))
        factor_weight = float(context.params.get("factor_weight", 0.2))
        ml_weight = float(context.params.get("ml_weight", 0.3))
        pool_bonus = _series(df, "pool_bonus", 0.0)
        df["rank_score"] = (
            _series(df, str(context.params.get("sim_col", "sim_score")), 0.0) * sim_weight
            + _series(df, str(context.params.get("factor_col", "factor_score")), 0.0) * factor_weight
            + _series(df, str(context.params.get("ml_col", "ml_score")), 0.0) * ml_weight
            + pool_bonus
        )
        return df
