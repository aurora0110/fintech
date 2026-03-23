from __future__ import annotations

import pandas as pd

from utils.backtest_pipeline.rankers.base import BaseRanker, RankerContext


def _series_or_default(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _resolve_score_alias(name: str) -> str:
    alias_map = {
        "similarity": "template_similarity_score",
        "factor_discovery": "discovery_factor_score",
        "xgboost": "xgb_full_score",
        "lightgbm": "lgb_full_score",
        "naive_bayes": "gnb_full_score",
        "template_full_fusion": "template_hard_full_fusion_score",
    }
    return alias_map.get(name, name)


class SimilarityRanker(BaseRanker):
    name = "ranker.similarity"

    def score(self, context: RankerContext) -> pd.DataFrame:
        df = context.candidate_df.copy()
        score_col = _resolve_score_alias(str(context.params.get("score_col", "template_similarity_score")))
        df["rank_score"] = _series_or_default(df, score_col, 0.0)
        return df


class FactorDiscoveryRanker(BaseRanker):
    name = "ranker.factor_discovery"

    def score(self, context: RankerContext) -> pd.DataFrame:
        df = context.candidate_df.copy()
        score_col = _resolve_score_alias(str(context.params.get("score_col", "discovery_factor_score")))
        df["rank_score"] = _series_or_default(df, score_col, 0.0)
        return df


class XGBoostRanker(BaseRanker):
    name = "ranker.xgboost"

    def score(self, context: RankerContext) -> pd.DataFrame:
        df = context.candidate_df.copy()
        score_col = _resolve_score_alias(str(context.params.get("score_col", "xgb_full_score")))
        df["rank_score"] = _series_or_default(df, score_col, 0.0)
        return df


class LightGBMRanker(BaseRanker):
    name = "ranker.lightgbm"

    def score(self, context: RankerContext) -> pd.DataFrame:
        df = context.candidate_df.copy()
        score_col = _resolve_score_alias(str(context.params.get("score_col", "lgb_full_score")))
        df["rank_score"] = _series_or_default(df, score_col, 0.0)
        return df


class NaiveBayesRanker(BaseRanker):
    name = "ranker.naive_bayes"

    def score(self, context: RankerContext) -> pd.DataFrame:
        df = context.candidate_df.copy()
        score_col = _resolve_score_alias(str(context.params.get("score_col", "gnb_full_score")))
        df["rank_score"] = _series_or_default(df, score_col, 0.0)
        return df


class ReinforcementLearningRanker(BaseRanker):
    """占位模块：后续 RL 直接以模块形式接入，而不是改主流程。"""

    name = "ranker.reinforcement_learning"

    def score(self, context: RankerContext) -> pd.DataFrame:
        df = context.candidate_df.copy()
        score_col = _resolve_score_alias(str(context.params.get("score_col", "rl_score")))
        df["rank_score"] = _series_or_default(df, score_col, 0.0)
        return df


class FusionRanker(BaseRanker):
    name = "ranker.fusion"

    def score(self, context: RankerContext) -> pd.DataFrame:
        df = context.candidate_df.copy()
        components = context.params.get("components") or []
        if not components:
            components = [
                ("template_hard_full_fusion_score", 1.0),
            ]
        normalized = []
        for item in components:
            if isinstance(item, str):
                normalized.append((_resolve_score_alias(item), 1.0))
            else:
                normalized.append((_resolve_score_alias(str(item[0])), float(item[1])))
        score = pd.Series(0.0, index=df.index, dtype=float)
        total_weight = 0.0
        for col, weight in normalized:
            score = score + _series_or_default(df, col, 0.0) * weight
            total_weight += weight
        df["rank_score"] = score / total_weight if total_weight > 0 else score
        return df
