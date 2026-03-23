from __future__ import annotations

import pandas as pd

from utils.backtest_pipeline.artifacts import load_b1_base_candidates, load_b1_template_candidates
from utils.backtest_pipeline.candidate_pools.base import BaseCandidatePool, CandidatePoolContext


class B1LowCrossPool(BaseCandidatePool):
    """B1 主冠军池：低位 + 主回踩语义。"""

    strategy_family = "b1"
    name = "b1.low_cross"
    requires_market_data = False

    def generate(self, context: CandidatePoolContext) -> pd.DataFrame:
        df = load_b1_base_candidates()
        df = df[df.get("pool_low_cross", False).fillna(False).astype(bool)].copy()
        df["strategy_family"] = self.strategy_family
        df["candidate_pool"] = self.name
        df["signal_type"] = "b1_low_cross"
        df["base_score"] = pd.to_numeric(df.get("discovery_factor_score", 0.0), errors="coerce").fillna(0.0)
        return df


class B1TxtConfirmedPool(BaseCandidatePool):
    """B1 文本模板确认池：回踩趋势线 / 回踩多空线 + 正例模板语义。"""

    strategy_family = "b1"
    name = "b1.txt_confirmed"
    requires_market_data = False

    def generate(self, context: CandidatePoolContext) -> pd.DataFrame:
        df = load_b1_template_candidates()
        df = df[df.get("pool_txt_confirmed", False).fillna(False).astype(bool)].copy()
        df["strategy_family"] = self.strategy_family
        df["candidate_pool"] = self.name
        df["signal_type"] = "b1_txt_confirmed"
        df["base_score"] = pd.to_numeric(df.get("template_hard_full_fusion_score", 0.0), errors="coerce").fillna(0.0)
        return df
