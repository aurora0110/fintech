from __future__ import annotations

import pandas as pd

from utils import b3filter
from utils.backtest_pipeline.candidate_pools.base import BaseCandidatePool, CandidatePoolContext
from utils.backtest_pipeline.candidate_pools.bridge_utils import scan_with_check


class B3FollowThroughPool(BaseCandidatePool):
    """B3：B2 后承接确认的主候选池。"""

    strategy_family = "b3"
    name = "b3.follow_through"
    requires_market_data = False

    def generate(self, context: CandidatePoolContext) -> pd.DataFrame:
        return scan_with_check(
            data_dir=context.data_dir,
            scan_fn=lambda file_path, cache: b3filter.check(file_path, hold_list=[], feature_cache=cache),
            strategy_family=self.strategy_family,
            candidate_pool=self.name,
            signal_type="b3_follow_through",
        )


class B3WeeklyAlignedPool(BaseCandidatePool):
    """B3：带周线背景一致性的候选池。"""

    strategy_family = "b3"
    name = "b3.weekly_aligned"

    def generate(self, context: CandidatePoolContext) -> pd.DataFrame:
        return pd.DataFrame(
            columns=["code", "signal_date", "entry_date", "strategy_family", "candidate_pool", "signal_type", "base_score"]
        )
