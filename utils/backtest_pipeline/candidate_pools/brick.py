from __future__ import annotations

import pandas as pd

from utils import brick_filter
from utils.backtest_pipeline.candidate_pools.base import BaseCandidatePool, CandidatePoolContext
from utils.backtest_pipeline.candidate_pools.bridge_utils import scan_with_check


class BrickMainPool(BaseCandidatePool):
    strategy_family = "brick"
    name = "brick.main"
    requires_market_data = False

    def generate(self, context: CandidatePoolContext) -> pd.DataFrame:
        return scan_with_check(
            data_dir=context.data_dir,
            scan_fn=lambda file_path, cache: brick_filter.check(file_path, hold_list=[], feature_cache=cache),
            strategy_family=self.strategy_family,
            candidate_pool=self.name,
            signal_type="brick_main",
        )
