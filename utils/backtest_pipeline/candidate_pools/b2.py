from __future__ import annotations

import pandas as pd

from utils import b2filter
from utils.backtest_pipeline.candidate_pools.base import BaseCandidatePool, CandidatePoolContext
from utils.backtest_pipeline.candidate_pools.bridge_utils import scan_with_check


class B2MainPool(BaseCandidatePool):
    strategy_family = "b2"
    name = "b2.main"
    requires_market_data = False

    def generate(self, context: CandidatePoolContext) -> pd.DataFrame:
        return scan_with_check(
            data_dir=context.data_dir,
            scan_fn=lambda file_path, cache: b2filter.check(file_path, hold_list=[], feature_cache=cache),
            strategy_family=self.strategy_family,
            candidate_pool=self.name,
            signal_type="b2_main",
        )


class B2Type1Pool(BaseCandidatePool):
    strategy_family = "b2"
    name = "b2.type1"

    def generate(self, context: CandidatePoolContext) -> pd.DataFrame:
        return pd.DataFrame(
            columns=["code", "signal_date", "entry_date", "strategy_family", "candidate_pool", "signal_type", "base_score"]
        )


class B2Type4Pool(BaseCandidatePool):
    strategy_family = "b2"
    name = "b2.type4"

    def generate(self, context: CandidatePoolContext) -> pd.DataFrame:
        return pd.DataFrame(
            columns=["code", "signal_date", "entry_date", "strategy_family", "candidate_pool", "signal_type", "base_score"]
        )
