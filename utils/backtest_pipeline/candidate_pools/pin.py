from __future__ import annotations

import pandas as pd

from utils import pinfilter
from utils.backtest_pipeline.candidate_pools.base import BaseCandidatePool, CandidatePoolContext
from utils.backtest_pipeline.candidate_pools.bridge_utils import scan_with_check


class PinTrendWashPool(BaseCandidatePool):
    """单针：强趋势中的洗盘回踩池。"""

    strategy_family = "pin"
    name = "pin.trend_wash"
    requires_market_data = False

    def generate(self, context: CandidatePoolContext) -> pd.DataFrame:
        return scan_with_check(
            data_dir=context.data_dir,
            scan_fn=lambda file_path, cache: pinfilter.check(file_path, feature_cache=cache),
            strategy_family=self.strategy_family,
            candidate_pool=self.name,
            signal_type="pin_trend_wash",
        )


class PinStructureSupportPool(BaseCandidatePool):
    """单针：关键K / 结构支撑型候选池。"""

    strategy_family = "pin"
    name = "pin.structure_support"

    def generate(self, context: CandidatePoolContext) -> pd.DataFrame:
        return pd.DataFrame(
            columns=["code", "signal_date", "entry_date", "strategy_family", "candidate_pool", "signal_type", "base_score"]
        )
