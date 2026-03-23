from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from utils.backtest_pipeline.candidate_pools.base import BaseCandidatePool, CandidatePoolContext
from utils.backtest_pipeline.candidate_pools.bridge_utils import iter_data_files
from utils.strategy_feature_cache import StrategyFeatureCache


def _scan_b3_rows(data_dir: str, candidate_pool: str, weekly_only: bool) -> pd.DataFrame:
    rows: List[dict] = []
    for path in iter_data_files(data_dir):
        cache = StrategyFeatureCache(str(path))
        weekly_ok, weekly_reason = cache.weekly_screen()
        if not weekly_ok:
            continue
        x = cache.b3_features()
        if x is None or x.empty:
            continue
        latest = x.iloc[-1]
        if not bool(latest.get("b3_signal", False)):
            continue

        # weekly_aligned 目前收敛为“有明确周线强势/进碗背景”的 B3。
        if weekly_only and not weekly_reason:
            continue

        code = Path(path).stem
        weekly_bonus = 0.10 if "周线强势" in weekly_reason else 0.05 if "周线进碗" in weekly_reason else 0.0
        rows.append(
            {
                "code": code,
                "signal_date": pd.Timestamp(latest["date"]),
                "entry_date": pd.Timestamp(latest["date"]),
                "strategy_family": "b3",
                "candidate_pool": candidate_pool,
                "signal_type": "b3_weekly_aligned" if weekly_only else "b3_follow_through",
                "base_score": float(latest.get("b3_score", 0.0) or 0.0) + weekly_bonus,
                "b3_score": float(latest.get("b3_score", 0.0) or 0.0),
                "ret1": float(latest.get("ret1", np.nan)),
                "amplitude": float(latest.get("amplitude", np.nan)),
                "vol_vs_prev": float(latest.get("vol_vs_prev", np.nan)),
                "prev_b2_any": bool(latest.get("prev_b2_any", False)),
                "daily_b1_signal": bool(latest.get("daily_b1_signal", False)),
                "weekly_reason": weekly_reason,
                "stop_loss_price": float(latest.get("low", np.nan)),
                "close_price": float(latest.get("close", np.nan)),
                "note": weekly_reason,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "code",
                "signal_date",
                "entry_date",
                "strategy_family",
                "candidate_pool",
                "signal_type",
                "base_score",
            ]
        )
    return pd.DataFrame(rows).sort_values(["signal_date", "base_score", "code"], ascending=[True, False, True]).reset_index(drop=True)


class B3FollowThroughPool(BaseCandidatePool):
    """B3：B2 后承接确认的主候选池。"""

    strategy_family = "b3"
    name = "b3.follow_through"
    requires_market_data = False

    def generate(self, context: CandidatePoolContext) -> pd.DataFrame:
        return _scan_b3_rows(context.data_dir, self.name, weekly_only=False)


class B3WeeklyAlignedPool(BaseCandidatePool):
    """B3：带周线背景一致性的候选池。"""

    strategy_family = "b3"
    name = "b3.weekly_aligned"
    requires_market_data = False

    def generate(self, context: CandidatePoolContext) -> pd.DataFrame:
        return _scan_b3_rows(context.data_dir, self.name, weekly_only=True)
