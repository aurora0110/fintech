from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from utils import b2filter
from utils.backtest_pipeline.candidate_pools.base import BaseCandidatePool, CandidatePoolContext
from utils.backtest_pipeline.candidate_pools.bridge_utils import iter_data_files
from utils.strategy_feature_cache import StrategyFeatureCache


TYPE1_NEAR_RATIO = 1.02
TYPE1_J_RANK20_MAX = 0.10
TYPE4_TOUCH_RATIO = 1.01
TYPE4_LOOKBACK = 20


def _scan_b2_rows(data_dir: str, candidate_pool: str, mode: str) -> pd.DataFrame:
    rows: List[dict] = []
    for path in iter_data_files(data_dir):
        cache = StrategyFeatureCache(str(path))
        x = cache.b2_features()
        if x is None or x.empty or len(x) < 30:
            continue
        latest = x.iloc[-1]
        if not bool(latest.get("b2_signal", False)):
            continue

        prev = x.iloc[-2] if len(x) >= 2 else latest
        type1 = bool(
            pd.notna(prev.get("close"))
            and pd.notna(prev.get("long_line"))
            and float(prev["close"]) <= float(prev["long_line"]) * TYPE1_NEAR_RATIO
            and pd.notna(latest.get("j_rank20_prev"))
            and float(latest["j_rank20_prev"]) <= TYPE1_J_RANK20_MAX
        )

        bull_cross = (x["trend_line"] > x["long_line"]) & (x["trend_line"].shift(1) <= x["long_line"].shift(1))
        left = max(1, len(x) - TYPE4_LOOKBACK)
        crosses = np.where(bull_cross.iloc[left:].to_numpy())[0]
        type4 = False
        if len(crosses) > 0 and len(x) >= 2:
            cross_idx = left + int(crosses[-1])
            prev_touch = (
                float(prev["low"]) <= float(prev["trend_line"]) * TYPE4_TOUCH_RATIO
                or float(prev["close"]) <= float(prev["trend_line"]) * TYPE4_TOUCH_RATIO
            )
            if prev_touch:
                if len(x) - 2 > cross_idx:
                    between = x.iloc[cross_idx + 1 : len(x) - 1]
                    had_touch = (
                        (between["low"] <= between["trend_line"] * TYPE4_TOUCH_RATIO)
                        | (between["close"] <= between["trend_line"] * TYPE4_TOUCH_RATIO)
                    ).any()
                    type4 = not bool(had_touch)
                else:
                    type4 = True

        if mode == "type1" and not type1:
            continue
        if mode == "type4" and not type4:
            continue

        code = Path(path).stem
        rows.append(
            {
                "code": code,
                "signal_date": pd.Timestamp(latest["date"]),
                "entry_date": pd.Timestamp(latest["date"]),
                "strategy_family": "b2",
                "candidate_pool": candidate_pool,
                "signal_type": mode if mode != "main" else "b2_main",
                "base_score": float(latest.get("sort_score", 0.0) or 0.0),
                "b2_sort_score": float(latest.get("sort_score", 0.0) or 0.0),
                "ret1": float(latest.get("ret1", 0.0) or 0.0),
                "j_value": float(latest.get("J", np.nan)),
                "j_rank20_prev": float(latest.get("j_rank20_prev", np.nan)),
                "signal_vs_ma5": float(latest.get("signal_vs_ma5", np.nan)),
                "close_position": float(latest.get("close_position", np.nan)),
                "trend_line_lead": float((latest.get("trend_line", np.nan) - latest.get("long_line", np.nan)) / latest.get("close", np.nan))
                if pd.notna(latest.get("close")) and float(latest.get("close", 0.0) or 0.0) > 0
                else np.nan,
                "near_long_prev_ratio": float(prev.get("close", np.nan) / prev.get("long_line", np.nan))
                if pd.notna(prev.get("long_line")) and abs(float(prev.get("long_line", 0.0) or 0.0)) > 1e-12
                else np.nan,
                "type1": bool(type1),
                "type4": bool(type4),
                "stop_loss_price": float(latest.get("low", np.nan)),
                "close_price": float(latest.get("close", np.nan)),
                "note": "B2" if mode == "main" else f"B2_{mode}",
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
    return pd.DataFrame(rows).sort_values(["signal_date", "code"]).reset_index(drop=True)


class B2MainPool(BaseCandidatePool):
    strategy_family = "b2"
    name = "b2.main"
    requires_market_data = False

    def generate(self, context: CandidatePoolContext) -> pd.DataFrame:
        return _scan_b2_rows(context.data_dir, self.name, "main")


class B2Type1Pool(BaseCandidatePool):
    strategy_family = "b2"
    name = "b2.type1"
    requires_market_data = False

    def generate(self, context: CandidatePoolContext) -> pd.DataFrame:
        return _scan_b2_rows(context.data_dir, self.name, "type1")


class B2Type4Pool(BaseCandidatePool):
    strategy_family = "b2"
    name = "b2.type4"
    requires_market_data = False

    def generate(self, context: CandidatePoolContext) -> pd.DataFrame:
        return _scan_b2_rows(context.data_dir, self.name, "type4")
