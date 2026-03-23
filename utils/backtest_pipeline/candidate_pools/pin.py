from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from utils import pinfilter, technical_indicators
from utils.backtest_pipeline.candidate_pools.base import BaseCandidatePool, CandidatePoolContext
from utils.backtest_pipeline.candidate_pools.bridge_utils import iter_data_files
from utils.strategy_feature_cache import StrategyFeatureCache


def _scan_pin_rows(data_dir: str, candidate_pool: str, structure_only: bool) -> pd.DataFrame:
    rows: List[dict] = []
    for path in iter_data_files(data_dir):
        cache = StrategyFeatureCache(str(path))
        weekly_ok, weekly_reason = cache.weekly_screen()
        if not weekly_ok:
            continue
        df_cn = cache.daily_cn_df()
        if df_cn is None or len(df_cn) < 40:
            continue
        feat = cache.pin_today_features()
        if feat is None:
            continue
        if feat["trend_line"] <= feat["long_line"]:
            continue
        if not technical_indicators.caculate_pin(df_cn):
            continue

        matched = pinfilter.detect_subtypes(feat)
        if not matched:
            continue
        if structure_only and "C型(结构支撑)" not in matched:
            continue
        if (not structure_only) and not any(name in matched for name in ("A型(缩量回踩)", "B型(强趋势加速)", "C型(结构支撑)")):
            continue

        code = Path(path).stem
        rows.append(
            {
                "code": code,
                "signal_date": pd.Timestamp(df_cn["日期"].iloc[-1]),
                "entry_date": pd.Timestamp(df_cn["日期"].iloc[-1]),
                "strategy_family": "pin",
                "candidate_pool": candidate_pool,
                "signal_type": "pin_structure_support" if structure_only else "pin_trend_wash",
                "base_score": float(
                    0.40 * (1.0 if "A型(缩量回踩)" in matched else 0.0)
                    + 0.25 * (1.0 if "B型(强趋势加速)" in matched else 0.0)
                    + 0.35 * (1.0 if "C型(结构支撑)" in matched else 0.0)
                ),
                "matched_subtypes": "+".join(matched),
                "along_trend_up": bool(feat.get("along_trend_up", False)),
                "n_up_any": bool(feat.get("n_up_any", False)),
                "keyk_support_active": bool(feat.get("keyk_support_active", False)),
                "signal_vs_ma20": float(feat.get("signal_vs_ma20", np.nan)),
                "vol_vs_prev": float(feat.get("vol_vs_prev", np.nan)),
                "ret3": float(feat.get("ret3", np.nan)),
                "ret10": float(feat.get("ret10", np.nan)),
                "trend_line_lead": float(feat.get("trend_line_lead", np.nan)),
                "note": f"周线：{weekly_reason} | 日线：{'+'.join(matched)}",
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


class PinTrendWashPool(BaseCandidatePool):
    """单针：强趋势中的洗盘回踩池。"""

    strategy_family = "pin"
    name = "pin.trend_wash"
    requires_market_data = False

    def generate(self, context: CandidatePoolContext) -> pd.DataFrame:
        return _scan_pin_rows(context.data_dir, self.name, structure_only=False)


class PinStructureSupportPool(BaseCandidatePool):
    """单针：关键K / 结构支撑型候选池。"""

    strategy_family = "pin"
    name = "pin.structure_support"
    requires_market_data = False

    def generate(self, context: CandidatePoolContext) -> pd.DataFrame:
        return _scan_pin_rows(context.data_dir, self.name, structure_only=True)
