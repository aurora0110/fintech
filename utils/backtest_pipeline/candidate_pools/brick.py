from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from utils import brick_filter
from utils.backtest_pipeline.candidate_pools.base import BaseCandidatePool, CandidatePoolContext
from utils.backtest_pipeline.candidate_pools.bridge_utils import scan_with_check


BRICK_BASE_PATH = Path("/Users/lidongyang/Desktop/Qstrategy/utils/backtest/run_momentum_tail_experiment.py")
BRICK_RANKING_PATH = Path("/Users/lidongyang/Desktop/Qstrategy/utils/backtest/compare_momentum_tail_ranking_models.py")
TOP_N = 10
PCT_RANK_THRESHOLD = 0.50


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


brick_base = _load_module(BRICK_BASE_PATH, "brick_pipeline_base")
brick_ranking = _load_module(BRICK_RANKING_PATH, "brick_pipeline_ranking")


def _load_feature_map(data_dir: str, file_limit: int = 0) -> Dict[str, pd.DataFrame]:
    feature_map: Dict[str, pd.DataFrame] = {}
    files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith((".csv", ".txt"))])
    if file_limit > 0:
        files = files[:file_limit]
    for file_name in files:
        df = brick_base.load_one_csv(os.path.join(data_dir, file_name))
        if df is None:
            continue
        code = str(df["code"].iloc[0])
        feature_map[code] = brick_base.build_feature_df(df)
    return feature_map


def _build_formal_best_candidates(data_dir: str, file_limit: int = 0) -> pd.DataFrame:
    combo = brick_base.Combo(
        rebound_threshold=1.2,
        gain_limit=0.08,
        take_profit=0.03,
        stop_mode="entry_low_x_0.99",
    )
    rows: List[dict] = []
    feature_map = _load_feature_map(data_dir, file_limit=file_limit)
    for code, raw_df in feature_map.items():
        df = brick_ranking.add_long_line(raw_df)
        mask_a = df["pattern_a"] & (df["rebound_ratio"] >= combo.rebound_threshold)
        mask_b = df["pattern_b"] & (df["rebound_ratio"] >= 1.0)
        mask = (
            df["signal_base"]
            & (df["ret1"] <= combo.gain_limit)
            & (mask_a | mask_b)
            & (df["trend_line"] > df["long_line"])
        )
        signal_idxs = np.flatnonzero(mask.to_numpy())
        for signal_idx in signal_idxs:
            signal_idx = int(signal_idx)
            entry_idx = signal_idx + 1
            if entry_idx >= len(df):
                continue
            feat = brick_ranking.compute_signal_features(df, signal_idx)
            rows.append(
                {
                    "code": code,
                    "signal_idx": signal_idx,
                    "signal_date": df.at[signal_idx, "date"],
                    "entry_date": df.at[entry_idx, "date"],
                    "candidate_pool": "brick.formal_best",
                    "strategy_family": "brick",
                    "signal_type": "brick_formal_best",
                    "pattern_a": bool(df.at[signal_idx, "pattern_a"]),
                    "pattern_b": bool(df.at[signal_idx, "pattern_b"]),
                    "ret1": float(df.at[signal_idx, "ret1"]),
                    "rebound_ratio": float(df.at[signal_idx, "rebound_ratio"]),
                    "brick_red_len": float(df.at[signal_idx, "brick_red_len"]),
                    "signal_vs_ma5": float(df.at[signal_idx, "signal_vs_ma5"]),
                    "trend_line": float(df.at[signal_idx, "trend_line"]),
                    "long_line": float(df.at[signal_idx, "long_line"]),
                    "base_score": 0.0,
                    **feat,
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["signal_date", "code"]).reset_index(drop=True)
    out = brick_ranking.assign_rank_scores(out)
    out["base_score"] = brick_ranking.build_sort_score(out, "shrink_focus")
    out["daily_pct_rank"] = out.groupby("signal_date")["base_score"].rank(pct=True, method="first")
    out = out[out["daily_pct_rank"] >= PCT_RANK_THRESHOLD].copy()
    out = out.sort_values(["signal_date", "base_score", "code"], ascending=[True, False, True])
    out = out.groupby("signal_date", group_keys=False).head(TOP_N).reset_index(drop=True)
    return out


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


class BrickFormalBestPool(BaseCandidatePool):
    strategy_family = "brick"
    name = "brick.formal_best"
    requires_market_data = False

    def generate(self, context: CandidatePoolContext) -> pd.DataFrame:
        file_limit = int(context.strategy_config.params.get("file_limit", 0) or 0)
        return _build_formal_best_candidates(context.data_dir, file_limit=file_limit)
