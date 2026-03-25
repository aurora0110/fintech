from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils import brick_filter
from utils.backtest_pipeline.candidate_pools.base import BaseCandidatePool, CandidatePoolContext
from utils.backtest_pipeline.candidate_pools.bridge_utils import scan_with_check


BRICK_BASE_PATH = Path("/Users/lidongyang/Desktop/Qstrategy/utils/backtest/run_momentum_tail_experiment.py")
BRICK_RANKING_PATH = Path("/Users/lidongyang/Desktop/Qstrategy/utils/backtest/compare_momentum_tail_ranking_models.py")
BRICK_SIM_PATH = Path("/Users/lidongyang/Desktop/Qstrategy/utils/tmp/similarity_filter_research.py")
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
brick_similarity = _load_module(BRICK_SIM_PATH, "brick_similarity_pipeline")
_RELAXED_CANDIDATE_CACHE: Dict[Tuple[str, int], pd.DataFrame] = {}


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


def _assign_turn_strength_layer(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=str)
    source = pd.to_numeric(df["brick_green_len_prev"], errors="coerce").fillna(0.0)
    q1 = float(source.quantile(0.33))
    q2 = float(source.quantile(0.67))
    if abs(q2 - q1) < 1e-12:
        q1 = float(source.median())
        q2 = q1
    layer = np.where(source <= q1, "low", np.where(source >= q2, "high", "mid"))
    return pd.Series(layer, index=df.index, dtype=object)


def _build_relaxed_base_candidates(data_dir: str, file_limit: int = 0) -> pd.DataFrame:
    cache_key = (data_dir, int(file_limit))
    if cache_key in _RELAXED_CANDIDATE_CACHE:
        return _RELAXED_CANDIDATE_CACHE[cache_key].copy()

    records = brick_similarity.load_full_signal_dataset(file_limit=file_limit, data_dir=Path(data_dir))
    if not records:
        out = pd.DataFrame()
        _RELAXED_CANDIDATE_CACHE[cache_key] = out
        return out.copy()

    rows = []
    for record in records:
        rows.append(
            {
                "code": record["code"],
                "signal_idx": int(record["signal_idx"]),
                "signal_date": pd.Timestamp(record["date"]),
                "entry_date": pd.Timestamp(record["entry_date"]),
                "candidate_pool": "brick.relaxed_base",
                "strategy_family": "brick",
                "signal_type": "brick_relaxed",
                "result": record["result"],
                "label": int(record["label"]),
                "ret": float(record["ret"]),
                "entry_price": float(record["entry_price"]),
                "signal_low": float(record.get("signal_low", np.nan)),
                "ret1": float(record["ret1"]),
                "ret5": float(record["ret5"]),
                "ret10": float(record["ret10"]),
                "signal_ret": float(record["signal_ret"]),
                "trend_spread": float(record["trend_spread"]),
                "close_to_trend": float(record["close_to_trend"]),
                "close_to_long": float(record["close_to_long"]),
                "brick_red_len": float(record["brick_red_len"]),
                "brick_green_len": float(record.get("brick_green_len", 0.0)),
                "brick_green_len_prev": float(record["brick_green_len_prev"]),
                "rebound_ratio": float(record["rebound_ratio"]),
                "RSI14": float(record["RSI14"]),
                "MACD_hist": float(record["MACD_hist"]),
                "body_ratio": float(record["body_ratio"]),
                "upper_shadow_pct": float(record["upper_shadow_pct"]),
                "lower_shadow_pct": float(record["lower_shadow_pct"]),
                "prev_green_streak": int(record.get("prev_green_streak", 0)),
                "prev_red_streak": int(record.get("prev_red_streak", 0)),
                "trend_ok": bool(record.get("trend_ok", False)),
                "green4_flag": bool(record.get("green4_flag", False)),
                "red4_flag": bool(record.get("red4_flag", False)),
                "pattern_a_relaxed": bool(record.get("pattern_a_relaxed", False)),
                "pattern_b_relaxed": bool(record.get("pattern_b_relaxed", False)),
                "base_score": 0.0,
                "pool_bonus": 0.0,
            }
        )

    out = pd.DataFrame(rows).sort_values(["signal_date", "code"]).reset_index(drop=True)
    out["trend_layer"] = np.where(out["trend_ok"], "high", "low")
    out["green4_low_flag"] = out["green4_flag"] & (out["trend_layer"] == "low")
    out["turn_strength_layer"] = _assign_turn_strength_layer(out)
    _RELAXED_CANDIDATE_CACHE[cache_key] = out.copy()
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


class BrickRelaxedBasePool(BaseCandidatePool):
    strategy_family = "brick"
    name = "brick.relaxed_base"
    requires_market_data = False

    def generate(self, context: CandidatePoolContext) -> pd.DataFrame:
        file_limit = int(context.strategy_config.params.get("file_limit", 0) or 0)
        return _build_relaxed_base_candidates(context.data_dir, file_limit=file_limit)


class BrickGreen4EnhancePool(BaseCandidatePool):
    strategy_family = "brick"
    name = "brick.green4_enhance"
    requires_market_data = False

    def generate(self, context: CandidatePoolContext) -> pd.DataFrame:
        df = BrickRelaxedBasePool().generate(context).copy()
        if df.empty:
            return df
        df["candidate_pool"] = self.name
        df["pool_bonus"] = np.where(df["green4_flag"], 0.10, 0.0)
        return df


class BrickGreen4LowEnhancePool(BaseCandidatePool):
    strategy_family = "brick"
    name = "brick.green4_low_enhance"
    requires_market_data = False

    def generate(self, context: CandidatePoolContext) -> pd.DataFrame:
        df = BrickRelaxedBasePool().generate(context).copy()
        if df.empty:
            return df
        df["candidate_pool"] = self.name
        df["pool_bonus"] = np.where(df["green4_low_flag"], 0.15, 0.0)
        return df


class BrickRed4FilterPool(BaseCandidatePool):
    strategy_family = "brick"
    name = "brick.red4_filter"
    requires_market_data = False

    def generate(self, context: CandidatePoolContext) -> pd.DataFrame:
        df = BrickRelaxedBasePool().generate(context).copy()
        if df.empty:
            return df
        df["candidate_pool"] = self.name
        df["pool_bonus"] = np.where(df["red4_flag"], -0.10, 0.0)
        return df


class BrickGreen4LowHardFilterPool(BaseCandidatePool):
    strategy_family = "brick"
    name = "brick.green4_low_hardfilter"
    requires_market_data = False

    def generate(self, context: CandidatePoolContext) -> pd.DataFrame:
        df = BrickRelaxedBasePool().generate(context).copy()
        if df.empty:
            return df
        df = df[df["green4_low_flag"]].copy()
        df["candidate_pool"] = self.name
        df["pool_bonus"] = 0.20
        return df.reset_index(drop=True)
