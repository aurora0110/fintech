from __future__ import annotations

from itertools import combinations
from typing import Dict, List
import numpy as np
import pandas as pd

from backtester import backtest_topk
from config import RunConfig
from metrics import objective_value


def _perf(
    df: pd.DataFrame,
    cfg: RunConfig,
    factor_weights: Dict[str, float],
    threshold_params: Dict[str, float],
) -> float:
    bt = backtest_topk(df, cfg, factor_weights=factor_weights, threshold_params=threshold_params)
    return objective_value(bt.metrics, cfg.objective_name, cfg.objective_weights)


def marginal_contribution(
    df: pd.DataFrame,
    cfg: RunConfig,
    full_weights: Dict[str, float],
    threshold_params: Dict[str, float],
) -> pd.DataFrame:
    base = _perf(df, cfg, full_weights, threshold_params)
    rows = []
    for f in full_weights:
        sub = {k: v for k, v in full_weights.items() if k != f}
        drop_perf = _perf(df, cfg, sub, threshold_params)
        rows.append(
            {
                "factor": f,
                "full_perf": base,
                "without_factor_perf": drop_perf,
                "contribution": base - drop_perf,
                "label": "positive" if base - drop_perf > 0 else "negative",
            }
        )
    return pd.DataFrame(rows).sort_values("contribution", ascending=False).reset_index(drop=True)


def standalone_factor_performance(
    df: pd.DataFrame,
    cfg: RunConfig,
    factors: List[str],
    threshold_params: Dict[str, float],
) -> pd.DataFrame:
    rows = []
    for f in factors:
        perf = _perf(df, cfg, {f: 1.0}, threshold_params)
        bt = backtest_topk(df, cfg, factor_weights={f: 1.0}, threshold_params=threshold_params)
        rows.append(
            {
                "factor": f,
                "objective": perf,
                "annual_return": bt.metrics["annual_return"],
                "max_drawdown": bt.metrics["max_drawdown"],
                "sharpe": bt.metrics["sharpe"],
            }
        )
    return pd.DataFrame(rows).sort_values("objective", ascending=False).reset_index(drop=True)


def shapley_contribution(
    df: pd.DataFrame,
    cfg: RunConfig,
    factors: List[str],
    threshold_params: Dict[str, float],
) -> pd.DataFrame:
    if len(factors) > 8:
        raise ValueError("Shapley contribution only supports <= 8 factors.")

    perf_cache: Dict[tuple, float] = {}

    def perf_of(subset: List[str]) -> float:
        key = tuple(sorted(subset))
        if key in perf_cache:
            return perf_cache[key]
        if not subset:
            perf_cache[key] = 0.0
            return 0.0
        w = {f: 1.0 / len(subset) for f in subset}
        perf_cache[key] = _perf(df, cfg, w, threshold_params)
        return perf_cache[key]

    n = len(factors)
    shapley = {f: 0.0 for f in factors}
    for f in factors:
        others = [x for x in factors if x != f]
        for r in range(len(others) + 1):
            for sub in combinations(others, r):
                sub = list(sub)
                with_f = sub + [f]
                marg = perf_of(with_f) - perf_of(sub)
                weight = 1.0 / (n * _comb(n - 1, r))
                shapley[f] += weight * marg

    rows = [{"factor": f, "shapley": v} for f, v in shapley.items()]
    return pd.DataFrame(rows).sort_values("shapley", ascending=False).reset_index(drop=True)


def _comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    if k == 0:
        return 1
    num = 1
    den = 1
    for i in range(k):
        num *= (n - i)
        den *= (i + 1)
    return num // den

