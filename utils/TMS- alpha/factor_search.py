from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from backtester import backtest_topk
from config import FactorConfig, RunConfig
from metrics import objective_value
from walkforward import generate_walkforward_splits


def _iter_param_grid(threshold_params: Dict[str, Dict[str, Any]]) -> List[Dict[str, float]]:
    if not threshold_params:
        return [{}]
    keys = []
    values = []
    for k, info in threshold_params.items():
        grid = info.get("grid", [])
        if not grid:
            continue
        keys.append(k)
        values.append(grid)
    if not keys:
        return [{}]
    out = []
    for comb in product(*values):
        out.append({k: float(v) for k, v in zip(keys, comb)})
    return out


def _objective_of_subset(
    subset: pd.DataFrame,
    cfg: RunConfig,
    factor_weights: Dict[str, float],
    params: Dict[str, float],
) -> float:
    result = backtest_topk(subset, cfg, factor_weights=factor_weights, threshold_params=params)
    return objective_value(result.metrics, cfg.objective_name, cfg.objective_weights)


def search_threshold_params_walkforward(
    df: pd.DataFrame,
    cfg: RunConfig,
    factor_weights: Dict[str, float],
) -> Tuple[Dict[str, float], pd.DataFrame]:
    param_grid = _iter_param_grid(cfg.threshold_params)
    if param_grid == [{}]:
        return {}, pd.DataFrame([{"params": {}, "objective_mean": 0.0}])

    w = cfg.walk_forward
    splits = generate_walkforward_splits(
        df[cfg.date_col],
        train_years=w["train_years"],
        test_years=w["test_years"],
        step_years=w["step_years"],
    )
    if not splits:
        splits = [(df[cfg.date_col].min(), df[cfg.date_col].max(), df[cfg.date_col].min(), df[cfg.date_col].max())]

    rows = []
    for p in param_grid:
        objs = []
        for _, _, te_s, te_e in splits:
            te = df[(df[cfg.date_col] >= te_s) & (df[cfg.date_col] <= te_e)]
            if te.empty:
                continue
            obj = _objective_of_subset(te, cfg, factor_weights, p)
            objs.append(obj)
        mean_obj = float(np.mean(objs)) if objs else -1e9
        rows.append({"params": p, "objective_mean": mean_obj})

    res = pd.DataFrame(rows).sort_values("objective_mean", ascending=False).reset_index(drop=True)
    best = dict(res.iloc[0]["params"]) if not res.empty else {}
    return best, res


def search_single_factor_threshold(
    df: pd.DataFrame,
    cfg: RunConfig,
    factor: FactorConfig,
    thresholds: List[float],
) -> pd.DataFrame:
    # Single-factor threshold test for J/RSI-like variables.
    if factor.name not in df.columns:
        return pd.DataFrame()

    rows = []
    tmp = df.copy()
    for th in thresholds:
        if factor.comparator == "<=":
            signal = tmp[factor.name] <= th
        elif factor.comparator == ">=":
            signal = tmp[factor.name] >= th
        else:
            signal = tmp[factor.name] >= th

        local = tmp[signal].copy()
        if local.empty:
            continue
        local[f"{factor.name}_score"] = 1.0
        result = backtest_topk(
            local,
            cfg,
            factor_weights={factor.name: 1.0},
            threshold_params={},
        )
        m = result.metrics
        rows.append(
            {
                "factor": factor.name,
                "threshold": th,
                "annual_return": m["annual_return"],
                "win_rate": m["win_rate"],
                "profit_loss_ratio": m["profit_loss_ratio"],
                "max_drawdown": m["max_drawdown"],
                "sharpe": m["sharpe"],
                "calmar": m["calmar"],
                "objective": objective_value(m, cfg.objective_name, cfg.objective_weights),
            }
        )

    return pd.DataFrame(rows).sort_values("objective", ascending=False).reset_index(drop=True)

