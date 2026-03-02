from __future__ import annotations

from typing import Dict, List
import numpy as np
import pandas as pd

from backtester import backtest_topk
from config import RunConfig
from metrics import objective_value
from walkforward import generate_walkforward_splits


def rolling_walkforward_eval(
    df: pd.DataFrame,
    cfg: RunConfig,
    weights: Dict[str, float],
    threshold_params: Dict[str, float],
) -> pd.DataFrame:
    w = cfg.walk_forward
    splits = generate_walkforward_splits(
        df[cfg.date_col], w["train_years"], w["test_years"], w["step_years"]
    )
    rows = []
    for tr_s, tr_e, te_s, te_e in splits:
        train_df = df[(df[cfg.date_col] >= tr_s) & (df[cfg.date_col] <= tr_e)]
        test_df = df[(df[cfg.date_col] >= te_s) & (df[cfg.date_col] <= te_e)]
        if train_df.empty or test_df.empty:
            continue
        train_bt = backtest_topk(train_df, cfg, weights, threshold_params)
        test_bt = backtest_topk(test_df, cfg, weights, threshold_params)
        rows.append(
            {
                "train_start": str(tr_s.date()),
                "train_end": str(tr_e.date()),
                "test_start": str(te_s.date()),
                "test_end": str(te_e.date()),
                "train_objective": objective_value(
                    train_bt.metrics, cfg.objective_name, cfg.objective_weights
                ),
                "test_objective": objective_value(
                    test_bt.metrics, cfg.objective_name, cfg.objective_weights
                ),
                "test_annual_return": test_bt.metrics["annual_return"],
                "test_max_drawdown": test_bt.metrics["max_drawdown"],
                "test_sharpe": test_bt.metrics["sharpe"],
            }
        )
    return pd.DataFrame(rows)


def regime_stability(df: pd.DataFrame, cfg: RunConfig, weights: Dict[str, float], params: Dict[str, float]) -> pd.DataFrame:
    # Regime proxy based on equal-weight market return rolling mean.
    base = df.groupby(cfg.date_col)[cfg.next_return_col].mean().sort_index()
    roll = base.rolling(20, min_periods=10).mean().dropna()
    if roll.empty:
        return pd.DataFrame()

    q_low, q_high = roll.quantile([0.33, 0.67]).tolist()
    regime = pd.Series(index=roll.index, dtype="object")
    regime[roll >= q_high] = "bull"
    regime[roll <= q_low] = "bear"
    regime[(roll > q_low) & (roll < q_high)] = "sideways"

    rows = []
    for tag in ["bull", "bear", "sideways"]:
        rg_dates = set(regime[regime == tag].index)
        subset = df[df[cfg.date_col].isin(rg_dates)]
        if subset.empty:
            continue
        bt = backtest_topk(subset, cfg, weights, params)
        rows.append(
            {
                "regime": tag,
                "annual_return": bt.metrics["annual_return"],
                "max_drawdown": bt.metrics["max_drawdown"],
                "sharpe": bt.metrics["sharpe"],
                "objective": objective_value(bt.metrics, cfg.objective_name, cfg.objective_weights),
            }
        )
    return pd.DataFrame(rows).sort_values("objective", ascending=False).reset_index(drop=True)


def parameter_sensitivity(
    df: pd.DataFrame,
    cfg: RunConfig,
    weights: Dict[str, float],
    best_params: Dict[str, float],
) -> pd.DataFrame:
    if not best_params:
        return pd.DataFrame()

    rows = []
    for k, v in best_params.items():
        delta = cfg.sensitivity_ranges.get(k, 0.1 * abs(v) if v != 0 else 0.01)
        tests = [v - delta, v, v + delta]
        for tv in tests:
            p = dict(best_params)
            p[k] = max(0.0, float(tv))
            bt = backtest_topk(df, cfg, weights, p)
            rows.append(
                {
                    "param": k,
                    "value": p[k],
                    "objective": objective_value(bt.metrics, cfg.objective_name, cfg.objective_weights),
                    "annual_return": bt.metrics["annual_return"],
                    "max_drawdown": bt.metrics["max_drawdown"],
                    "sharpe": bt.metrics["sharpe"],
                }
            )
    return pd.DataFrame(rows).sort_values(["param", "value"]).reset_index(drop=True)

