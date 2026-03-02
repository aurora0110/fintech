from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from config import RunConfig
from metrics import evaluate_returns


@dataclass
class BacktestResult:
    daily_returns: pd.Series
    metrics: Dict[str, float]
    trade_stats: Dict[str, float]


def apply_threshold_params(
    returns: pd.Series,
    params: Dict[str, float],
) -> pd.Series:
    # Basic risk modeling: cap profit/loss by configurable levels if supplied.
    out = returns.copy()
    if "fixed_stop_loss" in params:
        sl = abs(float(params["fixed_stop_loss"]))
        out = out.clip(lower=-sl)
    if "fixed_take_profit" in params:
        tp = abs(float(params["fixed_take_profit"]))
        out = out.clip(upper=tp)
    if "trend_deviation_take_profit" in params:
        tdtp = abs(float(params["trend_deviation_take_profit"]))
        out = out.clip(upper=tdtp)
    return out


def build_total_score(df: pd.DataFrame, factor_weights: Dict[str, float]) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    for f, w in factor_weights.items():
        col = f"{f}_score"
        if col in df.columns:
            score = score + w * df[col].fillna(0.0)
    return score


def backtest_topk(
    df: pd.DataFrame,
    cfg: RunConfig,
    factor_weights: Dict[str, float],
    threshold_params: Optional[Dict[str, float]] = None,
    market_filter_col: Optional[str] = None,
) -> BacktestResult:
    work = df.copy()
    work["total_score"] = build_total_score(work, factor_weights)
    work = work.dropna(subset=[cfg.next_return_col])
    work = work.sort_values([cfg.date_col, "total_score"], ascending=[True, False])

    prev_holding: Set[str] = set()
    daily_ret: List[float] = []
    turnovers: List[float] = []
    hit_days = 0
    active_days = 0

    for dt, g in work.groupby(cfg.date_col, sort=True):
        if market_filter_col and market_filter_col in g.columns:
            if g[market_filter_col].mean() <= 0:
                daily_ret.append(0.0)
                turnovers.append(0.0)
                prev_holding = set()
                continue

        n = len(g)
        if n == 0:
            continue
        active_days += 1
        k = max(cfg.min_holdings, int(np.ceil(n * cfg.top_k_pct)))
        sel = g.head(k)
        holding = set(sel[cfg.code_col].astype(str).tolist())
        raw = sel[cfg.next_return_col].astype(float)

        if threshold_params:
            raw = apply_threshold_params(raw, threshold_params)

        day_ret = float(raw.mean()) if len(raw) else 0.0
        turnover = 0.0
        if prev_holding:
            turnover = 1.0 - (len(prev_holding & holding) / max(len(prev_holding), 1))
        cost = turnover * (cfg.cost_rate + cfg.slippage_rate)
        net = day_ret - cost
        daily_ret.append(net)
        turnovers.append(turnover)
        if net > 0:
            hit_days += 1
        prev_holding = holding

    daily = pd.Series(daily_ret)
    metric = evaluate_returns(daily, risk_free_rate=cfg.risk_free_rate)
    trade_stats = {
        "active_days": float(active_days),
        "hit_days": float(hit_days),
        "hit_rate_days": float(hit_days / active_days) if active_days else 0.0,
        "avg_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
    }
    return BacktestResult(daily_returns=daily, metrics=metric, trade_stats=trade_stats)

