from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd


def max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    peak = equity_curve.cummax()
    dd = equity_curve / peak - 1.0
    return float(dd.min())


def annual_return(daily_returns: pd.Series) -> float:
    if daily_returns.empty:
        return 0.0
    n = len(daily_returns)
    total = float((1.0 + daily_returns).prod())
    if total <= 0:
        return -1.0
    return total ** (252.0 / n) - 1.0


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    if daily_returns.empty:
        return 0.0
    excess = daily_returns - (risk_free_rate / 252.0)
    vol = float(excess.std(ddof=0))
    if vol == 0:
        return 0.0
    return float(excess.mean() / vol * np.sqrt(252.0))


def calmar_ratio(ann_ret: float, mdd: float) -> float:
    den = abs(mdd)
    return 0.0 if den == 0 else float(ann_ret / den)


def profit_loss_ratio(daily_returns: pd.Series) -> float:
    if daily_returns.empty:
        return 0.0
    pos = daily_returns[daily_returns > 0]
    neg = daily_returns[daily_returns < 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    return float(pos.mean() / abs(neg.mean()))


def evaluate_returns(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> Dict[str, float]:
    if daily_returns.empty:
        return {
            "annual_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "calmar": 0.0,
            "win_rate": 0.0,
            "profit_loss_ratio": 0.0,
            "volatility": 0.0,
            "days": 0,
        }

    curve = (1.0 + daily_returns).cumprod()
    ann = annual_return(daily_returns)
    mdd = max_drawdown(curve)
    sharpe = sharpe_ratio(daily_returns, risk_free_rate=risk_free_rate)
    calmar = calmar_ratio(ann, mdd)

    return {
        "annual_return": float(ann),
        "max_drawdown": float(abs(mdd)),
        "sharpe": float(sharpe),
        "calmar": float(calmar),
        "win_rate": float((daily_returns > 0).mean()),
        "profit_loss_ratio": float(profit_loss_ratio(daily_returns)),
        "volatility": float(daily_returns.std(ddof=0) * np.sqrt(252.0)),
        "days": int(len(daily_returns)),
    }


def objective_value(metric: Dict[str, float], objective_name: str, objective_weights: Dict[str, float]) -> float:
    if objective_name == "return_dd":
        mdd = metric.get("max_drawdown", 0.0)
        if mdd == 0:
            return metric.get("annual_return", 0.0)
        return metric.get("annual_return", 0.0) / mdd

    # composite: Sharpe + 0.5*Calmar - 0.3*MaxDrawdown
    sharpe_w = objective_weights.get("sharpe", 1.0)
    calmar_w = objective_weights.get("calmar", 0.5)
    mdd_w = objective_weights.get("max_drawdown", -0.3)
    return (
        sharpe_w * metric.get("sharpe", 0.0)
        + calmar_w * metric.get("calmar", 0.0)
        + mdd_w * metric.get("max_drawdown", 0.0)
    )

