from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_metrics(equity_curve: pd.Series) -> Dict[str, float]:
    if equity_curve.empty:
        return {
            "final_multiple": 1.0,
            "annual_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "volatility": 0.0,
            "days": 0,
        }

    returns = equity_curve.pct_change().dropna()
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    total = float(equity_curve.iloc[-1] / equity_curve.iloc[0])
    days = len(equity_curve)
    years = days / 252.0 if days > 0 else 0.0
    annual_return = total ** (1.0 / years) - 1.0 if years > 0 and total > 0 else 0.0

    if returns.empty or returns.std(ddof=0) == 0:
        sharpe = 0.0
        volatility = 0.0
    else:
        sharpe = float(returns.mean() / returns.std(ddof=0) * np.sqrt(252.0))
        volatility = float(returns.std(ddof=0) * np.sqrt(252.0))

    return {
        "final_multiple": total,
        "annual_return": float(annual_return),
        "max_drawdown": float(abs(drawdown.min())),
        "sharpe": sharpe,
        "volatility": volatility,
        "days": int(days),
    }
