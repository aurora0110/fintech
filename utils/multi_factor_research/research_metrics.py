from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _calc_streak(mask: pd.Series) -> int:
    max_streak = 0
    current = 0
    for flag in mask.fillna(False).astype(bool):
        if flag:
            current += 1
            if current > max_streak:
                max_streak = current
        else:
            current = 0
    return int(max_streak)


def summarize_trade_metrics(dataset: pd.DataFrame) -> Dict[str, float]:
    if dataset.empty:
        return {
            "sample_count": 0,
            "positive_return_rate": 0.0,
            "quality_success_rate": 0.0,
            "take_profit_hit_rate": 0.0,
            "stop_loss_rate": 0.0,
            "avg_return": 0.0,
            "median_return": 0.0,
            "avg_win_return": 0.0,
            "avg_loss_return": 0.0,
            "profit_loss_ratio": 0.0,
            "expectancy": 0.0,
            "return_std": 0.0,
            "downside_std": 0.0,
            "sharpe_trade": 0.0,
            "sortino_trade": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "trade_sequence_max_drawdown": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "avg_holding_days": 0.0,
            "median_holding_days": 0.0,
        }

    entry_dates = pd.to_datetime(dataset["entry_date"])
    exit_dates = pd.to_datetime(dataset["exit_date"])
    holding_days_realized = (exit_dates - entry_dates).dt.days.clip(lower=0)
    exit_reason = dataset["exit_reason"]
    returns = dataset["return_pct"].astype(float)
    positive_mask = returns.gt(0)
    quality_success = positive_mask & exit_reason.ne("stop_loss")
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    downside = returns[returns < 0]

    avg_return = float(returns.mean())
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss < 0 else 0.0
    expectancy = float(avg_return)
    std = float(returns.std(ddof=0)) if len(returns) > 1 else 0.0
    downside_std = float(downside.std(ddof=0)) if len(downside) > 1 else 0.0
    sharpe_trade = avg_return / std if std > 0 else 0.0
    sortino_trade = avg_return / downside_std if downside_std > 0 else 0.0

    # Use cumulative log returns to avoid overflow on long trade sequences.
    safe_returns = returns.clip(lower=-0.999999)
    log_equity = np.log1p(safe_returns).cumsum()
    running_max_log = np.maximum.accumulate(log_equity.to_numpy())
    drawdown = np.exp(log_equity.to_numpy() - running_max_log) - 1.0

    return {
        "sample_count": int(len(dataset)),
        "positive_return_rate": float(positive_mask.mean()),
        "quality_success_rate": float(quality_success.mean()),
        "take_profit_hit_rate": float(exit_reason.eq("take_profit").mean()),
        "stop_loss_rate": float(exit_reason.eq("stop_loss").mean()),
        "avg_return": avg_return,
        "median_return": float(returns.median()),
        "avg_win_return": avg_win,
        "avg_loss_return": avg_loss,
        "profit_loss_ratio": float(profit_loss_ratio),
        "expectancy": expectancy,
        "return_std": std,
        "downside_std": downside_std,
        "sharpe_trade": float(sharpe_trade),
        "sortino_trade": float(sortino_trade),
        "best_trade": float(returns.max()),
        "worst_trade": float(returns.min()),
        "trade_sequence_max_drawdown": float(abs(drawdown.min())) if drawdown.size else 0.0,
        "max_consecutive_wins": _calc_streak(positive_mask),
        "max_consecutive_losses": _calc_streak(returns.lt(0)),
        "avg_holding_days": float(holding_days_realized.mean()),
        "median_holding_days": float(holding_days_realized.median()),
    }
