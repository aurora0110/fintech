from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

from utils.multi_factor_research.combo_search import summarize_best_combos
from utils.multi_factor_research.factor_calculator import FACTOR_COLUMNS, FACTOR_NAME_MAP


每因子最小分 = 5
每因子最大分 = 20
总分 = 100
分值步长 = 5

因子分组 = {
    "低位修复类": [
        "low_volume_pullback_factor",
        "price_amplitude_factor",
        "long_bear_short_volume_factor",
        "j_decline_acceleration_factor",
    ],
    "趋势动量类": [
        "staged_volume_burst_factor",
        "daily_ma_bull_factor",
        "macd_dif_factor",
        "rsi_bull_factor",
    ],
    "结构确认类": [
        "pullback_confirmation_factor",
        "key_k_support_factor",
        "price_sideways_10_factor",
        "price_sideways_factor",
    ],
}

分组约束 = {
    "低位修复类": (20, 40),
    "趋势动量类": (20, 40),
    "结构确认类": (20, 35),
}


def _objective(metrics: Dict[str, float]) -> float:
    return (
        metrics["avg_return"] * 0.35
        + metrics["quality_success_rate"] * 0.25
        + metrics["profit_loss_ratio"] * 0.20
        + metrics["take_profit_hit_rate"] * 0.10
        - metrics["trade_sequence_max_drawdown"] * 0.20
        - metrics["stop_loss_rate"] * 0.10
    )


def _calc_streak(mask: np.ndarray) -> int:
    max_streak = 0
    current = 0
    for flag in mask:
        if flag:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return int(max_streak)


def _summarize_masked_metrics(
    returns: np.ndarray,
    stop_loss_hit: np.ndarray,
    take_profit_hit: np.ndarray,
    holding_days: np.ndarray,
) -> Dict[str, float]:
    sample_count = int(returns.size)
    if sample_count == 0:
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

    positive = returns > 0
    quality_success = positive & (~stop_loss_hit)
    wins = returns[positive]
    losses = returns[returns < 0]
    downside = losses

    avg_return = float(returns.mean())
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(losses.mean()) if losses.size else 0.0
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss < 0 else 0.0
    std = float(returns.std(ddof=0)) if sample_count > 1 else 0.0
    downside_std = float(downside.std(ddof=0)) if downside.size > 1 else 0.0
    sharpe_trade = avg_return / std if std > 0 else 0.0
    sortino_trade = avg_return / downside_std if downside_std > 0 else 0.0

    safe_returns = np.clip(returns, -0.999999, None)
    log_equity = np.log1p(safe_returns).cumsum()
    running_max_log = np.maximum.accumulate(log_equity)
    drawdown = np.exp(log_equity - running_max_log) - 1.0

    return {
        "sample_count": sample_count,
        "positive_return_rate": float(positive.mean()),
        "quality_success_rate": float(quality_success.mean()),
        "take_profit_hit_rate": float(take_profit_hit.mean()),
        "stop_loss_rate": float(stop_loss_hit.mean()),
        "avg_return": avg_return,
        "median_return": float(np.median(returns)),
        "avg_win_return": avg_win,
        "avg_loss_return": avg_loss,
        "profit_loss_ratio": float(profit_loss_ratio),
        "expectancy": avg_return,
        "return_std": std,
        "downside_std": downside_std,
        "sharpe_trade": float(sharpe_trade),
        "sortino_trade": float(sortino_trade),
        "best_trade": float(returns.max()),
        "worst_trade": float(returns.min()),
        "trade_sequence_max_drawdown": float(abs(drawdown.min())) if drawdown.size else 0.0,
        "max_consecutive_wins": _calc_streak(positive),
        "max_consecutive_losses": _calc_streak(returns < 0),
        "avg_holding_days": float(holding_days.mean()),
        "median_holding_days": float(np.median(holding_days)),
    }


def _group_points(points: np.ndarray) -> Dict[str, int]:
    factor_to_index = {factor: idx for idx, factor in enumerate(FACTOR_COLUMNS)}
    result: Dict[str, int] = {}
    for group_name, members in 因子分组.items():
        result[group_name] = int(sum(points[factor_to_index[factor]] for factor in members))
    return result


def _valid_group_constraints(points: np.ndarray) -> bool:
    grouped = _group_points(points)
    for group_name, (lower, upper) in 分组约束.items():
        value = grouped[group_name]
        if value < lower or value > upper:
            return False
    return True


def _iter_constrained_point_vectors() -> Iterable[np.ndarray]:
    base_units = 每因子最小分 // 分值步长
    max_units = 每因子最大分 // 分值步长
    total_units = 总分 // 分值步长
    units_left = total_units - len(FACTOR_COLUMNS) * base_units

    extras = [0] * len(FACTOR_COLUMNS)

    def dfs(index: int, remaining: int) -> Iterable[np.ndarray]:
        if index == len(FACTOR_COLUMNS):
            if remaining == 0:
                points = np.array([(base_units + extra) * 分值步长 for extra in extras], dtype=int)
                if _valid_group_constraints(points):
                    yield points
            return

        max_extra = min(max_units - base_units, remaining)
        for extra in range(max_extra + 1):
            extras[index] = extra
            remaining_after = remaining - extra
            slots_left = len(FACTOR_COLUMNS) - index - 1
            if remaining_after < 0:
                continue
            if remaining_after > slots_left * (max_units - base_units):
                continue
            yield from dfs(index + 1, remaining_after)

    yield from dfs(0, units_left)


def _evaluate_points(
    points: np.ndarray,
    factor_matrix: np.ndarray,
    penalty_all: np.ndarray,
    returns_all: np.ndarray,
    stop_loss_all: np.ndarray,
    take_profit_all: np.ndarray,
    holding_days_all: np.ndarray,
    top_quantile: float,
    min_samples: int,
) -> dict | None:
    weights = points.astype(float) / 总分
    reward_scores = factor_matrix @ weights
    scores = reward_scores - penalty_all
    cutoff = float(np.quantile(scores, max(0.0, min(1.0, 1.0 - top_quantile))))
    mask = scores >= cutoff
    sample_count = int(mask.sum())
    if sample_count < min_samples:
        return None

    metrics = _summarize_masked_metrics(
        returns=returns_all[mask],
        stop_loss_hit=stop_loss_all[mask],
        take_profit_hit=take_profit_all[mask],
        holding_days=holding_days_all[mask],
    )
    grouped = _group_points(points)
    return {
        "weight_spec": "; ".join(
            f"{FACTOR_NAME_MAP.get(factor, factor)}={int(point)}分"
            for factor, point in zip(FACTOR_COLUMNS, points)
        ),
        "active_factor_count": int(np.count_nonzero(points)),
        "top_quantile": float(top_quantile),
        "samples": sample_count,
        "avg_reward_score": float(reward_scores[mask].mean()) if sample_count else 0.0,
        "avg_penalty_score": float(penalty_all[mask].mean()) if sample_count else 0.0,
        "avg_net_score": float(scores[mask].mean()) if sample_count else 0.0,
        "score": float(_objective(metrics)),
        **{f"weight_{factor}": int(point) for factor, point in zip(FACTOR_COLUMNS, points)},
        **{f"group_{name}": value for name, value in grouped.items()},
        **metrics,
        "search_stage": "受约束评分卡",
    }


def search_weighted_score_combinations(
    dataset: pd.DataFrame,
    top_quantile: float,
    min_samples: int = 30,
    weight_step: float = 0.05,
    initial_weights: Dict[str, float] | None = None,
) -> pd.DataFrame:
    del weight_step
    del initial_weights

    if dataset.empty:
        return pd.DataFrame()

    factor_matrix = dataset[FACTOR_COLUMNS].fillna(0.0).to_numpy(dtype=float)
    penalty_all = dataset.get("penalty_score", pd.Series(0.0, index=dataset.index)).to_numpy(dtype=float)
    returns_all = dataset["return_pct"].to_numpy(dtype=float)
    stop_loss_all = dataset["exit_reason"].eq("stop_loss").to_numpy(dtype=bool)
    take_profit_all = dataset["exit_reason"].eq("take_profit").to_numpy(dtype=bool)
    holding_days_all = (
        pd.to_datetime(dataset["exit_date"]).to_numpy(dtype="datetime64[D]")
        - pd.to_datetime(dataset["entry_date"]).to_numpy(dtype="datetime64[D]")
    ).astype(int)

    rows: List[dict] = []
    candidates = list(_iter_constrained_point_vectors())
    for points in tqdm(candidates, desc="受约束评分卡搜索", unit="combo"):
        row = _evaluate_points(
            points=points,
            factor_matrix=factor_matrix,
            penalty_all=penalty_all,
            returns_all=returns_all,
            stop_loss_all=stop_loss_all,
            take_profit_all=take_profit_all,
            holding_days_all=holding_days_all,
            top_quantile=top_quantile,
            min_samples=min_samples,
        )
        if row is not None:
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["score", "avg_return", "quality_success_rate", "profit_loss_ratio", "samples"],
        ascending=False,
    ).reset_index(drop=True)


def summarize_best_weighted_combos(weighted_df: pd.DataFrame) -> Dict[str, dict]:
    return summarize_best_combos(weighted_df.rename(columns={"weight_spec": "combo"}))
