from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

from utils.multi_factor_research.combo_search import summarize_best_combos
from utils.multi_factor_research.factor_calculator import FACTOR_COLUMNS, FACTOR_NAME_MAP


def _objective(metrics: Dict[str, float]) -> float:
    return (
        metrics["avg_return"] * 0.40
        + metrics["quality_success_rate"] * 0.25
        + metrics["profit_loss_ratio"] * 0.20
        + metrics["take_profit_hit_rate"] * 0.15
        - metrics["trade_sequence_max_drawdown"] * 0.25
        - metrics["stop_loss_rate"] * 0.15
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


def _weights_to_points(initial_weights: Dict[str, float]) -> np.ndarray:
    raw = np.array([max(initial_weights.get(factor, 0.0), 0.0) for factor in FACTOR_COLUMNS], dtype=float)
    if raw.sum() <= 0:
        raw = np.ones(len(FACTOR_COLUMNS), dtype=float)
    raw = raw / raw.sum() * 100.0
    points = np.floor(raw).astype(int)
    points = np.maximum(points, 1)
    diff = 100 - int(points.sum())
    frac = raw - np.floor(raw)

    while diff > 0:
        idx = int(np.argmax(frac))
        points[idx] += 1
        frac[idx] = -1.0
        diff -= 1

    while diff < 0:
        candidates = np.where(points > 1)[0]
        if candidates.size == 0:
            break
        idx = int(candidates[np.argmin(frac[candidates])])
        points[idx] -= 1
        frac[idx] = 2.0
        diff += 1

    return points


def _points_to_spec(points: np.ndarray) -> str:
    return "; ".join(
        f"{FACTOR_NAME_MAP.get(factor, factor)}={int(point)}分"
        for factor, point in zip(FACTOR_COLUMNS, points)
    )


def _move_points(points: np.ndarray, receiver: int, donor: int, amount: int) -> np.ndarray | None:
    if receiver == donor or amount <= 0:
        return None
    if points[donor] - amount < 1:
        return None
    moved = points.copy()
    moved[receiver] += amount
    moved[donor] -= amount
    return moved


def _build_coarse_candidates(initial_points: np.ndarray, coarse_step_points: int) -> List[np.ndarray]:
    seeds: List[np.ndarray] = []
    seen: set[Tuple[int, ...]] = set()

    def add(points: np.ndarray) -> None:
        key = tuple(int(v) for v in points)
        if key not in seen:
            seen.add(key)
            seeds.append(points.copy())

    add(initial_points)
    equal_points = np.full(len(FACTOR_COLUMNS), 100 // len(FACTOR_COLUMNS), dtype=int)
    equal_points[: 100 - int(equal_points.sum())] += 1
    add(equal_points)

    order_desc = np.argsort(-initial_points)
    order_asc = np.argsort(initial_points)
    for amount in (coarse_step_points, coarse_step_points * 2, coarse_step_points * 3):
        for receiver in order_desc[: min(6, len(order_desc))]:
            for donor in order_asc[: min(6, len(order_asc))]:
                moved = _move_points(initial_points, int(receiver), int(donor), amount)
                if moved is not None:
                    add(moved)

    for receiver in range(len(FACTOR_COLUMNS)):
        for donor in range(len(FACTOR_COLUMNS)):
            moved = _move_points(initial_points, receiver, donor, coarse_step_points)
            if moved is not None:
                add(moved)

    return seeds


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
    weights = points.astype(float) / 100.0
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
    row = {
        "weight_spec": _points_to_spec(points),
        "active_factor_count": int(np.count_nonzero(points)),
        "top_quantile": float(top_quantile),
        "samples": sample_count,
        "avg_reward_score": float(reward_scores[mask].mean()) if sample_count else 0.0,
        "avg_penalty_score": float(penalty_all[mask].mean()) if sample_count else 0.0,
        "avg_net_score": float(scores[mask].mean()) if sample_count else 0.0,
        "score": float(_objective(metrics)),
        **{f"weight_{factor}": int(point) for factor, point in zip(FACTOR_COLUMNS, points)},
        **metrics,
    }
    return row


def _refine_locally(
    seed_points: np.ndarray,
    factor_matrix: np.ndarray,
    penalty_all: np.ndarray,
    returns_all: np.ndarray,
    stop_loss_all: np.ndarray,
    take_profit_all: np.ndarray,
    holding_days_all: np.ndarray,
    top_quantile: float,
    min_samples: int,
    evaluation_cache: Dict[Tuple[int, ...], dict | None],
    max_rounds: int = 80,
) -> Tuple[np.ndarray, List[dict]]:
    history: List[dict] = []

    def evaluate(points: np.ndarray) -> dict | None:
        key = tuple(int(v) for v in points)
        if key not in evaluation_cache:
            evaluation_cache[key] = _evaluate_points(
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
        return evaluation_cache[key]

    current = seed_points.copy()
    current_row = evaluate(current)
    if current_row is not None:
        history.append(current_row)

    for _ in range(max_rounds):
        best_neighbor = None
        best_row = current_row
        for receiver in range(len(FACTOR_COLUMNS)):
            for donor in range(len(FACTOR_COLUMNS)):
                neighbor = _move_points(current, receiver, donor, 1)
                if neighbor is None:
                    continue
                row = evaluate(neighbor)
                if row is None:
                    continue
                if best_row is None or row["score"] > best_row["score"]:
                    best_neighbor = neighbor
                    best_row = row
        if best_neighbor is None or best_row is None or (current_row is not None and best_row["score"] <= current_row["score"]):
            break
        current = best_neighbor
        current_row = best_row
        history.append(best_row)

    return current, history


def search_weighted_score_combinations(
    dataset: pd.DataFrame,
    top_quantile: float,
    min_samples: int = 30,
    weight_step: float = 0.05,
    initial_weights: Dict[str, float] | None = None,
    coarse_top_k: int = 12,
) -> pd.DataFrame:
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

    base_weights = initial_weights or {factor: 1.0 / len(FACTOR_COLUMNS) for factor in FACTOR_COLUMNS}
    initial_points = _weights_to_points(base_weights)
    coarse_step_points = max(5, int(round(weight_step * 100)))
    coarse_candidates = _build_coarse_candidates(initial_points, coarse_step_points)

    evaluation_cache: Dict[Tuple[int, ...], dict | None] = {}
    coarse_rows: List[dict] = []
    for points in tqdm(coarse_candidates, desc="粗网格搜索", unit="combo"):
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
        evaluation_cache[tuple(int(v) for v in points)] = row
        if row is not None:
            row["search_stage"] = "粗网格"
            coarse_rows.append(row)

    coarse_df = pd.DataFrame(coarse_rows)
    if coarse_df.empty:
        return coarse_df

    top_seed_specs = (
        coarse_df.sort_values(["score", "avg_return", "quality_success_rate"], ascending=False)
        .head(coarse_top_k)
    )
    refined_rows: List[dict] = []
    for spec in tqdm(top_seed_specs["weight_spec"].tolist(), desc="局部细化", unit="seed"):
        row = top_seed_specs[top_seed_specs["weight_spec"] == spec].iloc[0]
        seed_points = np.array([int(row[f"weight_{factor}"]) for factor in FACTOR_COLUMNS], dtype=int)
        _, history = _refine_locally(
            seed_points=seed_points,
            factor_matrix=factor_matrix,
            penalty_all=penalty_all,
            returns_all=returns_all,
            stop_loss_all=stop_loss_all,
            take_profit_all=take_profit_all,
            holding_days_all=holding_days_all,
            top_quantile=top_quantile,
            min_samples=min_samples,
            evaluation_cache=evaluation_cache,
        )
        for item in history:
            enriched = item.copy()
            enriched["search_stage"] = "局部细化"
            refined_rows.append(enriched)

    all_rows = coarse_rows + refined_rows
    if not all_rows:
        return pd.DataFrame()

    result = pd.DataFrame(all_rows).drop_duplicates(subset=["weight_spec"]).sort_values(
        ["score", "avg_return", "quality_success_rate", "profit_loss_ratio", "samples"],
        ascending=False,
    )
    return result.reset_index(drop=True)


def summarize_best_weighted_combos(weighted_df: pd.DataFrame) -> Dict[str, dict]:
    return summarize_best_combos(weighted_df.rename(columns={"weight_spec": "combo"}))
