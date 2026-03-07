from __future__ import annotations

from itertools import combinations
from typing import Dict, List

import pandas as pd

from utils.multi_factor_research.factor_calculator import FACTOR_COLUMNS
from utils.multi_factor_research.research_metrics import summarize_trade_metrics


def _combo_name(combo: tuple[str, ...]) -> str:
    return " + ".join(combo)


def _combo_mask(dataset: pd.DataFrame, combo: tuple[str, ...]) -> pd.Series:
    mask = pd.Series(True, index=dataset.index)
    for factor in combo:
        mask &= dataset[factor] > 0
    return mask


def search_factor_combinations(dataset: pd.DataFrame, min_samples: int = 30) -> pd.DataFrame:
    rows: List[dict] = []
    if dataset.empty:
        return pd.DataFrame()

    for size in range(1, len(FACTOR_COLUMNS) + 1):
        for combo in combinations(FACTOR_COLUMNS, size):
            mask = _combo_mask(dataset, combo)
            subset = dataset.loc[mask].copy()
            if len(subset) < min_samples:
                continue
            metrics = summarize_trade_metrics(subset)
            score = (
                metrics["avg_return"] * 0.40
                + metrics["quality_success_rate"] * 0.25
                + metrics["profit_loss_ratio"] * 0.20
                + metrics["take_profit_hit_rate"] * 0.15
                - metrics["trade_sequence_max_drawdown"] * 0.25
                - metrics["stop_loss_rate"] * 0.15
            )
            rows.append(
                {
                    "combo": _combo_name(combo),
                    "factor_count": size,
                    "samples": int(len(subset)),
                    "score": float(score),
                    **metrics,
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["score", "avg_return", "quality_success_rate", "profit_loss_ratio"],
        ascending=False,
    ).reset_index(drop=True)


def summarize_best_combos(combo_df: pd.DataFrame) -> Dict[str, dict]:
    if combo_df.empty:
        return {
            "best_by_score": {},
            "best_by_avg_return": {},
            "best_by_quality_success_rate": {},
            "best_by_profit_loss_ratio": {},
        }

    def pick_best(column: str) -> dict:
        row = combo_df.sort_values([column, "samples"], ascending=[False, False]).iloc[0]
        return row.to_dict()

    return {
        "best_by_score": pick_best("score"),
        "best_by_avg_return": pick_best("avg_return"),
        "best_by_quality_success_rate": pick_best("quality_success_rate"),
        "best_by_profit_loss_ratio": pick_best("profit_loss_ratio"),
    }
