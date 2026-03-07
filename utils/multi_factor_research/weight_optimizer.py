from __future__ import annotations

from typing import Dict

import pandas as pd

from utils.multi_factor_research.factor_calculator import FACTOR_COLUMNS, PENALTY_COLUMNS


PENALTY_WEIGHTS = {
    "bearish_volume_penalty": 0.08,
    "break_trend_penalty": 0.10,
    "break_bull_bear_penalty": 0.12,
    "bull_bear_above_trend_penalty": 0.10,
    "bearish_candle_dominance_penalty": 0.06,
    "extreme_bull_run_penalty": 0.06,
    "high_overbought_penalty": 0.04,
    "abnormal_amplitude_penalty": 0.03,
    "volume_stagnation_penalty": 0.03,
    "key_k_close_break_penalty": 0.04,
    "key_k_low_break_penalty": 0.06,
}


def analyze_factor_contributions(dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame(
            columns=[
                "factor",
                "coverage",
                "active_samples",
                "mean_return_active",
                "mean_return_inactive",
                "return_lift",
                "win_rate_active",
                "win_rate_inactive",
                "success_lift",
                "return_contribution",
                "success_contribution",
                "combined_contribution",
            ]
        )

    baseline_return = float(dataset["return_pct"].mean())
    baseline_success = float(dataset["success"].mean())
    rows = []
    for factor in FACTOR_COLUMNS:
        active_mask = dataset[factor] > 0
        active = dataset.loc[active_mask]
        inactive = dataset.loc[~active_mask]
        if active.empty:
            rows.append(
                {
                    "factor": factor,
                    "coverage": 0.0,
                    "active_samples": 0,
                    "mean_return_active": 0.0,
                    "mean_return_inactive": baseline_return,
                    "return_lift": 0.0,
                    "win_rate_active": 0.0,
                    "win_rate_inactive": baseline_success,
                    "success_lift": 0.0,
                    "return_contribution": 0.0,
                    "success_contribution": 0.0,
                    "combined_contribution": 0.0,
                }
            )
            continue

        mean_return_active = float(active["return_pct"].mean())
        mean_return_inactive = float(inactive["return_pct"].mean()) if not inactive.empty else baseline_return
        win_rate_active = float(active["success"].mean())
        win_rate_inactive = float(inactive["success"].mean()) if not inactive.empty else baseline_success
        coverage = float(active_mask.mean())
        return_lift = mean_return_active - mean_return_inactive
        success_lift = win_rate_active - win_rate_inactive
        return_contribution = max(return_lift, 0.0) * coverage
        success_contribution = max(success_lift, 0.0) * coverage
        rows.append(
            {
                "factor": factor,
                "coverage": coverage,
                "active_samples": int(active_mask.sum()),
                "mean_return_active": mean_return_active,
                "mean_return_inactive": mean_return_inactive,
                "return_lift": return_lift,
                "win_rate_active": win_rate_active,
                "win_rate_inactive": win_rate_inactive,
                "success_lift": success_lift,
                "return_contribution": return_contribution,
                "success_contribution": success_contribution,
            }
        )

    contributions = pd.DataFrame(rows)
    return_total = contributions["return_contribution"].sum()
    success_total = contributions["success_contribution"].sum()
    contributions["return_weight"] = contributions["return_contribution"] / return_total if return_total > 0 else 1.0 / max(len(contributions), 1)
    contributions["success_weight"] = contributions["success_contribution"] / success_total if success_total > 0 else 1.0 / max(len(contributions), 1)
    contributions["combined_contribution"] = 0.5 * contributions["return_weight"] + 0.5 * contributions["success_weight"]
    return contributions.sort_values(
        ["combined_contribution", "return_contribution", "success_contribution"],
        ascending=False,
    ).reset_index(drop=True)


def derive_factor_weights(contributions: pd.DataFrame) -> Dict[str, float]:
    if contributions.empty:
        return {factor: 1.0 / len(FACTOR_COLUMNS) for factor in FACTOR_COLUMNS}
    weights = {}
    working = contributions.set_index("factor")
    total = float(working["combined_contribution"].sum())
    if total <= 0:
        return {factor: 1.0 / len(FACTOR_COLUMNS) for factor in FACTOR_COLUMNS}
    for factor in FACTOR_COLUMNS:
        weights[factor] = float(working["combined_contribution"].get(factor, 0.0) / total)
    return weights


def apply_weighted_score(dataset: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    if dataset.empty:
        return dataset.copy()
    out = dataset.copy()
    out["weighted_factor_score"] = 0.0
    for factor, weight in weights.items():
        out["weighted_factor_score"] += out[factor].fillna(0.0) * weight
    out["penalty_score"] = 0.0
    for factor in PENALTY_COLUMNS:
        out["penalty_score"] += out[factor].fillna(0.0) * PENALTY_WEIGHTS.get(factor, 0.0)
    out["net_factor_score"] = out["weighted_factor_score"] - out["penalty_score"]
    return out.sort_values(["signal_date", "net_factor_score", "code"], ascending=[True, False, True]).reset_index(drop=True)
