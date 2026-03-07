from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from utils.multi_factor_research.combo_search import search_factor_combinations, summarize_best_combos
from utils.multi_factor_research.data_processor import load_stock_directory
from utils.multi_factor_research.exit_models import (
    FixedDaysExit,
    FixedTakeProfitExit,
    TieredExit,
    exit_config_label,
    simulate_signal_exits,
)
from utils.multi_factor_research.factor_calculator import (
    FACTOR_NAME_MAP,
    ResearchConfig,
    build_prepared_stock_data,
    build_signal_candidates,
)
from utils.multi_factor_research.research_metrics import summarize_trade_metrics
from utils.multi_factor_research.weighted_combo_search import (
    search_weighted_score_combinations,
    summarize_best_weighted_combos,
)
from utils.multi_factor_research.weight_optimizer import (
    PENALTY_WEIGHTS,
    analyze_factor_contributions,
    apply_weighted_score,
    derive_factor_weights,
)


def summarize_score_groups(dataset: pd.DataFrame, top_quantile: float) -> dict:
    if dataset.empty:
        return {
            "samples": 0,
            "baseline_avg_return": 0.0,
            "baseline_positive_return_rate": 0.0,
            "baseline_quality_success_rate": 0.0,
            "baseline_take_profit_hit_rate": 0.0,
            "baseline_stop_loss_rate": 0.0,
            "top_group_avg_return": 0.0,
            "top_group_positive_return_rate": 0.0,
            "top_group_quality_success_rate": 0.0,
            "top_group_take_profit_hit_rate": 0.0,
            "top_group_stop_loss_rate": 0.0,
            "top_group_samples": 0,
        }

    cutoff = dataset["net_factor_score"].quantile(max(0.0, min(1.0, 1.0 - top_quantile)))
    top_group = dataset[dataset["net_factor_score"] >= cutoff].copy()
    base_metrics = summarize_trade_metrics(dataset)
    top_metrics = summarize_trade_metrics(top_group)
    return {
        "samples": int(len(dataset)),
        "baseline_avg_return": base_metrics["avg_return"],
        "baseline_positive_return_rate": base_metrics["positive_return_rate"],
        "baseline_quality_success_rate": base_metrics["quality_success_rate"],
        "baseline_take_profit_hit_rate": base_metrics["take_profit_hit_rate"],
        "baseline_stop_loss_rate": base_metrics["stop_loss_rate"],
        "top_group_avg_return": top_metrics["avg_return"],
        "top_group_positive_return_rate": top_metrics["positive_return_rate"],
        "top_group_quality_success_rate": top_metrics["quality_success_rate"],
        "top_group_take_profit_hit_rate": top_metrics["take_profit_hit_rate"],
        "top_group_stop_loss_rate": top_metrics["stop_loss_rate"],
        "top_group_samples": int(len(top_group)),
    }


def _add_chinese_factor_names(contributions: pd.DataFrame) -> pd.DataFrame:
    out = contributions.copy()
    out["factor_cn"] = out["factor"].map(FACTOR_NAME_MAP).fillna(out["factor"])
    return out


def _evaluate_exit_config(
    prepared_stock_data: dict[str, pd.DataFrame],
    signal_candidates: pd.DataFrame,
    exit_config: object,
    config: ResearchConfig,
    combo_min_samples: int,
    weighted_min_samples: int,
    weighted_search_step: float,
    output_dir: Path,
) -> dict:
    label = exit_config_label(exit_config)
    dataset = simulate_signal_exits(
        prepared_stock_data=prepared_stock_data,
        signal_candidates=signal_candidates,
        exit_config=exit_config,
        success_return_threshold=config.success_return_threshold,
    )
    contributions = _add_chinese_factor_names(analyze_factor_contributions(dataset))
    weights = derive_factor_weights(contributions)
    scored = apply_weighted_score(dataset, weights)
    combo_df = search_factor_combinations(scored, min_samples=combo_min_samples)
    best_combos = summarize_best_combos(combo_df)
    weighted_combo_df = search_weighted_score_combinations(
        scored,
        top_quantile=config.top_quantile,
        min_samples=weighted_min_samples,
        weight_step=weighted_search_step,
        initial_weights=weights,
    )
    best_weighted_combos = summarize_best_weighted_combos(weighted_combo_df)
    overall_metrics = summarize_trade_metrics(scored)
    score_summary = summarize_score_groups(scored, config.top_quantile)

    run_dir = output_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)
    contributions = contributions.copy()
    contributions["weight"] = contributions["factor"].map(weights).fillna(0.0)
    contributions.to_csv(run_dir / "factor_contributions.csv", index=False, encoding="utf-8-sig")
    scored.to_csv(run_dir / "signal_scores.csv", index=False, encoding="utf-8-sig")
    if not combo_df.empty:
        combo_df.to_csv(run_dir / "factor_combo_ranking.csv", index=False, encoding="utf-8-sig")
    if not weighted_combo_df.empty:
        weighted_combo_df.to_csv(run_dir / "weighted_score_ranking.csv", index=False, encoding="utf-8-sig")

    payload = {
        "exit_model": label,
        "sample_count": int(len(scored)),
        "factor_weights": {FACTOR_NAME_MAP.get(k, k): v for k, v in weights.items()},
        "penalty_weights": {FACTOR_NAME_MAP.get(k, k): v for k, v in PENALTY_WEIGHTS.items()},
        "overall_metrics": overall_metrics,
        "score_summary": score_summary,
        "best_combos": best_combos,
        "best_weighted_combos": best_weighted_combos,
        "factor_ranking": contributions[
            [
                "factor_cn",
                "factor",
                "weight",
                "coverage",
                "active_samples",
                "mean_return_active",
                "return_lift",
                "win_rate_active",
                "success_lift",
                "return_contribution",
                "success_contribution",
                "combined_contribution",
            ]
        ].to_dict(orient="records"),
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "label": label,
        "sample_count": int(len(scored)),
        "overall_metrics": overall_metrics,
        "best_combos": best_combos,
        "best_weighted_combos": best_weighted_combos,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-exit factor scoring research")
    parser.add_argument("data_dir")
    parser.add_argument("--success-threshold", type=float, default=0.0)
    parser.add_argument("--burst-window", type=int, default=20)
    parser.add_argument("--top-quantile", type=float, default=0.30)
    parser.add_argument("--combo-min-samples", type=int, default=30)
    parser.add_argument("--weighted-min-samples", type=int, default=30)
    parser.add_argument("--weighted-search-step", type=float, default=0.1)
    parser.add_argument("--output-dir", default="results/multi_factor_research")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ResearchConfig(
        success_return_threshold=args.success_threshold,
        burst_window=args.burst_window,
        top_quantile=args.top_quantile,
    )

    stock_data = load_stock_directory(args.data_dir)
    prepared_stock_data = build_prepared_stock_data(stock_data, burst_window=config.burst_window)
    signal_candidates = build_signal_candidates(prepared_stock_data)
    signal_candidates.to_csv(output_dir / "base_signal_candidates.csv", index=False, encoding="utf-8-sig")

    exit_configs: list[object] = [
        FixedTakeProfitExit(0.10),
        FixedTakeProfitExit(0.20),
        FixedTakeProfitExit(0.30),
        FixedTakeProfitExit(0.40),
        FixedTakeProfitExit(0.50),
        FixedDaysExit(5),
        FixedDaysExit(10),
        FixedDaysExit(20),
        FixedDaysExit(30),
        TieredExit(60),
    ]

    summaries = []
    for exit_config in exit_configs:
        summaries.append(
            _evaluate_exit_config(
                prepared_stock_data=prepared_stock_data,
                signal_candidates=signal_candidates,
                exit_config=exit_config,
                config=config,
                combo_min_samples=args.combo_min_samples,
                weighted_min_samples=args.weighted_min_samples,
                weighted_search_step=args.weighted_search_step,
                output_dir=output_dir,
            )
        )

    fixed_take_profit_runs = [item for item in summaries if item["label"].startswith("固定涨幅止盈")]
    fixed_days_runs = [item for item in summaries if item["label"].startswith("固定持有")]
    tiered_runs = [item for item in summaries if item["label"] == "分批顺序止盈"]

    final_summary = {
        "base_signal_count": int(len(signal_candidates)),
        "fixed_take_profit_best_by_score": max(
            fixed_take_profit_runs,
            key=lambda item: item["best_weighted_combos"].get("best_by_score", {}).get("score", float("-inf")),
        ) if fixed_take_profit_runs else {},
        "fixed_days_best_by_score": max(
            fixed_days_runs,
            key=lambda item: item["best_weighted_combos"].get("best_by_score", {}).get("score", float("-inf")),
        ) if fixed_days_runs else {},
        "tiered_best": tiered_runs[0] if tiered_runs else {},
        "tiered_best_weighted_by_score": tiered_runs[0].get("best_weighted_combos", {}).get("best_by_score", {}) if tiered_runs else {},
        "all_runs": summaries,
    }

    (output_dir / "all_exit_models_summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
