from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from core.data_loader import load_price_directory
from utils.multi_factor_research.run_weighted_portfolio_backtest import (
    PortfolioConfig,
    _load_refined_weights,
    _parse_weight_spec,
    _prepare_stock_frames,
    _run_model,
)


def _run_single(
    prepared: dict,
    all_dates: list,
    label: str,
    weights: dict[str, float],
    initial_capital: float,
    replacement_threshold: float,
    max_positions: int,
    buy_mode: str,
) -> dict:
    config = PortfolioConfig(
        initial_capital=initial_capital,
        max_positions=max_positions,
        replacement_threshold=replacement_threshold,
        min_hold_days_for_replace=5,
        max_daily_replacements=1,
        buy_mode=buy_mode,
    )
    result = _run_model(label, prepared, all_dates, weights, config)
    metrics = result["metrics"]
    return {
        "策略": label,
        "换仓阈值": replacement_threshold,
        "最大持仓数": max_positions,
        "买入方式": "一次性买入" if buy_mode == "full" else "分批买入",
        "最终资金": result["final_equity"],
        "资金倍数": metrics["final_multiple"],
        "年化收益": metrics["annual_return"],
        "最大回撤": metrics["max_drawdown"],
        "夏普比率": metrics["sharpe"],
        "波动率": metrics["volatility"],
        "交易次数": result["trade_count"],
    }


def main() -> None:
    root = Path("/Users/lidongyang/Desktop/Qstrategy")
    refined_root = root / "results/multi_factor_research_v18_constrained_card"
    output_root = root / "results/portfolio_matrix_v1"
    output_root.mkdir(parents=True, exist_ok=True)

    weights_payload = _load_refined_weights(refined_root)
    weights_map = {
        "fixed_take_profit": _parse_weight_spec(weights_payload["fixed_take_profit"]["combo"]),
        "fixed_days": _parse_weight_spec(weights_payload["fixed_days"]["combo"]),
    }

    stock_data, all_dates = load_price_directory(str(root / "data/20260226"))
    prepared = _prepare_stock_frames(stock_data)

    stage1_rows: list[dict] = []
    stage1_tasks = []
    for label in ["fixed_take_profit", "fixed_days"]:
        for replacement_threshold in [0.03, 0.05, 0.08]:
            for buy_mode in ["staged", "full"]:
                stage1_tasks.append((label, replacement_threshold, buy_mode))

    for label, replacement_threshold, buy_mode in tqdm(stage1_tasks, desc="第一阶段组合实验", unit="组"):
        row = _run_single(
            prepared,
            all_dates,
            label,
            weights_map[label],
            10_000_000.0,
            replacement_threshold,
            10,
            buy_mode,
        )
        stage1_rows.append(row)

    stage1_path = output_root / "第一阶段结果.json"
    stage1_path.write_text(json.dumps(stage1_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    best_params: dict[str, tuple[float, str]] = {}
    for label in ["fixed_take_profit", "fixed_days"]:
        candidates = [row for row in stage1_rows if row["策略"] == label]
        best = max(candidates, key=lambda row: (row["年化收益"], -row["最大回撤"], row["夏普比率"]))
        best_params[label] = (best["换仓阈值"], "full" if best["买入方式"] == "一次性买入" else "staged")

    stage2_rows: list[dict] = []
    stage2_tasks = []
    for label in ["fixed_take_profit", "fixed_days"]:
        replacement_threshold, buy_mode = best_params[label]
        for max_positions in [5, 10, 15]:
            stage2_tasks.append((label, replacement_threshold, buy_mode, max_positions))

    for label, replacement_threshold, buy_mode, max_positions in tqdm(stage2_tasks, desc="第二阶段组合实验", unit="组"):
        row = _run_single(
            prepared,
            all_dates,
            label,
            weights_map[label],
            10_000_000.0,
            replacement_threshold,
            max_positions,
            buy_mode,
        )
        stage2_rows.append(row)

    stage2_path = output_root / "第二阶段结果.json"
    stage2_path.write_text(json.dumps(stage2_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    final_selection = {}
    for label in ["fixed_take_profit", "fixed_days"]:
        candidates = [row for row in stage2_rows if row["策略"] == label]
        final_selection[label] = max(candidates, key=lambda row: (row["年化收益"], -row["最大回撤"], row["夏普比率"]))

    summary = {
        "第一阶段最优参数": {
            label: {
                "换仓阈值": params[0],
                "买入方式": "一次性买入" if params[1] == "full" else "分批买入",
            }
            for label, params in best_params.items()
        },
        "第二阶段最优结果": final_selection,
    }
    (output_root / "汇总结果.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
