from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from core.data_loader import load_price_directory
from utils.multi_factor_research.run_weighted_portfolio_backtest import (
    PortfolioConfig,
    _load_formal_scorecard,
    _prepare_stock_frames,
    _run_model,
)


def _综合评分(row: dict) -> float:
    return (
        float(row["年化收益"]) * 100.0
        - float(row["最大回撤"]) * 35.0
        + float(row["夏普比率"]) * 10.0
    )


def _write_status(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _非零权重中位数(add_weights: dict, penalty_weights: dict) -> float:
    values = [abs(float(v)) for v in list(add_weights.values()) + list(penalty_weights.values()) if abs(float(v)) > 0]
    if not values:
        return 1.0
    return max(1.0, float(statistics.median(values)))


def _load_best_profiles(summary_path: Path) -> dict[str, dict]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "fixed_take_profit": payload["固定止盈"]["收益最优"]["参数"],
        "fixed_days": payload["固定持有"]["收益最优"]["参数"],
        "tiered": payload["分批止盈"]["收益最优"]["参数"],
    }


def _run_single(
    prepared,
    all_dates,
    strategy_label: str,
    add_weights: dict,
    penalty_weights: dict,
    exit_profile: dict,
    initial_capital: float,
    max_positions: int,
    buy_mode: str,
    replacement_threshold: float,
    min_hold_days_for_replace: int,
    max_daily_replacements: int,
    use_trend_start_pool: bool,
    min_confirmation_hits: int,
    min_support_hits: int,
    sideways_mode: str,
    sideways_filter_threshold: float,
    sideways_score_penalty_scale: float,
) -> dict:
    config = PortfolioConfig(
        initial_capital=initial_capital,
        max_positions=max_positions,
        replacement_threshold=replacement_threshold,
        min_hold_days_for_replace=min_hold_days_for_replace,
        max_daily_replacements=max_daily_replacements,
        buy_mode=buy_mode,
        exit_profile=exit_profile,
        use_trend_start_pool=use_trend_start_pool,
        min_confirmation_hits=min_confirmation_hits,
        min_support_hits=min_support_hits,
        sideways_mode=sideways_mode,
        sideways_filter_threshold=sideways_filter_threshold,
        sideways_score_penalty_scale=sideways_score_penalty_scale,
    )
    result = _run_model(strategy_label, prepared, all_dates, add_weights, penalty_weights, config)
    metrics = result["metrics"]
    row = {
        "策略": strategy_label,
        "横盘模式": sideways_mode,
        "横盘过滤阈值": sideways_filter_threshold,
        "横盘降分系数": sideways_score_penalty_scale,
        "参数": exit_profile,
        "年化收益": float(metrics["annual_return"]),
        "最大回撤": float(metrics["max_drawdown"]),
        "夏普比率": float(metrics["sharpe"]),
        "波动率": float(metrics["volatility"]),
        "资金倍数": float(metrics["final_multiple"]),
        "最终资金": float(result["final_equity"]),
        "交易次数": int(result["trade_count"]),
        "完整轮次交易数": int(result["完整轮次交易数"]),
        "平均持有期间收益率": float(result["平均持有期间收益率"]),
        "盈利轮次占比": float(result["盈利轮次占比"]),
        "盈利轮次平均收益率": float(result["盈利轮次平均收益率"]),
        "亏损轮次平均收益率": float(result["亏损轮次平均收益率"]),
    }
    row["综合评分"] = _综合评分(row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="横盘/无效震荡专项实验")
    parser.add_argument("--data-dir", default="/Users/lidongyang/Desktop/Qstrategy/data/20260226")
    parser.add_argument("--weighted-scorecard-root", default="/Users/lidongyang/Desktop/Qstrategy/results/formal_scorecard_trend_v2_pool_weighted")
    parser.add_argument("--binary-scorecard-root", default="/Users/lidongyang/Desktop/Qstrategy/results/formal_scorecard_trend_v2_pool_binary")
    parser.add_argument("--weighted-summary", default="/Users/lidongyang/Desktop/Qstrategy/results/final_trend_matrix_weighted/汇总结果.json")
    parser.add_argument("--binary-summary", default="/Users/lidongyang/Desktop/Qstrategy/results/final_trend_matrix_binary/汇总结果.json")
    parser.add_argument("--output-root", default="/Users/lidongyang/Desktop/Qstrategy/results/sideways_special_experiment_v1")
    parser.add_argument("--initial-capital", type=float, default=10_000_000.0)
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--buy-mode", choices=["staged", "full", "strict_full"], default="full")
    parser.add_argument("--replacement-threshold", type=float, default=0.03)
    parser.add_argument("--min-hold-days-for-replace", type=int, default=5)
    parser.add_argument("--max-daily-replacements", type=int, default=1)
    parser.add_argument("--use-trend-start-pool", dest="use_trend_start_pool", action="store_true")
    parser.add_argument("--no-trend-start-pool", dest="use_trend_start_pool", action="store_false")
    parser.add_argument("--min-confirmation-hits", type=int, default=1)
    parser.add_argument("--min-support-hits", type=int, default=1)
    parser.add_argument("--sideways-filter-threshold", type=float, default=0.55)
    parser.set_defaults(use_trend_start_pool=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "状态.json"
    _write_status(status_path, {"阶段": "加载行情数据", "完成组数": 0})

    stock_data, all_dates = load_price_directory(args.data_dir)
    _write_status(status_path, {"阶段": "预处理因子", "股票数": len(stock_data), "完成组数": 0})
    prepared = _prepare_stock_frames(stock_data)
    _write_status(status_path, {"阶段": "准备实验任务", "股票数": len(stock_data), "完成组数": 0})

    methods = {
        "加权法": {
            "scorecard": _load_formal_scorecard(Path(args.weighted_scorecard_root)),
            "profiles": _load_best_profiles(Path(args.weighted_summary)),
        },
        "二值法": {
            "scorecard": _load_formal_scorecard(Path(args.binary_scorecard_root)),
            "profiles": _load_best_profiles(Path(args.binary_summary)),
        },
    }

    rows: list[dict] = []
    sideways_modes = ["关闭", "只过滤", "只降分", "过滤+降分"]

    rows = []
    total_tasks = sum(len(payload["profiles"]) for payload in methods.values()) * len(sideways_modes)
    done_tasks = 0
    for method_name, payload in methods.items():
        add_weights, penalty_weights = payload["scorecard"]
        scale = _非零权重中位数(add_weights, penalty_weights)
        for strategy_label, profile in payload["profiles"].items():
            for sideways_mode in tqdm(sideways_modes, desc=f"{method_name}-{strategy_label}", unit="模式"):
                _write_status(
                    status_path,
                    {
                        "阶段": "运行横盘专项实验",
                        "评分法": method_name,
                        "策略": strategy_label,
                        "横盘模式": sideways_mode,
                        "参数名称": profile["名称"],
                        "完成组数": done_tasks,
                        "总组数": total_tasks,
                    },
                )
                row = _run_single(
                    prepared,
                    all_dates,
                    strategy_label,
                    add_weights,
                    penalty_weights,
                    profile,
                    args.initial_capital,
                    args.max_positions,
                    args.buy_mode,
                    args.replacement_threshold,
                    args.min_hold_days_for_replace,
                    args.max_daily_replacements,
                    args.use_trend_start_pool,
                    args.min_confirmation_hits,
                    args.min_support_hits,
                    sideways_mode,
                    args.sideways_filter_threshold,
                    0.0 if sideways_mode in {"关闭", "只过滤"} else scale,
                )
                row["评分法"] = method_name
                row["参数名称"] = profile["名称"]
                rows.append(row)
                pd.DataFrame(rows).to_csv(output_root / "横盘专项实验结果.csv", index=False, encoding="utf-8-sig")
                done_tasks += 1

    result_df = pd.DataFrame(rows)
    summary: dict[str, dict] = {}
    for method_name in sorted(result_df["评分法"].unique()):
        method_df = result_df[result_df["评分法"] == method_name].copy()
        summary[method_name] = {}
        for strategy_label in ["fixed_take_profit", "fixed_days", "tiered"]:
            sub = method_df[method_df["策略"] == strategy_label].copy()
            if sub.empty:
                continue
            summary[method_name][strategy_label] = {
                "收益最优": sub.sort_values(["年化收益", "夏普比率", "综合评分"], ascending=[False, False, False]).iloc[0].to_dict(),
                "综合最优": sub.sort_values(["综合评分", "年化收益", "夏普比率"], ascending=[False, False, False]).iloc[0].to_dict(),
                "回撤最优": sub.sort_values(["最大回撤", "综合评分", "年化收益"], ascending=[True, False, False]).iloc[0].to_dict(),
            }
    (output_root / "汇总结果.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_status(status_path, {"阶段": "已完成", "完成组数": done_tasks, "总组数": total_tasks})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
