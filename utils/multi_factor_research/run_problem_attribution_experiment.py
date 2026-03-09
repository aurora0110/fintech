from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
import sys

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def _写状态(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _非零权重中位数(add_weights: dict, penalty_weights: dict) -> float:
    values = [abs(float(v)) for v in list(add_weights.values()) + list(penalty_weights.values()) if abs(float(v)) > 0]
    if not values:
        return 1.0
    return max(1.0, float(statistics.median(values)))


def _读取横盘实验最优(summary_path: Path) -> dict[str, dict]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    result: dict[str, dict] = {}
    for strategy_label in ["fixed_take_profit", "fixed_days", "tiered"]:
        best = payload["加权法"][strategy_label]["综合最优"]
        result[strategy_label] = {
            "参数": best["参数"],
            "横盘模式": best["横盘模式"],
            "横盘过滤阈值": float(best["横盘过滤阈值"]),
            "横盘降分系数": float(best["横盘降分系数"]),
        }
    return result


def _候选池方案(method_name: str, scale: float) -> list[dict]:
    baseline_sideways = "只降分" if method_name == "加权法" else "关闭"
    baseline_scale = scale if baseline_sideways == "只降分" else 0.0
    return [
        {
            "名称": "基础池",
            "use_trend_start_pool": False,
            "min_confirmation_hits": 0,
            "min_support_hits": 0,
            "sideways_mode": baseline_sideways,
            "sideways_filter_threshold": 0.55,
            "sideways_score_penalty_scale": baseline_scale,
        },
        {
            "名称": "趋势池V1",
            "use_trend_start_pool": True,
            "min_confirmation_hits": 1,
            "min_support_hits": 1,
            "sideways_mode": baseline_sideways,
            "sideways_filter_threshold": 0.55,
            "sideways_score_penalty_scale": baseline_scale,
        },
        {
            "名称": "强趋势池_确认2",
            "use_trend_start_pool": True,
            "min_confirmation_hits": 2,
            "min_support_hits": 1,
            "sideways_mode": baseline_sideways,
            "sideways_filter_threshold": 0.55,
            "sideways_score_penalty_scale": baseline_scale,
        },
        {
            "名称": "强趋势池_支撑2",
            "use_trend_start_pool": True,
            "min_confirmation_hits": 1,
            "min_support_hits": 2,
            "sideways_mode": baseline_sideways,
            "sideways_filter_threshold": 0.55,
            "sideways_score_penalty_scale": baseline_scale,
        },
        {
            "名称": "强趋势池_确认2支撑2",
            "use_trend_start_pool": True,
            "min_confirmation_hits": 2,
            "min_support_hits": 2,
            "sideways_mode": baseline_sideways,
            "sideways_filter_threshold": 0.55,
            "sideways_score_penalty_scale": baseline_scale,
        },
    ]


def _run_single(
    prepared,
    all_dates,
    strategy_label: str,
    add_weights: dict,
    penalty_weights: dict,
    profile: dict,
    candidate_cfg: dict,
    initial_capital: float,
    max_positions: int,
    buy_mode: str,
    replacement_threshold: float,
    min_hold_days_for_replace: int,
    max_daily_replacements: int,
) -> dict:
    config = PortfolioConfig(
        initial_capital=initial_capital,
        max_positions=max_positions,
        replacement_threshold=replacement_threshold,
        min_hold_days_for_replace=min_hold_days_for_replace,
        max_daily_replacements=max_daily_replacements,
        buy_mode=buy_mode,
        exit_profile=profile,
        use_trend_start_pool=candidate_cfg["use_trend_start_pool"],
        min_confirmation_hits=candidate_cfg["min_confirmation_hits"],
        min_support_hits=candidate_cfg["min_support_hits"],
        sideways_mode=candidate_cfg["sideways_mode"],
        sideways_filter_threshold=candidate_cfg["sideways_filter_threshold"],
        sideways_score_penalty_scale=candidate_cfg["sideways_score_penalty_scale"],
    )
    result = _run_model(strategy_label, prepared, all_dates, add_weights, penalty_weights, config)
    metrics = result["metrics"]
    row = {
        "策略": strategy_label,
        "参数": profile,
        "候选池": candidate_cfg["名称"],
        "使用趋势启动池": candidate_cfg["use_trend_start_pool"],
        "确认因子最少命中数": int(candidate_cfg["min_confirmation_hits"]),
        "动量结构因子最少命中数": int(candidate_cfg["min_support_hits"]),
        "横盘模式": candidate_cfg["sideways_mode"],
        "横盘过滤阈值": float(candidate_cfg["sideways_filter_threshold"]),
        "横盘降分系数": float(candidate_cfg["sideways_score_penalty_scale"]),
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
    parser = argparse.ArgumentParser(description="问题归因实验：固定候选池测不同卖出；固定卖出测不同候选池")
    parser.add_argument("--data-dir", default="/Users/lidongyang/Desktop/Qstrategy/data/20260226")
    parser.add_argument("--weighted-scorecard-root", default="/Users/lidongyang/Desktop/Qstrategy/results/formal_scorecard_trend_v2_pool_weighted")
    parser.add_argument("--binary-scorecard-root", default="/Users/lidongyang/Desktop/Qstrategy/results/formal_scorecard_trend_v2_pool_binary")
    parser.add_argument("--sideways-summary", default="/Users/lidongyang/Desktop/Qstrategy/results/sideways_special_experiment_v1/汇总结果.json")
    parser.add_argument("--output-root", default="/Users/lidongyang/Desktop/Qstrategy/results/problem_attribution_experiment_v1")
    parser.add_argument("--initial-capital", type=float, default=10_000_000.0)
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--buy-mode", choices=["staged", "full", "strict_full"], default="full")
    parser.add_argument("--replacement-threshold", type=float, default=0.03)
    parser.add_argument("--min-hold-days-for-replace", type=int, default=5)
    parser.add_argument("--max-daily-replacements", type=int, default=1)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "状态.json"

    _写状态(status_path, {"阶段": "加载行情数据", "完成组数": 0})
    stock_data, all_dates = load_price_directory(args.data_dir)
    _写状态(status_path, {"阶段": "预处理因子", "股票数": len(stock_data), "完成组数": 0})
    prepared = _prepare_stock_frames(stock_data)

    sideways_payload = json.loads(Path(args.sideways_summary).read_text(encoding="utf-8"))
    methods = {
        "加权法": {
            "scorecard": _load_formal_scorecard(Path(args.weighted_scorecard_root)),
            "best": {
                strategy: sideways_payload["加权法"][strategy]["综合最优"]
                for strategy in ["fixed_take_profit", "fixed_days", "tiered"]
            },
        },
        "二值法": {
            "scorecard": _load_formal_scorecard(Path(args.binary_scorecard_root)),
            "best": {
                strategy: sideways_payload["二值法"][strategy]["综合最优"]
                for strategy in ["fixed_take_profit", "fixed_days", "tiered"]
            },
        },
    }

    tasks: list[dict] = []
    for method_name, payload in methods.items():
        add_weights, penalty_weights = payload["scorecard"]
        scale = _非零权重中位数(add_weights, penalty_weights)
        fixed_pool_cfg = {
            "名称": "当前最优候选环境",
            "use_trend_start_pool": True,
            "min_confirmation_hits": 1,
            "min_support_hits": 1,
            "sideways_mode": "只降分" if method_name == "加权法" else "关闭",
            "sideways_filter_threshold": 0.55,
            "sideways_score_penalty_scale": scale if method_name == "加权法" else 0.0,
        }
        for strategy_label in ["fixed_take_profit", "fixed_days", "tiered"]:
            best_payload = payload["best"][strategy_label]
            tasks.append(
                {
                    "实验类别": "固定候选池测不同卖出",
                    "评分法": method_name,
                    "策略": strategy_label,
                    "参数名称": best_payload["参数名称"],
                    "profile": best_payload["参数"],
                    "candidate_cfg": fixed_pool_cfg,
                    "scorecard": payload["scorecard"],
                }
            )

        fixed_exit = payload["best"]["fixed_days"]
        for candidate_cfg in _候选池方案(method_name, scale):
            tasks.append(
                {
                    "实验类别": "固定卖出测不同候选池",
                    "评分法": method_name,
                    "策略": "fixed_days",
                    "参数名称": fixed_exit["参数名称"],
                    "profile": fixed_exit["参数"],
                    "candidate_cfg": candidate_cfg,
                    "scorecard": payload["scorecard"],
                }
            )

    result_path = output_root / "问题归因实验结果.csv"
    rows: list[dict] = []
    done = 0
    total = len(tasks)
    for task in tqdm(tasks, desc="问题归因实验", unit="组"):
        _写状态(
            status_path,
            {
                "阶段": "运行问题归因实验",
                "实验类别": task["实验类别"],
                "评分法": task["评分法"],
                "策略": task["策略"],
                "候选池": task["candidate_cfg"]["名称"],
                "参数名称": task["参数名称"],
                "完成组数": done,
                "总组数": total,
            },
        )
        add_weights, penalty_weights = task["scorecard"]
        row = _run_single(
            prepared,
            all_dates,
            task["策略"],
            add_weights,
            penalty_weights,
            task["profile"],
            task["candidate_cfg"],
            args.initial_capital,
            args.max_positions,
            args.buy_mode,
            args.replacement_threshold,
            args.min_hold_days_for_replace,
            args.max_daily_replacements,
        )
        row["实验类别"] = task["实验类别"]
        row["评分法"] = task["评分法"]
        row["参数名称"] = task["参数名称"]
        rows.append(row)
        pd.DataFrame(rows).to_csv(result_path, index=False, encoding="utf-8-sig")
        done += 1

    result_df = pd.DataFrame(rows)
    summary: dict[str, dict] = {}
    for experiment_name in ["固定候选池测不同卖出", "固定卖出测不同候选池"]:
        sub_exp = result_df[result_df["实验类别"] == experiment_name].copy()
        if sub_exp.empty:
            continue
        summary[experiment_name] = {}
        for method_name in sorted(sub_exp["评分法"].unique()):
            sub = sub_exp[sub_exp["评分法"] == method_name].copy()
            summary[experiment_name][method_name] = {
                "收益最优": sub.sort_values(["年化收益", "夏普比率", "综合评分"], ascending=[False, False, False]).iloc[0].to_dict(),
                "综合最优": sub.sort_values(["综合评分", "年化收益", "夏普比率"], ascending=[False, False, False]).iloc[0].to_dict(),
                "回撤最优": sub.sort_values(["最大回撤", "综合评分", "年化收益"], ascending=[True, False, False]).iloc[0].to_dict(),
            }
    (output_root / "汇总结果.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _写状态(status_path, {"阶段": "已完成", "完成组数": done, "总组数": total})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
