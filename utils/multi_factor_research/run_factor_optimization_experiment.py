from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data_loader import load_price_directory
from utils.multi_factor_research.factor_calculator import FACTOR_NAME_MAP
from utils.multi_factor_research.run_weighted_portfolio_backtest import (
    PortfolioConfig,
    _load_formal_scorecard,
    _prepare_stock_frames,
    _run_model,
)


def _综合评分(row: dict) -> float:
    return float(row["年化收益"]) * 100.0 - float(row["最大回撤"]) * 35.0 + float(row["夏普比率"]) * 10.0


def _写状态(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


中文到字段 = {v: k for k, v in FACTOR_NAME_MAP.items()}


def 字段(name: str) -> str:
    return 中文到字段[name]


趋势核心 = [
    "阶段性放量因子",
    "趋势线上穿多空线后第一次回踩趋势线因子",
    "趋势线上穿多空线后第一次J满足买入条件因子",
    "MACD DIF因子",
    "RSI多头结构因子",
    "日线均线多头结构因子",
    "关键K支撑因子",
]

排序辅助 = [
    "三十日连续大阳线过热扣分",
    "长阴短柱因子",
    "跌破趋势线扣分",
    "放量滞涨扣分",
]

修复型正向 = [
    "异常振幅扣分",
    "J值快速下行因子",
    "跌破关键K最低价重扣分",
    "跌破关键K收盘扣分",
]

基础风险扣分 = [
    "价格波动幅度因子",
    "跌破多空线扣分",
    "B1确认因子",
    "三十日阴线显著偏多扣分",
    "回踩趋势线/多空线确认因子",
    "放量阴线扣分",
    "趋势线上穿多空线后第一次回踩多空线因子",
]

横盘相关扣分 = [
    "十日横盘因子",
    "二十日横盘因子",
]


def _weights_from_names(base_weights: dict[str, float], names: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in names:
        key = 字段(name)
        if key in base_weights:
            result[key] = float(base_weights[key])
    return result


def _variant_scorecards(method_name: str, add_weights: dict[str, float], penalty_weights: dict[str, float]) -> list[dict]:
    trend_core = _weights_from_names(add_weights, 趋势核心)
    sorting_support = _weights_from_names(add_weights, 排序辅助)
    repair_positive = _weights_from_names(add_weights, 修复型正向)
    base_risk = _weights_from_names(penalty_weights, 基础风险扣分)
    sideways_risk = _weights_from_names(penalty_weights, 横盘相关扣分)

    variants = [
        {
            "名称": "基线评分卡",
            "加分": dict(add_weights),
            "扣分": dict(penalty_weights),
        },
        {
            "名称": "仅趋势核心",
            "加分": dict(trend_core),
            "扣分": dict(base_risk | sideways_risk),
        },
        {
            "名称": "趋势核心加排序辅助",
            "加分": dict(trend_core | sorting_support),
            "扣分": dict(base_risk | sideways_risk),
        },
        {
            "名称": "基线去修复型正向",
            "加分": {k: v for k, v in add_weights.items() if k not in repair_positive},
            "扣分": dict(penalty_weights),
        },
        {
            "名称": "修复型转扣分",
            "加分": {k: v for k, v in add_weights.items() if k not in repair_positive},
            "扣分": dict(penalty_weights | repair_positive),
        },
        {
            "名称": "趋势精简版",
            "加分": dict(trend_core | sorting_support),
            "扣分": dict(base_risk),
        },
    ]
    for item in variants:
        item["评分法"] = method_name
    return variants


def _best_profiles_from_problem_attr() -> dict[str, dict]:
    path = Path("/Users/lidongyang/Desktop/Qstrategy/results/problem_attribution_experiment_v1/问题归因实验结果.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")
    result: dict[str, dict] = {}
    for method_name in ["加权法", "二值法"]:
        sub = df[
            (df["实验类别"] == "固定候选池测不同卖出")
            & (df["评分法"] == method_name)
            & (df["策略"] == "fixed_days")
        ].copy()
        best = sub.sort_values(["年化收益", "夏普比率", "综合评分"], ascending=[False, False, False]).iloc[0].to_dict()
        result[method_name] = {
            "参数名称": best["参数名称"],
            "参数": eval(best["参数"]),
        }
    return result


def _run_single(
    prepared,
    all_dates,
    add_weights: dict[str, float],
    penalty_weights: dict[str, float],
    exit_profile: dict,
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
        exit_profile=exit_profile,
        use_trend_start_pool=True,
        min_confirmation_hits=1,
        min_support_hits=1,
        sideways_mode=candidate_cfg["横盘模式"],
        sideways_filter_threshold=float(candidate_cfg["横盘过滤阈值"]),
        sideways_score_penalty_scale=float(candidate_cfg["横盘降分系数"]),
    )
    result = _run_model("fixed_days", prepared, all_dates, add_weights, penalty_weights, config)
    metrics = result["metrics"]
    row = {
        "策略": "fixed_days",
        "参数": exit_profile,
        "横盘模式": candidate_cfg["横盘模式"],
        "横盘过滤阈值": float(candidate_cfg["横盘过滤阈值"]),
        "横盘降分系数": float(candidate_cfg["横盘降分系数"]),
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
    parser = argparse.ArgumentParser(description="因子优化实验：固定趋势池V1与固定持有退出，仅比较评分卡分组与因子角色")
    parser.add_argument("--data-dir", default="/Users/lidongyang/Desktop/Qstrategy/data/20260226")
    parser.add_argument("--weighted-scorecard-root", default="/Users/lidongyang/Desktop/Qstrategy/results/formal_scorecard_trend_v2_pool_weighted")
    parser.add_argument("--binary-scorecard-root", default="/Users/lidongyang/Desktop/Qstrategy/results/formal_scorecard_trend_v2_pool_binary")
    parser.add_argument("--output-root", default="/Users/lidongyang/Desktop/Qstrategy/results/factor_optimization_experiment_v1")
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
    result_path = output_root / "因子优化实验结果.csv"

    _写状态(status_path, {"阶段": "加载行情数据", "完成组数": 0})
    stock_data, all_dates = load_price_directory(args.data_dir)
    _写状态(status_path, {"阶段": "预处理因子", "股票数": len(stock_data), "完成组数": 0})
    prepared = _prepare_stock_frames(stock_data)

    weighted_card = _load_formal_scorecard(Path(args.weighted_scorecard_root))
    binary_card = _load_formal_scorecard(Path(args.binary_scorecard_root))
    best_profiles = _best_profiles_from_problem_attr()

    candidate_cfgs = {
        "加权法": {"横盘模式": "只降分", "横盘过滤阈值": 0.55, "横盘降分系数": 4.0},
        "二值法": {"横盘模式": "关闭", "横盘过滤阈值": 0.55, "横盘降分系数": 0.0},
    }

    tasks: list[dict] = []
    for method_name, (add_weights, penalty_weights) in [("加权法", weighted_card), ("二值法", binary_card)]:
        for variant in _variant_scorecards(method_name, add_weights, penalty_weights):
            tasks.append(
                {
                    "评分法": method_name,
                    "方案名称": variant["名称"],
                    "加分": variant["加分"],
                    "扣分": variant["扣分"],
                    "参数": best_profiles[method_name]["参数"],
                    "参数名称": best_profiles[method_name]["参数名称"],
                    "候选环境": candidate_cfgs[method_name],
                }
            )

    rows: list[dict] = []
    total = len(tasks)
    done = 0
    for task in tqdm(tasks, desc="因子优化实验", unit="组"):
        _写状态(
            status_path,
            {
                "阶段": "运行因子优化实验",
                "评分法": task["评分法"],
                "方案名称": task["方案名称"],
                "参数名称": task["参数名称"],
                "完成组数": done,
                "总组数": total,
            },
        )
        row = _run_single(
            prepared,
            all_dates,
            task["加分"],
            task["扣分"],
            task["参数"],
            task["候选环境"],
            args.initial_capital,
            args.max_positions,
            args.buy_mode,
            args.replacement_threshold,
            args.min_hold_days_for_replace,
            args.max_daily_replacements,
        )
        row["评分法"] = task["评分法"]
        row["方案名称"] = task["方案名称"]
        row["参数名称"] = task["参数名称"]
        row["加分因子数"] = len(task["加分"])
        row["扣分因子数"] = len(task["扣分"])
        row["加分因子"] = "; ".join(sorted(task["加分"].keys()))
        row["扣分因子"] = "; ".join(sorted(task["扣分"].keys()))
        rows.append(row)
        pd.DataFrame(rows).to_csv(result_path, index=False, encoding="utf-8-sig")
        done += 1

    result_df = pd.DataFrame(rows)
    summary: dict[str, dict] = {}
    for method_name in ["加权法", "二值法"]:
        sub = result_df[result_df["评分法"] == method_name].copy()
        summary[method_name] = {
            "收益最优": sub.sort_values(["年化收益", "夏普比率", "综合评分"], ascending=[False, False, False]).iloc[0].to_dict(),
            "综合最优": sub.sort_values(["综合评分", "年化收益", "夏普比率"], ascending=[False, False, False]).iloc[0].to_dict(),
            "回撤最优": sub.sort_values(["最大回撤", "综合评分", "年化收益"], ascending=[True, False, False]).iloc[0].to_dict(),
        }

    (output_root / "汇总结果.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _写状态(status_path, {"阶段": "已完成", "完成组数": done, "总组数": total})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
