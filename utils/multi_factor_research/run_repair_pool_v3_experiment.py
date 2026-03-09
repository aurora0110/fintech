from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data_loader import load_price_directory
from utils.multi_factor_research.run_weighted_portfolio_backtest import (
    PortfolioConfig,
    _prepare_stock_frames,
    _run_model,
)


def _写状态(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _综合评分(row: dict) -> float:
    trades = float(row["交易次数"])
    annual = float(row["年化收益"])
    mdd = float(row["最大回撤"])
    sharpe = float(row["夏普比率"])
    avg_round = float(row["平均持有期间收益率"])
    if trades <= 0:
        return -1e9
    return annual * 100 - mdd * 20 + sharpe * 10 + avg_round * 8


def _评分卡(method: str) -> tuple[dict[str, float], dict[str, float]]:
    weighted_add = {
        "first_pullback_trend_after_cross_factor": 8.0,
        "first_j_buy_after_cross_factor": 7.0,
        "low_volume_pullback_factor": 6.0,
        "rsi_bull_factor": 4.0,
        "daily_ma_bull_factor": 3.0,
        "staged_volume_burst_factor": 2.0,
    }
    weighted_penalty = {
        "flat_trend_slope_penalty": 8.0,
        "box_oscillation_penalty": 7.0,
        "extreme_bull_run_penalty": 6.0,
        "bearish_volume_penalty": 5.0,
        "price_amplitude_factor": 4.0,
    }
    if method == "二值法":
        return ({k: 1.0 for k in weighted_add}, {k: 1.0 for k in weighted_penalty})
    return weighted_add, weighted_penalty


def _参数列表() -> list[dict]:
    return [
        {
            "策略": "fixed_days",
            "变体名称": "固定持有5天",
            "参数": {
                "固定持有模式": "到期全卖",
                "固定持有天数": 5,
                "结构退出最长持有天数": 5,
                "初始止损系数": 0.95,
                "启动失败观察天数": 3,
                "启动失败最小涨幅": 0.02,
                "第二层启动失败观察天数": 0,
                "第二层启动失败最小涨幅": 0.0,
                "保本触发涨幅": 0.03,
                "保本止损系数": 1.0,
            },
        },
        {
            "策略": "fixed_days",
            "变体名称": "固定持有7天",
            "参数": {
                "固定持有模式": "到期全卖",
                "固定持有天数": 7,
                "结构退出最长持有天数": 7,
                "初始止损系数": 0.95,
                "启动失败观察天数": 3,
                "启动失败最小涨幅": 0.02,
                "第二层启动失败观察天数": 0,
                "第二层启动失败最小涨幅": 0.0,
                "保本触发涨幅": 0.03,
                "保本止损系数": 1.0,
            },
        },
        {
            "策略": "fixed_days",
            "变体名称": "固定持有10天",
            "参数": {
                "固定持有模式": "到期全卖",
                "固定持有天数": 10,
                "结构退出最长持有天数": 10,
                "初始止损系数": 0.95,
                "启动失败观察天数": 3,
                "启动失败最小涨幅": 0.02,
                "第二层启动失败观察天数": 0,
                "第二层启动失败最小涨幅": 0.0,
                "保本触发涨幅": 0.03,
                "保本止损系数": 1.0,
            },
        },
        {
            "策略": "fixed_take_profit",
            "变体名称": "固定止盈5pct",
            "参数": {
                "固定止盈模式": "全卖",
                "止盈涨幅": 0.05,
                "最长持有天数": 10,
                "初始止损系数": 0.95,
                "启动失败观察天数": 3,
                "启动失败最小涨幅": 0.02,
                "第二层启动失败观察天数": 0,
                "第二层启动失败最小涨幅": 0.0,
                "保本触发涨幅": 0.03,
                "保本止损系数": 1.0,
            },
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="修复型候选池3.0小矩阵实验")
    parser.add_argument("--data-dir", default="/Users/lidongyang/Desktop/Qstrategy/data/20260226")
    parser.add_argument("--output-root", default="/Users/lidongyang/Desktop/Qstrategy/results/repair_pool_v3_experiment_v1")
    parser.add_argument("--initial-capital", type=float, default=10_000_000.0)
    parser.add_argument("--max-positions", type=int, default=10)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "状态.json"
    result_path = output_root / "修复候选池3.0结果.csv"
    summary_path = output_root / "汇总结果.json"

    _写状态(status_path, {"阶段": "加载行情数据", "完成组数": 0})
    stock_data, all_dates = load_price_directory(args.data_dir)
    _写状态(status_path, {"阶段": "预处理因子", "股票数": len(stock_data), "完成组数": 0})
    prepared = _prepare_stock_frames(stock_data)

    tasks = []
    for method in ["加权法", "二值法"]:
        add_weights, penalty_weights = _评分卡(method)
        for base in _参数列表():
            tasks.append({"评分法": method, "加分": add_weights, "扣分": penalty_weights, **base})

    fieldnames = [
        "评分法", "策略", "变体名称", "参数", "年化收益", "最大回撤", "夏普比率", "资金倍数", "最终资金",
        "交易次数", "平均持有期间收益率", "盈利轮次占比", "盈利轮次平均收益率", "亏损轮次平均收益率", "综合评分",
    ]
    rows = []
    with result_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, task in enumerate(tqdm(tasks, desc="修复候选池3.0实验", unit="组"), start=1):
            _写状态(
                status_path,
                {
                    "阶段": "运行修复候选池3.0实验",
                    "完成组数": idx - 1,
                    "总组数": len(tasks),
                    "评分法": task["评分法"],
                    "策略": task["策略"],
                    "变体名称": task["变体名称"],
                },
            )
            config = PortfolioConfig(
                initial_capital=args.initial_capital,
                max_positions=args.max_positions,
                buy_mode="strict_full",
                replacement_threshold=0.03,
                min_hold_days_for_replace=3,
                max_daily_replacements=1,
                use_trend_start_pool=True,
                trend_pool_mode="repair_v3",
                rebuilt_min_confirmation_hits=1,
                rebuilt_min_support_hits=1,
                sideways_mode="只降分",
                sideways_score_penalty_scale=4.0,
                exit_profile=task["参数"],
            )
            result = _run_model(task["策略"], prepared, all_dates, task["加分"], task["扣分"], config)
            row = {
                "评分法": task["评分法"],
                "策略": task["策略"],
                "变体名称": task["变体名称"],
                "参数": json.dumps(task["参数"], ensure_ascii=False),
                "年化收益": float(result["metrics"]["annual_return"]),
                "最大回撤": float(result["metrics"]["max_drawdown"]),
                "夏普比率": float(result["metrics"]["sharpe"]),
                "资金倍数": float(result["metrics"]["final_multiple"]),
                "最终资金": float(result["final_equity"]),
                "交易次数": int(result["trade_count"]),
                "平均持有期间收益率": float(result["平均持有期间收益率"]),
                "盈利轮次占比": float(result["盈利轮次占比"]),
                "盈利轮次平均收益率": float(result["盈利轮次平均收益率"]),
                "亏损轮次平均收益率": float(result["亏损轮次平均收益率"]),
            }
            row["综合评分"] = _综合评分(row)
            writer.writerow(row)
            f.flush()
            rows.append(row)

    best = max(rows, key=_综合评分) if rows else None
    summary = {"总组数": len(rows), "最优结果": best}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _写状态(status_path, {"阶段": "已完成", "完成组数": len(rows), "总组数": len(tasks), "最优变体": best["变体名称"] if best else ""})


if __name__ == "__main__":
    main()
