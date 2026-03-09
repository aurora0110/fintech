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
from utils.multi_factor_research.run_trend_rebuild_experiment import _load_best_variant_scorecard
from utils.multi_factor_research.run_weighted_portfolio_backtest import (
    PortfolioConfig,
    _prepare_stock_frames,
    _run_model,
)


def _综合评分(row: dict) -> float:
    return float(row["年化收益"]) * 100.0 - float(row["最大回撤"]) * 35.0 + float(row["夏普比率"]) * 10.0


def _写状态(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _候选配置(
    *,
    横盘降分系数: float,
    重构支撑最少命中数: int,
    再入场模式: str,
) -> dict:
    return {
        "趋势候选池启用": True,
        "趋势候选池模式": "rebuilt_v1",
        "启动确认最少命中数": 1,
        "支撑最少命中数": 1,
        "重构确认最少命中数": 2,
        "重构支撑最少命中数": 重构支撑最少命中数,
        "重构要求收盘站上趋势线": True,
        "重构要求收盘站上多空线": True,
        "再入场模式": 再入场模式,
        "结构恢复确认最少命中数": 1,
        "结构恢复支撑最少命中数": 2,
        "结构恢复要求收盘站上趋势线": True,
        "结构恢复要求收盘站上多空线": True,
        "横盘模式": "只降分",
        "横盘过滤阈值": 0.55,
        "横盘降分系数": 横盘降分系数,
    }


def _退出配置(
    *,
    名称: str,
    固定持有天数: int,
    利润保护模式: str,
    启动失败观察天数: int,
    启动失败最小涨幅: float,
) -> dict:
    return {
        "名称": 名称,
        "固定持有天数": 固定持有天数,
        "初始止损系数": 0.95,
        "固定持有模式": "到期后结构退出",
        "结构退出最长持有天数": 60,
        "利润保护模式": 利润保护模式,
        "启动失败观察天数": 启动失败观察天数,
        "启动失败最小涨幅": 启动失败最小涨幅,
        "保本触发涨幅": 0.05,
        "保本止损系数": 1.0,
    }


def _方案列表() -> list[dict]:
    return [
        {
            "变体名称": "新主基线",
            "候选配置": _候选配置(横盘降分系数=4.0, 重构支撑最少命中数=2, 再入场模式="bull_bear_only"),
            "退出配置": _退出配置(
                名称="固定持有40_趋势线偏离_失败5天3pct",
                固定持有天数=40,
                利润保护模式="趋势线偏离",
                启动失败观察天数=5,
                启动失败最小涨幅=0.03,
            ),
        },
        {
            "变体名称": "持有缩短30天",
            "候选配置": _候选配置(横盘降分系数=4.0, 重构支撑最少命中数=2, 再入场模式="bull_bear_only"),
            "退出配置": _退出配置(
                名称="固定持有30_趋势线偏离_失败5天3pct",
                固定持有天数=30,
                利润保护模式="趋势线偏离",
                启动失败观察天数=5,
                启动失败最小涨幅=0.03,
            ),
        },
        {
            "变体名称": "持有延长50天",
            "候选配置": _候选配置(横盘降分系数=4.0, 重构支撑最少命中数=2, 再入场模式="bull_bear_only"),
            "退出配置": _退出配置(
                名称="固定持有50_趋势线偏离_失败5天3pct",
                固定持有天数=50,
                利润保护模式="趋势线偏离",
                启动失败观察天数=5,
                启动失败最小涨幅=0.03,
            ),
        },
        {
            "变体名称": "利润保护改历史启动特征",
            "候选配置": _候选配置(横盘降分系数=4.0, 重构支撑最少命中数=2, 再入场模式="bull_bear_only"),
            "退出配置": _退出配置(
                名称="固定持有40_历史启动特征_失败5天3pct",
                固定持有天数=40,
                利润保护模式="历史启动特征",
                启动失败观察天数=5,
                启动失败最小涨幅=0.03,
            ),
        },
        {
            "变体名称": "利润保护改ATR",
            "候选配置": _候选配置(横盘降分系数=4.0, 重构支撑最少命中数=2, 再入场模式="bull_bear_only"),
            "退出配置": _退出配置(
                名称="固定持有40_ATR_失败5天3pct",
                固定持有天数=40,
                利润保护模式="ATR",
                启动失败观察天数=5,
                启动失败最小涨幅=0.03,
            ),
        },
        {
            "变体名称": "启动失败放宽7天3pct",
            "候选配置": _候选配置(横盘降分系数=4.0, 重构支撑最少命中数=2, 再入场模式="bull_bear_only"),
            "退出配置": _退出配置(
                名称="固定持有40_趋势线偏离_失败7天3pct",
                固定持有天数=40,
                利润保护模式="趋势线偏离",
                启动失败观察天数=7,
                启动失败最小涨幅=0.03,
            ),
        },
        {
            "变体名称": "启动失败放宽7天4pct",
            "候选配置": _候选配置(横盘降分系数=4.0, 重构支撑最少命中数=2, 再入场模式="bull_bear_only"),
            "退出配置": _退出配置(
                名称="固定持有40_趋势线偏离_失败7天4pct",
                固定持有天数=40,
                利润保护模式="趋势线偏离",
                启动失败观察天数=7,
                启动失败最小涨幅=0.04,
            ),
        },
        {
            "变体名称": "候选池再加强_支撑3",
            "候选配置": _候选配置(横盘降分系数=4.0, 重构支撑最少命中数=3, 再入场模式="bull_bear_only"),
            "退出配置": _退出配置(
                名称="固定持有40_趋势线偏离_失败5天3pct",
                固定持有天数=40,
                利润保护模式="趋势线偏离",
                启动失败观察天数=5,
                启动失败最小涨幅=0.03,
            ),
        },
        {
            "变体名称": "横盘降分减弱2",
            "候选配置": _候选配置(横盘降分系数=2.0, 重构支撑最少命中数=2, 再入场模式="bull_bear_only"),
            "退出配置": _退出配置(
                名称="固定持有40_趋势线偏离_失败5天3pct",
                固定持有天数=40,
                利润保护模式="趋势线偏离",
                启动失败观察天数=5,
                启动失败最小涨幅=0.03,
            ),
        },
        {
            "变体名称": "横盘降分增强6",
            "候选配置": _候选配置(横盘降分系数=6.0, 重构支撑最少命中数=2, 再入场模式="bull_bear_only"),
            "退出配置": _退出配置(
                名称="固定持有40_趋势线偏离_失败5天3pct",
                固定持有天数=40,
                利润保护模式="趋势线偏离",
                启动失败观察天数=5,
                启动失败最小涨幅=0.03,
            ),
        },
        {
            "变体名称": "结构恢复再入场",
            "候选配置": _候选配置(横盘降分系数=4.0, 重构支撑最少命中数=2, 再入场模式="structure_recovery"),
            "退出配置": _退出配置(
                名称="固定持有40_趋势线偏离_失败5天3pct",
                固定持有天数=40,
                利润保护模式="趋势线偏离",
                启动失败观察天数=5,
                启动失败最小涨幅=0.03,
            ),
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="围绕新主基线做小范围局部优化")
    parser.add_argument("--data-dir", default="/Users/lidongyang/Desktop/Qstrategy/data/20260226")
    parser.add_argument("--weighted-scorecard-root", default="/Users/lidongyang/Desktop/Qstrategy/results/formal_scorecard_trend_v2_pool_weighted")
    parser.add_argument("--factor-opt-summary", default="/Users/lidongyang/Desktop/Qstrategy/results/factor_optimization_experiment_v1/汇总结果.json")
    parser.add_argument("--output-root", default="/Users/lidongyang/Desktop/Qstrategy/results/trend_mainline_refine_v1")
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
    result_path = output_root / "主基线局部优化结果.csv"
    summary_path = output_root / "汇总结果.json"

    _写状态(status_path, {"阶段": "加载行情数据", "完成组数": 0})
    stock_data, all_dates = load_price_directory(args.data_dir)
    _写状态(status_path, {"阶段": "预处理因子", "股票数": len(stock_data), "完成组数": 0})
    prepared = _prepare_stock_frames(stock_data)

    add_weights, penalty_weights, best = _load_best_variant_scorecard(
        "加权法",
        Path(args.weighted_scorecard_root),
        Path(args.factor_opt_summary),
    )

    fieldnames = [
        "评分法",
        "评分方案",
        "变体名称",
        "退出配置",
        "候选配置",
        "年化收益",
        "最大回撤",
        "夏普比率",
        "波动率",
        "资金倍数",
        "最终资金",
        "交易次数",
        "完整轮次交易数",
        "平均持有期间收益率",
        "盈利轮次占比",
        "盈利轮次平均收益率",
        "亏损轮次平均收益率",
        "综合评分",
    ]

    rows: list[dict] = []
    variants = _方案列表()
    total = len(variants)
    for idx, variant in enumerate(tqdm(variants, desc="主基线局部优化", unit="组"), start=1):
        _写状态(
            status_path,
            {
                "阶段": "运行主基线局部优化",
                "完成组数": idx - 1,
                "总组数": total,
                "评分法": "加权法",
                "评分方案": str(best["方案名称"]),
                "变体名称": variant["变体名称"],
            },
        )
        config = PortfolioConfig(
            initial_capital=args.initial_capital,
            max_positions=args.max_positions,
            replacement_threshold=args.replacement_threshold,
            min_hold_days_for_replace=args.min_hold_days_for_replace,
            max_daily_replacements=args.max_daily_replacements,
            buy_mode=args.buy_mode,
            exit_profile=variant["退出配置"],
            use_trend_start_pool=True,
            trend_pool_mode=str(variant["候选配置"]["趋势候选池模式"]),
            min_confirmation_hits=int(variant["候选配置"]["启动确认最少命中数"]),
            min_support_hits=int(variant["候选配置"]["支撑最少命中数"]),
            rebuilt_min_confirmation_hits=int(variant["候选配置"]["重构确认最少命中数"]),
            rebuilt_min_support_hits=int(variant["候选配置"]["重构支撑最少命中数"]),
            rebuilt_require_close_above_trend=bool(variant["候选配置"]["重构要求收盘站上趋势线"]),
            rebuilt_require_close_above_bull_bear=bool(variant["候选配置"]["重构要求收盘站上多空线"]),
            reentry_mode=str(variant["候选配置"]["再入场模式"]),
            recovery_min_confirmation_hits=int(variant["候选配置"]["结构恢复确认最少命中数"]),
            recovery_min_support_hits=int(variant["候选配置"]["结构恢复支撑最少命中数"]),
            recovery_require_close_above_trend=bool(variant["候选配置"]["结构恢复要求收盘站上趋势线"]),
            recovery_require_close_above_bull_bear=bool(variant["候选配置"]["结构恢复要求收盘站上多空线"]),
            sideways_mode=str(variant["候选配置"]["横盘模式"]),
            sideways_filter_threshold=float(variant["候选配置"]["横盘过滤阈值"]),
            sideways_score_penalty_scale=float(variant["候选配置"]["横盘降分系数"]),
        )
        result = _run_model("fixed_days", prepared, all_dates, add_weights, penalty_weights, config)
        metrics = result["metrics"]
        row = {
            "评分法": "加权法",
            "评分方案": str(best["方案名称"]),
            "变体名称": variant["变体名称"],
            "退出配置": json.dumps(variant["退出配置"], ensure_ascii=False),
            "候选配置": json.dumps(variant["候选配置"], ensure_ascii=False),
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
        rows.append(row)
        with result_path.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    rows_sorted = sorted(rows, key=lambda item: float(item["综合评分"]), reverse=True)
    summary = {
        "总组数": len(rows),
        "综合最优": rows_sorted[0] if rows_sorted else None,
        "收益最优": max(rows, key=lambda item: float(item["年化收益"])) if rows else None,
        "回撤最优": min(rows, key=lambda item: float(item["最大回撤"])) if rows else None,
        "全部结果": rows_sorted,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _写状态(
        status_path,
        {
            "阶段": "已完成",
            "完成组数": len(rows),
            "总组数": len(rows),
            "最优变体": summary["综合最优"]["变体名称"] if summary["综合最优"] else None,
        },
    )


if __name__ == "__main__":
    main()
