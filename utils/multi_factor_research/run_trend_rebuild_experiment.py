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
    _load_formal_scorecard,
    _prepare_stock_frames,
    _run_model,
)


def _综合评分(row: dict) -> float:
    return float(row["年化收益"]) * 100.0 - float(row["最大回撤"]) * 35.0 + float(row["夏普比率"]) * 10.0


def _写状态(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_best_variant_scorecard(method_name: str, scorecard_root: Path, factor_opt_summary_path: Path) -> tuple[dict[str, float], dict[str, float], dict]:
    summary = json.loads(factor_opt_summary_path.read_text(encoding="utf-8"))
    best = summary[method_name]["综合最优"]
    base_add, base_penalty = _load_formal_scorecard(scorecard_root)

    add_names = [name.strip() for name in str(best["加分因子"]).split(";") if name.strip()]
    penalty_names = [name.strip() for name in str(best["扣分因子"]).split(";") if name.strip()]

    add_weights = {name: float(base_add[name]) for name in add_names if name in base_add}
    penalty_weights = {name: float(base_penalty[name]) for name in penalty_names if name in base_penalty}
    return add_weights, penalty_weights, best


def _baseline_candidate_cfg(best: dict) -> dict:
    return {
        "趋势候选池启用": True,
        "趋势候选池模式": "start_v1",
        "启动确认最少命中数": 1,
        "支撑最少命中数": 1,
        "重构确认最少命中数": 1,
        "重构支撑最少命中数": 2,
        "重构要求收盘站上趋势线": True,
        "重构要求收盘站上多空线": True,
        "再入场模式": "bull_bear_only",
        "结构恢复确认最少命中数": 1,
        "结构恢复支撑最少命中数": 2,
        "结构恢复要求收盘站上趋势线": True,
        "结构恢复要求收盘站上多空线": True,
        "横盘模式": str(best["横盘模式"]),
        "横盘过滤阈值": float(best["横盘过滤阈值"]),
        "横盘降分系数": float(best["横盘降分系数"]),
    }


def _rebuilt_exit_profile(name: str, pool_strict: bool) -> dict:
    return {
        "名称": name,
        "固定持有天数": 20 if pool_strict else 25,
        "结构退出最长持有天数": 60,
        "固定持有模式": "趋势主线重构",
        "初始止损系数": 0.95,
        "利润保护模式": "趋势线偏离",
        "启动失败观察天数": 5,
        "启动失败最小涨幅": 0.03,
        "第二层启动失败观察天数": 10,
        "第二层启动失败最小涨幅": 0.05,
        "保本触发涨幅": 0.05,
        "保本止损系数": 1.0,
    }


def _variant_specs(best: dict) -> list[dict]:
    baseline_cfg = _baseline_candidate_cfg(best)
    baseline_exit = dict(best["参数"])
    return [
        {
            "变体名称": "当前最优基线",
            "候选配置": dict(baseline_cfg),
            "退出配置": baseline_exit,
        },
        {
            "变体名称": "重构候选池",
            "候选配置": {
                **baseline_cfg,
                "趋势候选池模式": "rebuilt_v1",
                "重构确认最少命中数": 1,
                "重构支撑最少命中数": 2,
                "重构要求收盘站上趋势线": True,
                "重构要求收盘站上多空线": True,
            },
            "退出配置": baseline_exit,
        },
        {
            "变体名称": "重构候选池_强确认",
            "候选配置": {
                **baseline_cfg,
                "趋势候选池模式": "rebuilt_v1",
                "重构确认最少命中数": 2,
                "重构支撑最少命中数": 2,
                "重构要求收盘站上趋势线": True,
                "重构要求收盘站上多空线": True,
            },
            "退出配置": baseline_exit,
        },
        {
            "变体名称": "重构候选池_主线重构退出",
            "候选配置": {
                **baseline_cfg,
                "趋势候选池模式": "rebuilt_v1",
                "重构确认最少命中数": 1,
                "重构支撑最少命中数": 2,
                "重构要求收盘站上趋势线": True,
                "重构要求收盘站上多空线": True,
            },
            "退出配置": _rebuilt_exit_profile("趋势主线重构退出", pool_strict=False),
        },
        {
            "变体名称": "重构候选池_主线重构退出_结构恢复再入场",
            "候选配置": {
                **baseline_cfg,
                "趋势候选池模式": "rebuilt_v1",
                "重构确认最少命中数": 1,
                "重构支撑最少命中数": 2,
                "重构要求收盘站上趋势线": True,
                "重构要求收盘站上多空线": True,
                "再入场模式": "structure_recovery",
                "结构恢复确认最少命中数": 1,
                "结构恢复支撑最少命中数": 2,
                "结构恢复要求收盘站上趋势线": True,
                "结构恢复要求收盘站上多空线": True,
            },
            "退出配置": _rebuilt_exit_profile("趋势主线重构退出_结构恢复再入场", pool_strict=False),
        },
    ]


def _run_single(prepared, all_dates, add_weights: dict[str, float], penalty_weights: dict[str, float], variant: dict, args) -> dict:
    candidate_cfg = variant["候选配置"]
    config = PortfolioConfig(
        initial_capital=args.initial_capital,
        max_positions=args.max_positions,
        replacement_threshold=args.replacement_threshold,
        min_hold_days_for_replace=args.min_hold_days_for_replace,
        max_daily_replacements=args.max_daily_replacements,
        buy_mode=args.buy_mode,
        exit_profile=variant["退出配置"],
        use_trend_start_pool=bool(candidate_cfg["趋势候选池启用"]),
        trend_pool_mode=str(candidate_cfg["趋势候选池模式"]),
        min_confirmation_hits=int(candidate_cfg["启动确认最少命中数"]),
        min_support_hits=int(candidate_cfg["支撑最少命中数"]),
        rebuilt_min_confirmation_hits=int(candidate_cfg["重构确认最少命中数"]),
        rebuilt_min_support_hits=int(candidate_cfg["重构支撑最少命中数"]),
        rebuilt_require_close_above_trend=bool(candidate_cfg["重构要求收盘站上趋势线"]),
        rebuilt_require_close_above_bull_bear=bool(candidate_cfg["重构要求收盘站上多空线"]),
        reentry_mode=str(candidate_cfg["再入场模式"]),
        recovery_min_confirmation_hits=int(candidate_cfg["结构恢复确认最少命中数"]),
        recovery_min_support_hits=int(candidate_cfg["结构恢复支撑最少命中数"]),
        recovery_require_close_above_trend=bool(candidate_cfg["结构恢复要求收盘站上趋势线"]),
        recovery_require_close_above_bull_bear=bool(candidate_cfg["结构恢复要求收盘站上多空线"]),
        sideways_mode=str(candidate_cfg["横盘模式"]),
        sideways_filter_threshold=float(candidate_cfg["横盘过滤阈值"]),
        sideways_score_penalty_scale=float(candidate_cfg["横盘降分系数"]),
    )
    result = _run_model("fixed_days", prepared, all_dates, add_weights, penalty_weights, config)
    metrics = result["metrics"]
    row = {
        "评分法": variant["评分法"],
        "评分方案": variant["评分方案"],
        "变体名称": variant["变体名称"],
        "参数名称": variant["参数名称"],
        "退出配置": json.dumps(variant["退出配置"], ensure_ascii=False),
        "候选配置": json.dumps(candidate_cfg, ensure_ascii=False),
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
    row["round_trips"] = result["round_trips"]
    row["trades"] = result["trades"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="趋势主线修复实验：重构候选池、固定持有退出、结构恢复再入场")
    parser.add_argument("--data-dir", default="/Users/lidongyang/Desktop/Qstrategy/data/20260226")
    parser.add_argument("--weighted-scorecard-root", default="/Users/lidongyang/Desktop/Qstrategy/results/formal_scorecard_trend_v2_pool_weighted")
    parser.add_argument("--binary-scorecard-root", default="/Users/lidongyang/Desktop/Qstrategy/results/formal_scorecard_trend_v2_pool_binary")
    parser.add_argument("--factor-opt-summary", default="/Users/lidongyang/Desktop/Qstrategy/results/factor_optimization_experiment_v1/汇总结果.json")
    parser.add_argument("--output-root", default="/Users/lidongyang/Desktop/Qstrategy/results/trend_rebuild_experiment_v1")
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
    result_path = output_root / "趋势主线重构实验结果.csv"

    _写状态(status_path, {"阶段": "加载行情数据", "完成组数": 0})
    stock_data, all_dates = load_price_directory(args.data_dir)
    _写状态(status_path, {"阶段": "预处理因子", "股票数": len(stock_data), "完成组数": 0})
    prepared = _prepare_stock_frames(stock_data)

    weighted_add, weighted_penalty, weighted_best = _load_best_variant_scorecard(
        "加权法",
        Path(args.weighted_scorecard_root),
        Path(args.factor_opt_summary),
    )
    binary_add, binary_penalty, binary_best = _load_best_variant_scorecard(
        "二值法",
        Path(args.binary_scorecard_root),
        Path(args.factor_opt_summary),
    )

    tasks: list[dict] = []
    for method_name, add_weights, penalty_weights, best in [
        ("加权法", weighted_add, weighted_penalty, weighted_best),
        ("二值法", binary_add, binary_penalty, binary_best),
    ]:
        for variant in _variant_specs(best):
            tasks.append(
                {
                    "评分法": method_name,
                    "评分方案": str(best["方案名称"]),
                    "参数名称": str(best["参数名称"]),
                    "变体名称": variant["变体名称"],
                    "退出配置": variant["退出配置"],
                    "候选配置": variant["候选配置"],
                    "加分": add_weights,
                    "扣分": penalty_weights,
                }
            )

    rows: list[dict] = []
    total = len(tasks)
    for idx, task in enumerate(tqdm(tasks, desc="趋势主线重构实验", unit="组"), start=1):
        _写状态(
            status_path,
            {
                "阶段": "运行趋势主线重构实验",
                "完成组数": idx - 1,
                "总组数": total,
                "评分法": task["评分法"],
                "评分方案": task["评分方案"],
                "变体名称": task["变体名称"],
            },
        )
        row = _run_single(prepared, all_dates, task["加分"], task["扣分"], task, args)
        rows.append(row)
        export_rows = [{k: v for k, v in item.items() if k not in {"round_trips", "trades"}} for item in rows]
        with result_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(export_rows[0].keys()))
            writer.writeheader()
            writer.writerows(export_rows)

    ranking = sorted(rows, key=lambda item: (item["年化收益"], item["夏普比率"], item["综合评分"]), reverse=True)
    best = ranking[0]
    best_dir = output_root / "最优组明细"
    best_dir.mkdir(parents=True, exist_ok=True)
    with (best_dir / "完整轮次交易.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["持仓编号", "股票代码", "首次买入日期", "最终卖出日期", "完整持有收益率", "持有天数", "最终卖出原因"])
        writer.writeheader()
        for item in best["round_trips"]:
            writer.writerow(item)
    with (best_dir / "逐笔交易.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["position_id", "code", "entry_date", "exit_date", "entry_price", "exit_price", "shares", "reason"])
        writer.writeheader()
        for item in best["trades"]:
            writer.writerow(item)

    summary = {
        "总组数": total,
        "收益最优": {k: v for k, v in best.items() if k not in {"round_trips", "trades"}},
        "全部结果": [{k: v for k, v in item.items() if k not in {"round_trips", "trades"}} for item in ranking],
    }
    (output_root / "汇总结果.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _写状态(status_path, {"阶段": "已完成", "完成组数": total, "总组数": total, "最优变体": best["变体名称"], "最优评分法": best["评分法"]})


if __name__ == "__main__":
    main()
