from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data_loader import load_price_directory
from utils.multi_factor_research.run_step7_high_volume_half_stop_experiment import _resolve_best_scorecard
from utils.multi_factor_research.run_weighted_portfolio_backtest import (
    PortfolioConfig,
    _prepare_stock_frames,
    _run_model,
)


def _写状态(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _综合评分(row: dict) -> float:
    return float(row["年化收益"]) * 100.0 - float(row["最大回撤"]) * 35.0 + float(row["夏普比率"]) * 10.0


def _candidate_cfg_from_best(best: dict) -> dict:
    return {
        "use_trend_start_pool": True,
        "min_confirmation_hits": 1,
        "min_support_hits": 1,
        "sideways_mode": str(best["横盘模式"]),
        "sideways_filter_threshold": float(best["横盘过滤阈值"]),
        "sideways_score_penalty_scale": float(best["横盘降分系数"]),
    }


def _load_step7_baseline(step7_summary_path: Path) -> dict:
    payload = json.loads(step7_summary_path.read_text(encoding="utf-8"))
    return dict(payload["基线组"])


def _make_config(
    exit_profile: dict,
    candidate_cfg: dict,
    initial_capital: float,
    max_positions: int,
    buy_mode: str,
    replacement_threshold: float,
    min_hold_days_for_replace: int,
    max_daily_replacements: int,
    cooldown_count: int,
    cooldown_days: int,
    skip_next_buys: int,
) -> PortfolioConfig:
    return PortfolioConfig(
        initial_capital=initial_capital,
        max_positions=max_positions,
        replacement_threshold=replacement_threshold,
        min_hold_days_for_replace=min_hold_days_for_replace,
        max_daily_replacements=max_daily_replacements,
        buy_mode=buy_mode,
        exit_profile=exit_profile,
        use_trend_start_pool=bool(candidate_cfg["use_trend_start_pool"]),
        min_confirmation_hits=int(candidate_cfg["min_confirmation_hits"]),
        min_support_hits=int(candidate_cfg["min_support_hits"]),
        stock_stop_cooldown_count=cooldown_count,
        stock_stop_cooldown_days=cooldown_days,
        stock_stop_skip_next_buys=skip_next_buys,
        sideways_mode=str(candidate_cfg["sideways_mode"]),
        sideways_filter_threshold=float(candidate_cfg["sideways_filter_threshold"]),
        sideways_score_penalty_scale=float(candidate_cfg["sideways_score_penalty_scale"]),
    )


def _stop_like_reasons() -> set[str]:
    return {"止损卖出", "启动失败卖出", "滴滴止损", "两根跌破多空线止损", "半仓止损后破低清仓"}


def _collect_result_row(task: dict, result: dict) -> dict:
    trades = pd.DataFrame(result["trades"])
    round_trips = pd.DataFrame(result["round_trips"])
    metrics = result["metrics"]
    stop_like = trades[trades["reason"].isin(_stop_like_reasons())] if not trades.empty else pd.DataFrame()
    stop_round_ids = set(stop_like["position_id"].dropna().astype(int).tolist()) if not stop_like.empty else set()
    repeated_stop_stock_count = int(
        stop_like.groupby("code").size().ge(2).sum()
    ) if not stop_like.empty else 0
    row = {
        "评分法": task["评分法"],
        "方案名称": task["方案名称"],
        "策略": task["策略"],
        "版本": task["版本"],
        "连续止损阈值": task["连续止损阈值"],
        "冷却天数": task["冷却天数"],
        "跳过下次买入次数": task["跳过下次买入次数"],
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
        "止损类卖出次数": int(len(stop_like)),
        "出现两次及以上止损的股票数": repeated_stop_stock_count,
        "止损类轮次数": int(len(stop_round_ids)),
    }
    row["综合评分"] = _综合评分(row)
    return row


def _write_exports(run_dir: Path, result: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result["trades"]).to_csv(run_dir / "trades.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(result["round_trips"]).to_csv(run_dir / "完整轮次交易.csv", index=False, encoding="utf-8-sig")
    result["equity_curve"].rename("equity").to_csv(run_dir / "equity_curve.csv", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="第8步实验：个股连续止损冷却方案实验")
    parser.add_argument("--data-dir", default="/Users/lidongyang/Desktop/Qstrategy/data/20260226")
    parser.add_argument("--factor-opt-summary", default="/Users/lidongyang/Desktop/Qstrategy/results/factor_optimization_experiment_v1/汇总结果.json")
    parser.add_argument("--step7-summary", default="/Users/lidongyang/Desktop/Qstrategy/results/step7_high_volume_half_stop_v1/汇总结果.json")
    parser.add_argument("--weighted-scorecard-root", default="/Users/lidongyang/Desktop/Qstrategy/results/formal_scorecard_trend_v2_pool_weighted")
    parser.add_argument("--binary-scorecard-root", default="/Users/lidongyang/Desktop/Qstrategy/results/formal_scorecard_trend_v2_pool_binary")
    parser.add_argument("--output-root", default="/Users/lidongyang/Desktop/Qstrategy/results/step8_stock_cooldown_v1")
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
    result_path = output_root / "第8步实验结果.csv"

    _写状态(status_path, {"阶段": "读取最优评分卡与基线策略", "完成组数": 0})
    method_name, add_weights, penalty_weights, best = _resolve_best_scorecard(
        Path(args.factor_opt_summary),
        Path(args.weighted_scorecard_root),
        Path(args.binary_scorecard_root),
    )
    candidate_cfg = _candidate_cfg_from_best(best)
    step7_baseline = _load_step7_baseline(Path(args.step7_summary))
    baseline_profile = ast.literal_eval(str(step7_baseline["参数"]))
    strategy_label = str(step7_baseline["策略"])
    scorecard_name = str(step7_baseline["方案名称"])

    _写状态(status_path, {"阶段": "加载行情数据", "完成组数": 0, "评分法": method_name, "方案名称": scorecard_name, "策略": strategy_label})
    stock_data, all_dates = load_price_directory(args.data_dir)
    prepared = _prepare_stock_frames(stock_data)

    tasks = [
        {"版本": "基线_无冷却", "连续止损阈值": 0, "冷却天数": 0, "跳过下次买入次数": 0},
        {"版本": "连续2次_跳过1次买入", "连续止损阈值": 2, "冷却天数": 0, "跳过下次买入次数": 1},
        {"版本": "连续2次_冷却10天", "连续止损阈值": 2, "冷却天数": 10, "跳过下次买入次数": 0},
        {"版本": "连续2次_冷却20天", "连续止损阈值": 2, "冷却天数": 20, "跳过下次买入次数": 0},
        {"版本": "连续3次_冷却20天", "连续止损阈值": 3, "冷却天数": 20, "跳过下次买入次数": 0},
    ]

    rows: list[dict] = []
    payload_cache: dict[str, dict] = {}
    total = len(tasks)
    for done, task in enumerate(tqdm(tasks, desc="第8步实验", unit="组"), start=1):
        _写状态(
            status_path,
            {
                "阶段": "运行第8步实验",
                "完成组数": done - 1,
                "总组数": total,
                "评分法": method_name,
                "方案名称": scorecard_name,
                "策略": strategy_label,
                "版本": task["版本"],
            },
        )
        config = _make_config(
            dict(baseline_profile),
            candidate_cfg,
            args.initial_capital,
            args.max_positions,
            args.buy_mode,
            args.replacement_threshold,
            args.min_hold_days_for_replace,
            args.max_daily_replacements,
            int(task["连续止损阈值"]),
            int(task["冷却天数"]),
            int(task["跳过下次买入次数"]),
        )
        result = _run_model(strategy_label, prepared, all_dates, add_weights, penalty_weights, config)
        row = _collect_result_row(
            {
                "评分法": method_name,
                "方案名称": scorecard_name,
                "策略": strategy_label,
                **task,
            },
            result,
        )
        rows.append(row)
        payload_cache[task["版本"]] = result
        pd.DataFrame(rows).to_csv(result_path, index=False, encoding="utf-8-sig")

    df = pd.DataFrame(rows)
    best_row = df.sort_values(["年化收益", "夏普比率", "综合评分"], ascending=[False, False, False]).iloc[0].to_dict()
    base_row = df[df["版本"] == "基线_无冷却"].iloc[0].to_dict()

    _write_exports(output_root / "最优组明细", payload_cache[best_row["版本"]])
    _write_exports(output_root / "基线组明细", payload_cache[base_row["版本"]])

    summary = {
        "评分法": method_name,
        "方案名称": scorecard_name,
        "策略": strategy_label,
        "候选池": "当前最优候选环境",
        "基线组": base_row,
        "最优组": best_row,
    }
    (output_root / "汇总结果.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _写状态(status_path, {"阶段": "已完成", "完成组数": total, "总组数": total, "评分法": method_name, "方案名称": scorecard_name, "策略": strategy_label})


if __name__ == "__main__":
    main()
