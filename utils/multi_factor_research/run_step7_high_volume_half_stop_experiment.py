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
from utils.multi_factor_research.run_factor_optimization_experiment import _variant_scorecards
from utils.multi_factor_research.run_weighted_portfolio_backtest import (
    PortfolioConfig,
    _load_formal_scorecard,
    _prepare_stock_frames,
    _run_model,
)


def _写状态(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _综合评分(row: dict) -> float:
    return float(row["年化收益"]) * 100.0 - float(row["最大回撤"]) * 35.0 + float(row["夏普比率"]) * 10.0


def _resolve_best_scorecard(factor_opt_summary_path: Path, weighted_root: Path, binary_root: Path) -> tuple[str, dict[str, float], dict[str, float], dict]:
    payload = json.loads(factor_opt_summary_path.read_text(encoding="utf-8"))
    candidates = [payload["加权法"]["综合最优"], payload["二值法"]["综合最优"]]
    best = sorted(
        candidates,
        key=lambda item: (float(item["年化收益"]), float(item["夏普比率"]), -float(item["最大回撤"])),
        reverse=True,
    )[0]
    method_name = str(best["评分法"])
    if method_name == "加权法":
        add_weights, penalty_weights = _load_formal_scorecard(weighted_root)
    else:
        add_weights, penalty_weights = _load_formal_scorecard(binary_root)
    variants = _variant_scorecards(method_name, add_weights, penalty_weights)
    for item in variants:
        if item["名称"] == str(best["方案名称"]):
            return method_name, dict(item["加分"]), dict(item["扣分"]), best
    raise RuntimeError(f"未找到评分卡方案: {best['评分法']} / {best['方案名称']}")


def _candidate_cfg_from_best(best: dict) -> dict:
    return {
        "use_trend_start_pool": True,
        "min_confirmation_hits": 1,
        "min_support_hits": 1,
        "sideways_mode": str(best["横盘模式"]),
        "sideways_filter_threshold": float(best["横盘过滤阈值"]),
        "sideways_score_penalty_scale": float(best["横盘降分系数"]),
    }


def _parse_strategy_profiles(problem_attr_csv: Path, method_name: str) -> dict[str, dict]:
    df = pd.read_csv(problem_attr_csv, encoding="utf-8-sig")
    sub = df[
        (df["实验类别"] == "固定候选池测不同卖出")
        & (df["评分法"] == method_name)
        & (df["候选池"] == "当前最优候选环境")
    ].copy()
    profiles: dict[str, dict] = {}
    for label in ["fixed_take_profit", "fixed_days", "tiered"]:
        row = sub[sub["策略"] == label].iloc[0]
        profiles[label] = {
            "参数名称": str(row["参数名称"]),
            "参数": ast.literal_eval(str(row["参数"])),
        }
    return profiles


def _make_high_volume_profile(base_profile: dict, high_mode: str) -> dict:
    profile = dict(base_profile)
    profile["名称"] = f"高位放量卖出_止损95_{'趋势偏离' if high_mode == 'trend_bias' else '均线偏离'}"
    profile["高位定义"] = high_mode
    profile["最长持有天数"] = int(profile.get("结构退出最长持有天数", profile.get("最长持有天数", 60)))
    return profile


def _make_config(
    exit_profile: dict,
    candidate_cfg: dict,
    initial_capital: float,
    max_positions: int,
    buy_mode: str,
    replacement_threshold: float,
    min_hold_days_for_replace: int,
    max_daily_replacements: int,
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
        sideways_mode=str(candidate_cfg["sideways_mode"]),
        sideways_filter_threshold=float(candidate_cfg["sideways_filter_threshold"]),
        sideways_score_penalty_scale=float(candidate_cfg["sideways_score_penalty_scale"]),
    )


def _half_stop_positive_ratio(trades: pd.DataFrame, round_trips: pd.DataFrame) -> float:
    if trades.empty or round_trips.empty:
        return 0.0
    partial_ids = set(
        trades[trades["reason"].astype(str).str.endswith("次日卖半仓", na=False)]["position_id"].dropna().astype(int).tolist()
    )
    if not partial_ids:
        return 0.0
    sub = round_trips[round_trips["持仓编号"].isin(partial_ids)]
    if sub.empty:
        return 0.0
    return float((sub["完整持有收益率"] > 0).mean())


def _collect_result_row(method_name: str, scorecard_name: str, label: str, variant_name: str, high_mode: str, result: dict) -> dict:
    trades = pd.DataFrame(result["trades"])
    round_trips = pd.DataFrame(result["round_trips"])
    metrics = result["metrics"]
    high_volume_count = int(trades["reason"].astype(str).str.startswith("高位放量卖出_", na=False).sum()) if not trades.empty else 0
    half_stop_partial = int(trades["reason"].astype(str).str.endswith("次日卖半仓", na=False).sum()) if not trades.empty else 0
    half_stop_liq = int((trades["reason"] == "半仓止损后破低清仓").sum()) if not trades.empty else 0
    row = {
        "评分法": method_name,
        "方案名称": scorecard_name,
        "策略": label,
        "版本": variant_name,
        "高位定义": high_mode,
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
        "高位放量卖出次数": high_volume_count,
        "半仓止损触发次数": half_stop_partial,
        "半仓止损后破低清仓次数": half_stop_liq,
        "半仓止损后剩余仓位转盈比例": _half_stop_positive_ratio(trades, round_trips),
    }
    row["综合评分"] = _综合评分(row)
    return row


def _write_trade_exports(run_dir: Path, result: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result["trades"]).to_csv(run_dir / "trades.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(result["round_trips"]).to_csv(run_dir / "完整轮次交易.csv", index=False, encoding="utf-8-sig")
    result["equity_curve"].rename("equity").to_csv(run_dir / "equity_curve.csv", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="第7步实验：高位放量卖出与半仓止损实验")
    parser.add_argument("--data-dir", default="/Users/lidongyang/Desktop/Qstrategy/data/20260226")
    parser.add_argument("--factor-opt-summary", default="/Users/lidongyang/Desktop/Qstrategy/results/factor_optimization_experiment_v1/汇总结果.json")
    parser.add_argument("--problem-attr-csv", default="/Users/lidongyang/Desktop/Qstrategy/results/problem_attribution_experiment_v1/问题归因实验结果.csv")
    parser.add_argument("--weighted-scorecard-root", default="/Users/lidongyang/Desktop/Qstrategy/results/formal_scorecard_trend_v2_pool_weighted")
    parser.add_argument("--binary-scorecard-root", default="/Users/lidongyang/Desktop/Qstrategy/results/formal_scorecard_trend_v2_pool_binary")
    parser.add_argument("--output-root", default="/Users/lidongyang/Desktop/Qstrategy/results/step7_high_volume_half_stop_v1")
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
    result_path = output_root / "第7步实验结果.csv"

    _写状态(status_path, {"阶段": "读取最优评分卡与参数", "完成组数": 0})
    method_name, add_weights, penalty_weights, best = _resolve_best_scorecard(
        Path(args.factor_opt_summary),
        Path(args.weighted_scorecard_root),
        Path(args.binary_scorecard_root),
    )
    candidate_cfg = _candidate_cfg_from_best(best)
    strategy_profiles = _parse_strategy_profiles(Path(args.problem_attr_csv), method_name)

    _写状态(status_path, {"阶段": "加载行情数据", "完成组数": 0, "评分法": method_name, "方案名称": best["方案名称"]})
    stock_data, all_dates = load_price_directory(args.data_dir)
    prepared = _prepare_stock_frames(stock_data)

    tasks: list[dict] = []
    for label in ["fixed_take_profit", "fixed_days", "tiered"]:
        for enable_half_stop in [False, True]:
            tasks.append(
                {
                    "策略": label,
                    "版本": "半仓止损" if enable_half_stop else "原始止损",
                    "高位定义": "",
                    "参数": dict(strategy_profiles[label]["参数"]),
                    "参数名称": strategy_profiles[label]["参数名称"],
                    "启用半仓止损": enable_half_stop,
                }
            )
    high_base = dict(strategy_profiles["fixed_days"]["参数"])
    for high_mode in ["trend_bias", "ma_bias"]:
        for enable_half_stop in [False, True]:
            tasks.append(
                {
                    "策略": "high_volume_exit",
                    "版本": "半仓止损" if enable_half_stop else "原始止损",
                    "高位定义": high_mode,
                    "参数": _make_high_volume_profile(high_base, high_mode),
                    "参数名称": f"高位放量卖出_{high_mode}",
                    "启用半仓止损": enable_half_stop,
                }
            )

    rows: list[dict] = []
    payload_cache: dict[tuple[str, str, str], dict] = {}
    total = len(tasks)
    for done, task in enumerate(tqdm(tasks, desc="第7步实验", unit="组"), start=1):
        profile = dict(task["参数"])
        profile["启用半仓止损"] = bool(task["启用半仓止损"])
        _写状态(
            status_path,
            {
                "阶段": "运行第7步实验",
                "完成组数": done - 1,
                "总组数": total,
                "评分法": method_name,
                "策略": task["策略"],
                "版本": task["版本"],
                "高位定义": task["高位定义"],
            },
        )
        config = _make_config(
            profile,
            candidate_cfg,
            args.initial_capital,
            args.max_positions,
            args.buy_mode,
            args.replacement_threshold,
            args.min_hold_days_for_replace,
            args.max_daily_replacements,
        )
        result = _run_model(task["策略"], prepared, all_dates, add_weights, penalty_weights, config)
        row = _collect_result_row(method_name, best["方案名称"], task["策略"], task["版本"], task["高位定义"], result)
        row["参数名称"] = task["参数名称"]
        row["参数"] = str(profile)
        rows.append(row)
        payload_cache[(task["策略"], task["版本"], task["高位定义"])] = result

    result_df = pd.DataFrame(rows)
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")

    baseline_row = result_df[
        (result_df["策略"] == "fixed_days")
        & (result_df["版本"] == "原始止损")
    ].sort_values(["年化收益", "夏普比率", "综合评分"], ascending=[False, False, False]).iloc[0].to_dict()
    best_row = result_df.sort_values(["年化收益", "夏普比率", "综合评分"], ascending=[False, False, False]).iloc[0].to_dict()

    baseline_key = (str(baseline_row["策略"]), str(baseline_row["版本"]), str(baseline_row["高位定义"]))
    best_key = (str(best_row["策略"]), str(best_row["版本"]), str(best_row["高位定义"]))
    _write_trade_exports(output_root / "基线组明细", payload_cache[baseline_key])
    _write_trade_exports(output_root / "最优组明细", payload_cache[best_key])

    summary = {
        "评分法": method_name,
        "方案名称": best["方案名称"],
        "候选池": "当前最优候选环境",
        "横盘模式": candidate_cfg["sideways_mode"],
        "横盘过滤阈值": candidate_cfg["sideways_filter_threshold"],
        "横盘降分系数": candidate_cfg["sideways_score_penalty_scale"],
        "基线组": baseline_row,
        "最优组": best_row,
    }
    (output_root / "汇总结果.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _写状态(status_path, {"阶段": "已完成", "完成组数": total, "总组数": total, "评分法": method_name, "方案名称": best["方案名称"]})


if __name__ == "__main__":
    main()
