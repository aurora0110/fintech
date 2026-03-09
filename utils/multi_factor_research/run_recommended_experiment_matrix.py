from __future__ import annotations

import argparse
import json
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
    )
    result = _run_model(strategy_label, prepared, all_dates, add_weights, penalty_weights, config)
    metrics = result["metrics"]
    row = {
        "策略": strategy_label,
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


def _fixed_take_profit_profiles() -> list[dict]:
    profiles = []
    for 止盈涨幅 in [0.08, 0.10, 0.12, 0.15, 0.20]:
        for 初始止损系数 in [0.95, 0.96, 0.97]:
            for 利润保护模式 in ["固定", "振幅", "ATR", "趋势线偏离", "历史启动特征"]:
                profiles.append(
                    {
                        "名称": f"固定止盈_{int(止盈涨幅 * 100)}pct_全卖_止损{int(初始止损系数 * 100)}_{利润保护模式}",
                        "止盈涨幅": 止盈涨幅,
                        "初始止损系数": 初始止损系数,
                        "固定止盈模式": "全卖",
                        "最长持有天数": 30,
                        "利润保护模式": 利润保护模式,
                    }
                )
                profiles.append(
                    {
                        "名称": f"固定止盈_{int(止盈涨幅 * 100)}pct_留仓趋势_止损{int(初始止损系数 * 100)}_{利润保护模式}",
                        "止盈涨幅": 止盈涨幅,
                        "初始止损系数": 初始止损系数,
                        "固定止盈模式": "留仓趋势",
                        "最长持有天数": 60,
                        "利润保护模式": 利润保护模式,
                    }
                )
    return profiles


def _fixed_days_profiles() -> list[dict]:
    profiles = []
    for 固定持有天数 in [10, 15, 20, 30, 40]:
        for 初始止损系数 in [0.95, 0.96, 0.97]:
            for 利润保护模式 in ["固定", "振幅", "ATR", "趋势线偏离", "历史启动特征"]:
                profiles.append(
                    {
                        "名称": f"固定持有_{固定持有天数}天_到期全卖_止损{int(初始止损系数 * 100)}_{利润保护模式}",
                        "固定持有天数": 固定持有天数,
                        "初始止损系数": 初始止损系数,
                        "固定持有模式": "到期全卖",
                        "结构退出最长持有天数": 固定持有天数,
                        "利润保护模式": 利润保护模式,
                    }
                )
                profiles.append(
                    {
                        "名称": f"固定持有_{固定持有天数}天_结构退出_止损{int(初始止损系数 * 100)}_{利润保护模式}",
                        "固定持有天数": 固定持有天数,
                        "初始止损系数": 初始止损系数,
                        "固定持有模式": "到期后结构退出",
                        "结构退出最长持有天数": 60,
                        "利润保护模式": 利润保护模式,
                    }
                )
    return profiles


def _tiered_profiles() -> list[dict]:
    profiles = []
    for 初始止损系数 in [0.95, 0.96, 0.97]:
        for 利润保护模式 in ["固定", "振幅", "ATR", "趋势线偏离", "历史启动特征"]:
            profiles.append(
                {
                    "名称": f"分批止盈_五段_止损{int(初始止损系数 * 100)}_{利润保护模式}",
                    "分批止盈模式": "五段",
                    "初始止损系数": 初始止损系数,
                    "最长持有天数": 60,
                    "利润保护模式": 利润保护模式,
                }
            )
            profiles.append(
                {
                    "名称": f"分批止盈_轻量_止损{int(初始止损系数 * 100)}_{利润保护模式}",
                    "分批止盈模式": "轻量",
                    "初始止损系数": 初始止损系数,
                    "最长持有天数": 40,
                    "利润保护模式": 利润保护模式,
                }
            )
    return profiles


def main() -> None:
    parser = argparse.ArgumentParser(description="基于最优评分卡一次跑完整建议实验矩阵")
    parser.add_argument("--data-dir", default="/Users/lidongyang/Desktop/Qstrategy/data/20260226")
    parser.add_argument("--scorecard-root", required=True)
    parser.add_argument("--output-root", default="/Users/lidongyang/Desktop/Qstrategy/results/recommended_experiment_matrix_v1")
    parser.add_argument("--initial-capital", type=float, default=10_000_000.0)
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--buy-mode", choices=["staged", "full", "strict_full"], default="staged")
    parser.add_argument("--replacement-threshold", type=float, default=0.03)
    parser.add_argument("--min-hold-days-for-replace", type=int, default=5)
    parser.add_argument("--max-daily-replacements", type=int, default=1)
    parser.add_argument("--startup-fail-days", type=int, default=0)
    parser.add_argument("--startup-fail-min-gain", type=float, default=0.0)
    parser.add_argument("--profit-protect-trigger", type=float, default=0.0)
    parser.add_argument("--profit-protect-stop-factor", type=float, default=1.0)
    parser.add_argument("--use-trend-start-pool", action="store_true")
    parser.add_argument("--min-confirmation-hits", type=int, default=1)
    parser.add_argument("--min-support-hits", type=int, default=1)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    add_weights, penalty_weights = _load_formal_scorecard(Path(args.scorecard_root))
    stock_data, all_dates = load_price_directory(args.data_dir)
    prepared = _prepare_stock_frames(stock_data)

    tasks: list[tuple[str, dict]] = []
    tasks.extend(("fixed_take_profit", profile) for profile in _fixed_take_profit_profiles())
    tasks.extend(("fixed_days", profile) for profile in _fixed_days_profiles())
    tasks.extend(("tiered", profile) for profile in _tiered_profiles())

    result_path = output_root / "完整实验矩阵结果.csv"
    rows: list[dict] = []
    已完成参数名 = set()
    if result_path.exists():
        existing = pd.read_csv(result_path)
        rows = existing.to_dict(orient="records")
        已完成参数名 = set(existing["参数名称"].astype(str).tolist())

    pending_tasks = [(label, profile) for label, profile in tasks if profile["名称"] not in 已完成参数名]
    for label, profile in tqdm(pending_tasks, desc="完整实验矩阵", unit="组"):
        profile = {
            **profile,
            "启动失败观察天数": args.startup_fail_days,
            "启动失败最小涨幅": args.startup_fail_min_gain,
            "保本触发涨幅": args.profit_protect_trigger,
            "保本止损系数": args.profit_protect_stop_factor,
        }
        row = _run_single(
            prepared,
            all_dates,
            label,
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
        )
        row["参数名称"] = profile["名称"]
        rows.append(row)
        pd.DataFrame(rows).to_csv(result_path, index=False, encoding="utf-8-sig")

    result_df = pd.DataFrame(rows)

    summary = {}
    for label, group_name in [("fixed_take_profit", "固定止盈"), ("fixed_days", "固定持有"), ("tiered", "分批止盈")]:
        sub = result_df[result_df["策略"] == label].copy()
        if sub.empty:
            continue
        best_return = sub.sort_values(["年化收益", "夏普比率", "综合评分"], ascending=[False, False, False]).iloc[0].to_dict()
        best_balanced = sub.sort_values(["综合评分", "年化收益", "夏普比率"], ascending=[False, False, False]).iloc[0].to_dict()
        lowest_drawdown = sub.sort_values(["最大回撤", "综合评分", "年化收益"], ascending=[True, False, False]).iloc[0].to_dict()
        summary[group_name] = {
            "收益最优": best_return,
            "综合最优": best_balanced,
            "回撤最优": lowest_drawdown,
        }

    (output_root / "汇总结果.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
