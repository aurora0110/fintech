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
from utils.multi_factor_research.factor_calculator import prepare_factor_frame
from utils.multi_factor_research.run_weighted_portfolio_backtest import (
    PortfolioConfig,
    _run_model,
)


def _write_state(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _score(row: dict) -> float:
    trades = float(row["交易次数"])
    if trades <= 0:
        return -1e9
    annual = float(row["年化收益"])
    mdd = float(row["最大回撤"])
    sharpe = float(row["夏普比率"])
    avg_round = float(row["平均持有期间收益率"])
    return annual * 100 - mdd * 25 + sharpe * 12 + avg_round * 10


def _weights() -> tuple[dict[str, float], dict[str, float]]:
    add = {
        "first_pullback_trend_after_cross_factor": 8.0,
        "first_j_buy_after_cross_factor": 7.0,
        "low_volume_pullback_factor": 6.0,
        "rsi_bull_factor": 4.0,
        "daily_ma_bull_factor": 3.0,
        "staged_volume_burst_factor": 2.0,
    }
    penalty = {
        "flat_trend_slope_penalty": 8.0,
        "box_oscillation_penalty": 7.0,
        "extreme_bull_run_penalty": 6.0,
        "bearish_volume_penalty": 5.0,
        "price_amplitude_factor": 4.0,
    }
    return add, penalty


def _prepare_stock_frames_with_progress(stock_data: dict, state_path: Path, total_tasks: int) -> dict:
    prepared: dict = {}
    total = len(stock_data)
    for idx, (code, df) in enumerate(stock_data.items(), start=1):
        raw = df.reset_index().copy()
        factor_df = prepare_factor_frame(raw, burst_window=20)
        factor_df["limit_pct"] = raw["limit_pct"].to_numpy()
        factor_df["is_suspended"] = raw["is_suspended"].to_numpy()
        factor_df = factor_df.set_index("date")
        prepared[code] = factor_df
        if idx == 1 or idx % 100 == 0 or idx == total:
            _write_state(
                state_path,
                {
                    "阶段": "预处理因子",
                    "股票数": total,
                    "已预处理股票数": idx,
                    "完成组数": 0,
                    "总组数": total_tasks,
                },
            )
    return prepared


def _tasks() -> list[dict]:
    tasks: list[dict] = []

    # 1. 固定候选池，测不同退出/失败退出
    pool_cfg = {
        "use_trend_start_pool": True,
        "trend_pool_mode": "repair_v3",
        "rebuilt_min_confirmation_hits": 1,
        "rebuilt_min_support_hits": 1,
        "sideways_mode": "只降分",
        "sideways_score_penalty_scale": 4.0,
    }
    for name, exit_profile in [
        ("持有5天_快失败", {
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
        }),
        ("持有10天_快失败", {
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
        }),
        ("持有10天_中失败", {
            "固定持有模式": "到期全卖",
            "固定持有天数": 10,
            "结构退出最长持有天数": 10,
            "初始止损系数": 0.95,
            "启动失败观察天数": 4,
            "启动失败最小涨幅": 0.02,
            "第二层启动失败观察天数": 0,
            "第二层启动失败最小涨幅": 0.0,
            "保本触发涨幅": 0.03,
            "保本止损系数": 1.0,
        }),
        ("持有10天_慢失败", {
            "固定持有模式": "到期全卖",
            "固定持有天数": 10,
            "结构退出最长持有天数": 10,
            "初始止损系数": 0.95,
            "启动失败观察天数": 5,
            "启动失败最小涨幅": 0.03,
            "第二层启动失败观察天数": 0,
            "第二层启动失败最小涨幅": 0.0,
            "保本触发涨幅": 0.03,
            "保本止损系数": 1.0,
        }),
        ("止盈5pct_快失败", {
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
        }),
    ]:
        tasks.append({
            "实验类别": "固定候选池测不同退出",
            "变体名称": name,
            "配置": pool_cfg | {"exit_profile": exit_profile},
            "策略": "fixed_days" if "持有" in name else "fixed_take_profit",
        })

    # 2. 固定退出，测不同候选池纯度
    exit_profile = {
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
    }
    for name, pool_cfg in [
        ("repair_v2_泛修复", {
            "use_trend_start_pool": True,
            "trend_pool_mode": "repair_v2",
            "rebuilt_min_confirmation_hits": 1,
            "rebuilt_min_support_hits": 1,
            "sideways_mode": "只降分",
            "sideways_score_penalty_scale": 4.0,
        }),
        ("repair_v3_第一次修复", {
            "use_trend_start_pool": True,
            "trend_pool_mode": "repair_v3",
            "rebuilt_min_confirmation_hits": 1,
            "rebuilt_min_support_hits": 1,
            "sideways_mode": "只降分",
            "sideways_score_penalty_scale": 4.0,
        }),
        ("repair_v3_支撑更强", {
            "use_trend_start_pool": True,
            "trend_pool_mode": "repair_v3",
            "rebuilt_min_confirmation_hits": 1,
            "rebuilt_min_support_hits": 2,
            "sideways_mode": "只降分",
            "sideways_score_penalty_scale": 4.0,
        }),
        ("repair_v3_确认更强", {
            "use_trend_start_pool": True,
            "trend_pool_mode": "repair_v3",
            "rebuilt_min_confirmation_hits": 2,
            "rebuilt_min_support_hits": 1,
            "sideways_mode": "只降分",
            "sideways_score_penalty_scale": 4.0,
        }),
        ("repair_v3_确认支撑都强", {
            "use_trend_start_pool": True,
            "trend_pool_mode": "repair_v3",
            "rebuilt_min_confirmation_hits": 2,
            "rebuilt_min_support_hits": 2,
            "sideways_mode": "只降分",
            "sideways_score_penalty_scale": 4.0,
        }),
    ]:
        tasks.append({
            "实验类别": "固定退出测不同候选池",
            "变体名称": name,
            "配置": pool_cfg | {"exit_profile": exit_profile},
            "策略": "fixed_days",
        })
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="修复型负收益原因归因实验")
    parser.add_argument("--data-dir", default="/Users/lidongyang/Desktop/Qstrategy/data/20260226")
    parser.add_argument("--output-root", default="/Users/lidongyang/Desktop/Qstrategy/results/repair_root_cause_experiment_v1")
    parser.add_argument("--initial-capital", type=float, default=10_000_000.0)
    parser.add_argument("--max-positions", type=int, default=10)
    args = parser.parse_args()

    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "状态.json"
    result_path = out / "修复型归因实验结果.csv"
    summary_path = out / "汇总结果.json"

    tasks = _tasks()
    _write_state(state_path, {"阶段": "加载行情数据", "完成组数": 0, "总组数": len(tasks)})
    stock_data, all_dates = load_price_directory(args.data_dir)
    _write_state(state_path, {"阶段": "预处理因子", "股票数": len(stock_data), "完成组数": 0, "总组数": len(tasks)})
    prepared = _prepare_stock_frames_with_progress(stock_data, state_path, len(tasks))
    add, penalty = _weights()

    fieldnames = [
        "实验类别", "变体名称", "策略", "年化收益", "最大回撤", "夏普比率", "资金倍数", "最终资金",
        "交易次数", "平均持有期间收益率", "盈利轮次占比", "盈利轮次平均收益率", "亏损轮次平均收益率", "综合评分"
    ]
    rows: list[dict] = []
    with result_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, task in enumerate(tqdm(tasks, desc="修复型归因实验", unit="组"), start=1):
            _write_state(
                state_path,
                {
                    "阶段": "运行修复型归因实验",
                    "完成组数": idx - 1,
                    "总组数": len(tasks),
                    "实验类别": task["实验类别"],
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
                use_trend_start_pool=task["配置"]["use_trend_start_pool"],
                trend_pool_mode=task["配置"]["trend_pool_mode"],
                rebuilt_min_confirmation_hits=task["配置"]["rebuilt_min_confirmation_hits"],
                rebuilt_min_support_hits=task["配置"]["rebuilt_min_support_hits"],
                sideways_mode=task["配置"]["sideways_mode"],
                sideways_score_penalty_scale=task["配置"]["sideways_score_penalty_scale"],
                exit_profile=task["配置"]["exit_profile"],
            )
            result = _run_model(task["策略"], prepared, all_dates, add, penalty, config)
            row = {
                "实验类别": task["实验类别"],
                "变体名称": task["变体名称"],
                "策略": task["策略"],
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
            row["综合评分"] = _score(row)
            writer.writerow(row)
            f.flush()
            rows.append(row)

    best = max(rows, key=_score) if rows else None
    summary_path.write_text(json.dumps({"总组数": len(rows), "最优结果": best}, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_state(state_path, {"阶段": "已完成", "完成组数": len(rows), "总组数": len(tasks), "最优变体": best["变体名称"] if best else ""})


if __name__ == "__main__":
    main()
