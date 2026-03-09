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
from utils.multi_factor_research.audit_factor_direction import (
    _build_candidate_dataset,
    _build_scorecard,
    _calc_factor_rows,
)
from utils.multi_factor_research.factor_calculator import build_prepared_stock_data
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
    return annual * 100.0 - mdd * 25.0 + sharpe * 10.0 + avg_round * 8.0


def _从评分卡提取权重(scorecard_df, method: str) -> tuple[dict[str, float], dict[str, float]]:
    add_df = scorecard_df[scorecard_df["评分方向"] == "加分"].copy()
    penalty_df = scorecard_df[scorecard_df["评分方向"] == "扣分"].copy()
    if method == "二值法":
        add_df["百分制评分"] = add_df["百分制评分"].apply(lambda v: 1 if int(v) > 0 else 0)
        penalty_df["百分制评分"] = penalty_df["百分制评分"].apply(lambda v: 1 if int(v) > 0 else 0)
    add_weights = {
        str(row["因子代码"]): float(row["百分制评分"])
        for _, row in add_df.iterrows()
        if float(row["百分制评分"]) > 0
    }
    penalty_weights = {
        str(row["因子代码"]): float(row["百分制评分"])
        for _, row in penalty_df.iterrows()
        if float(row["百分制评分"]) > 0
    }
    return add_weights, penalty_weights


def _构建修复型评分卡(stock_data: dict[str, object], output_root: Path) -> dict[str, tuple[dict[str, float], dict[str, float]]]:
    audit_root = output_root / "修复型方向审计"
    audit_root.mkdir(parents=True, exist_ok=True)
    prepared_stock_data = build_prepared_stock_data(stock_data, burst_window=20)
    dataset = _build_candidate_dataset(
        prepared_stock_data=prepared_stock_data,
        candidate_j_threshold=-5.0,
        require_trend_above=True,
        mode="repair",
        start_gain_10=0.08,
        start_gain_20=0.12,
        use_trend_start_pool=False,
        min_confirmation_hits=1,
        min_support_hits=1,
    )
    dataset.to_csv(audit_root / "候选信号样本.csv", index=False, encoding="utf-8-sig")
    factor_rows = _calc_factor_rows(dataset, mode="repair")
    factor_rows.to_csv(audit_root / "因子方向分析.csv", index=False, encoding="utf-8-sig")
    scorecard = _build_scorecard(factor_rows)
    scorecard.to_csv(audit_root / "百分制评分卡.csv", index=False, encoding="utf-8-sig")

    summary = {
        "样本数量": int(len(dataset)),
        "加权法加分项数量": int((scorecard["评分方向"] == "加分").sum()),
        "加权法扣分项数量": int((scorecard["评分方向"] == "扣分").sum()),
    }
    (audit_root / "汇总结果.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "加权法": _从评分卡提取权重(scorecard, "加权法"),
        "二值法": _从评分卡提取权重(scorecard, "二值法"),
    }


def _固定止盈配置(止盈涨幅: float, 止损系数: float, 失败天数: int, 失败涨幅: float) -> dict:
    return {
        "名称": f"修复固定止盈_{int(止盈涨幅 * 100)}pct",
        "止盈涨幅": 止盈涨幅,
        "固定止盈模式": "全卖",
        "最长持有天数": 15,
        "初始止损系数": 止损系数,
        "启动失败观察天数": 失败天数,
        "启动失败最小涨幅": 失败涨幅,
        "第二层启动失败观察天数": 0,
        "第二层启动失败最小涨幅": 0.0,
        "保本触发涨幅": 0.03,
        "保本止损系数": 1.0,
    }


def _固定持有配置(持有天数: int, 止损系数: float, 失败天数: int, 失败涨幅: float) -> dict:
    return {
        "名称": f"修复固定持有_{持有天数}天",
        "固定持有天数": 持有天数,
        "固定持有模式": "到期全卖",
        "结构退出最长持有天数": 持有天数,
        "初始止损系数": 止损系数,
        "启动失败观察天数": 失败天数,
        "启动失败最小涨幅": 失败涨幅,
        "第二层启动失败观察天数": 0,
        "第二层启动失败最小涨幅": 0.0,
        "保本触发涨幅": 0.03,
        "保本止损系数": 1.0,
    }


def _分批止盈配置(模式: str, 止损系数: float, 失败天数: int, 失败涨幅: float) -> dict:
    return {
        "名称": f"修复分批止盈_{模式}",
        "分批止盈模式": 模式,
        "最长持有天数": 15 if 模式 == "轻量" else 20,
        "初始止损系数": 止损系数,
        "启动失败观察天数": 失败天数,
        "启动失败最小涨幅": 失败涨幅,
        "第二层启动失败观察天数": 0,
        "第二层启动失败最小涨幅": 0.0,
        "保本触发涨幅": 0.03,
        "保本止损系数": 1.0,
    }


def _方案列表() -> list[dict]:
    止损失败组合 = [
        {"止损系数": 0.97, "失败天数": 3, "失败涨幅": 0.02, "名称": "快止损"},
        {"止损系数": 0.95, "失败天数": 5, "失败涨幅": 0.03, "名称": "标准止损"},
    ]
    tasks: list[dict] = []
    for combo in 止损失败组合:
        for 止盈涨幅 in [0.05, 0.08]:
            tasks.append(
                {
                    "策略": "fixed_take_profit",
                    "变体名称": f"固定止盈_{int(止盈涨幅 * 100)}pct_{combo['名称']}",
                    "参数": _固定止盈配置(止盈涨幅, combo["止损系数"], combo["失败天数"], combo["失败涨幅"]),
                }
            )
        for 持有天数 in [5, 10]:
            tasks.append(
                {
                    "策略": "fixed_days",
                    "变体名称": f"固定持有_{持有天数}天_{combo['名称']}",
                    "参数": _固定持有配置(持有天数, combo["止损系数"], combo["失败天数"], combo["失败涨幅"]),
                }
            )
        for 模式 in ["轻量", "五段"]:
            tasks.append(
                {
                    "策略": "tiered",
                    "变体名称": f"分批止盈_{模式}_{combo['名称']}",
                    "参数": _分批止盈配置(模式, combo["止损系数"], combo["失败天数"], combo["失败涨幅"]),
                }
            )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="修复型模型实验：比较加权因子与二值因子")
    parser.add_argument("--data-dir", default="/Users/lidongyang/Desktop/Qstrategy/data/20260226")
    parser.add_argument("--output-root", default="/Users/lidongyang/Desktop/Qstrategy/results/repair_model_experiment_v1")
    parser.add_argument("--initial-capital", type=float, default=10_000_000.0)
    parser.add_argument("--max-positions", type=int, default=10)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "状态.json"
    result_path = output_root / "修复型实验结果.csv"
    summary_path = output_root / "汇总结果.json"

    _写状态(status_path, {"阶段": "加载行情数据", "完成组数": 0})
    stock_data, all_dates = load_price_directory(args.data_dir)
    _写状态(status_path, {"阶段": "构建修复型评分卡", "股票数": len(stock_data), "完成组数": 0})
    scorecards = _构建修复型评分卡(stock_data, output_root)
    _写状态(status_path, {"阶段": "预处理因子", "股票数": len(stock_data), "完成组数": 0})
    prepared = _prepare_stock_frames(stock_data)

    tasks: list[dict] = []
    for method_name in ["加权法", "二值法"]:
        add_weights, penalty_weights = scorecards[method_name]
        for base in _方案列表():
            tasks.append(
                {
                    "评分法": method_name,
                    "加分": add_weights,
                    "扣分": penalty_weights,
                    **base,
                }
            )

    fieldnames = [
        "评分法",
        "策略",
        "变体名称",
        "参数",
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
    total = len(tasks)
    with result_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, task in enumerate(tqdm(tasks, desc="修复型实验", unit="组"), start=1):
            _写状态(
                status_path,
                {
                    "阶段": "运行修复型实验",
                    "完成组数": idx - 1,
                    "总组数": total,
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
                use_trend_start_pool=False,
                sideways_mode="关闭",
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
                "波动率": float(result["metrics"]["volatility"]),
                "资金倍数": float(result["metrics"]["final_multiple"]),
                "最终资金": float(result["final_equity"]),
                "交易次数": int(result["trade_count"]),
                "完整轮次交易数": int(result["完整轮次交易数"]),
                "平均持有期间收益率": float(result["平均持有期间收益率"]),
                "盈利轮次占比": float(result["盈利轮次占比"]),
                "盈利轮次平均收益率": float(result["盈利轮次平均收益率"]),
                "亏损轮次平均收益率": float(result["亏损轮次平均收益率"]),
            }
            row["综合评分"] = _综合评分(row)
            writer.writerow(row)
            f.flush()
            rows.append(row)

    valid_rows = [r for r in rows if int(r["交易次数"]) > 0]
    summary = {
        "总组数": len(rows),
        "有效组数": len(valid_rows),
        "综合最优": max(valid_rows if valid_rows else rows, key=_综合评分) if rows else None,
        "收益最优": max(valid_rows if valid_rows else rows, key=lambda r: float(r["年化收益"])) if rows else None,
        "回撤最优": min(valid_rows if valid_rows else rows, key=lambda r: float(r["最大回撤"])) if rows else None,
        "全部结果": sorted(valid_rows if valid_rows else rows, key=_综合评分, reverse=True),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _写状态(status_path, {"阶段": "已完成", "完成组数": len(rows), "总组数": total, "最优变体": summary["综合最优"]["变体名称"] if summary["综合最优"] else None})


if __name__ == "__main__":
    main()
