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
    _run_model,
)
from utils.multi_factor_research.factor_calculator import prepare_factor_frame


def _综合评分(row: dict) -> float:
    trades = float(row["交易次数"])
    annual = float(row["年化收益"])
    mdd = float(row["最大回撤"])
    sharpe = float(row["夏普比率"])
    avg_round = float(row["平均持有期间收益率"])
    if trades <= 0:
        return -1e9
    trade_penalty = 0.0
    if trades < 40:
        trade_penalty = (40 - trades) * 0.15
    return annual * 100.0 - mdd * 30.0 + sharpe * 12.0 + avg_round * 20.0 - trade_penalty


def _写状态(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _prepare_stock_frames_with_status(stock_data: dict, status_path: Path) -> dict:
    prepared: dict = {}
    total = len(stock_data)
    for idx, (code, df) in enumerate(tqdm(stock_data.items(), desc="预处理因子帧", unit="stock"), start=1):
        raw = df.reset_index().copy()
        factor_df = prepare_factor_frame(raw, burst_window=20)
        factor_df["limit_pct"] = raw["limit_pct"].to_numpy()
        factor_df["is_suspended"] = raw["is_suspended"].to_numpy()
        factor_df = factor_df.set_index("date")
        prepared[code] = factor_df
        if idx == 1 or idx % 250 == 0 or idx == total:
            _写状态(
                status_path,
                {
                    "阶段": "预处理因子",
                    "股票数": total,
                    "已预处理股票数": idx,
                    "完成组数": 0,
                },
            )
    return prepared


def _候选配置(
    *,
    重构确认最少命中数: int = 2,
    重构支撑最少命中数: int = 3,
    横盘降分系数: float = 4.0,
    再入场模式: str = "bull_bear_only",
) -> dict:
    return {
        "趋势候选池启用": True,
        "趋势候选池模式": "rebuilt_v1",
        "启动确认最少命中数": 1,
        "支撑最少命中数": 1,
        "重构确认最少命中数": 重构确认最少命中数,
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
    固定持有天数: int = 40,
    结构退出最长持有天数: int = 60,
    利润保护模式: str = "趋势线偏离",
    启动失败观察天数: int = 5,
    启动失败最小涨幅: float = 0.03,
    强票延长持有启用: bool = False,
    强票延长额外天数: int = 0,
    强票延长最低浮盈: float = 0.08,
) -> dict:
    return {
        "名称": 名称,
        "固定持有天数": 固定持有天数,
        "初始止损系数": 0.95,
        "固定持有模式": "到期后结构退出",
        "结构退出最长持有天数": 结构退出最长持有天数,
        "利润保护模式": 利润保护模式,
        "启动失败观察天数": 启动失败观察天数,
        "启动失败最小涨幅": 启动失败最小涨幅,
        "第二层启动失败观察天数": 0,
        "第二层启动失败最小涨幅": 0.0,
        "保本触发涨幅": 0.05,
        "保本止损系数": 1.0,
        "强票延长持有启用": 强票延长持有启用,
        "强票延长额外天数": 强票延长额外天数,
        "强票延长最低浮盈": 强票延长最低浮盈,
        "强票延长要求利润保护": True,
        "强票延长要求站上趋势线": True,
        "强票延长要求站上多空线": True,
    }


def _方案列表() -> list[dict]:
    return [
        {
            "变体名称": "主基线3.0_支撑3",
            "候选配置": _候选配置(),
            "退出配置": _退出配置(名称="固定持有40_趋势线偏离_支撑3"),
            "组合配置": {"最大持仓数": 10},
        },
        {
            "变体名称": "确认3_支撑3",
            "候选配置": _候选配置(重构确认最少命中数=3, 重构支撑最少命中数=3),
            "退出配置": _退出配置(名称="固定持有40_趋势线偏离_确认3支撑3"),
            "组合配置": {"最大持仓数": 10},
        },
        {
            "变体名称": "确认2_支撑4",
            "候选配置": _候选配置(重构确认最少命中数=2, 重构支撑最少命中数=4),
            "退出配置": _退出配置(名称="固定持有40_趋势线偏离_确认2支撑4"),
            "组合配置": {"最大持仓数": 10},
        },
        {
            "变体名称": "持有35天_支撑3",
            "候选配置": _候选配置(),
            "退出配置": _退出配置(名称="固定持有35_趋势线偏离_支撑3", 固定持有天数=35),
            "组合配置": {"最大持仓数": 10},
        },
        {
            "变体名称": "持有45天_支撑3",
            "候选配置": _候选配置(),
            "退出配置": _退出配置(名称="固定持有45_趋势线偏离_支撑3", 固定持有天数=45),
            "组合配置": {"最大持仓数": 10},
        },
        {
            "变体名称": "强票延长15天_支撑3",
            "候选配置": _候选配置(),
            "退出配置": _退出配置(
                名称="固定持有40_趋势线偏离_强票延长15",
                强票延长持有启用=True,
                强票延长额外天数=15,
                强票延长最低浮盈=0.08,
            ),
            "组合配置": {"最大持仓数": 10},
        },
        {
            "变体名称": "强票延长20天_支撑3",
            "候选配置": _候选配置(),
            "退出配置": _退出配置(
                名称="固定持有40_趋势线偏离_强票延长20",
                强票延长持有启用=True,
                强票延长额外天数=20,
                强票延长最低浮盈=0.08,
            ),
            "组合配置": {"最大持仓数": 10},
        },
        {
            "变体名称": "结构恢复再入场_支撑3",
            "候选配置": _候选配置(再入场模式="structure_recovery"),
            "退出配置": _退出配置(名称="固定持有40_趋势线偏离_结构恢复再入场"),
            "组合配置": {"最大持仓数": 10},
        },
        {
            "变体名称": "横盘降分增强6_支撑3",
            "候选配置": _候选配置(横盘降分系数=6.0),
            "退出配置": _退出配置(名称="固定持有40_趋势线偏离_横盘降分6"),
            "组合配置": {"最大持仓数": 10},
        },
        {
            "变体名称": "持仓集中8只_支撑3",
            "候选配置": _候选配置(),
            "退出配置": _退出配置(名称="固定持有40_趋势线偏离_集中8"),
            "组合配置": {"最大持仓数": 8},
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="围绕新主基线3.0继续做收益率提升实验V2")
    parser.add_argument("--data-dir", default="/Users/lidongyang/Desktop/Qstrategy/data/20260226")
    parser.add_argument("--weighted-scorecard-root", default="/Users/lidongyang/Desktop/Qstrategy/results/formal_scorecard_trend_v2_pool_weighted")
    parser.add_argument("--factor-opt-summary", default="/Users/lidongyang/Desktop/Qstrategy/results/factor_optimization_experiment_v1/汇总结果.json")
    parser.add_argument("--output-root", default="/Users/lidongyang/Desktop/Qstrategy/results/trend_yield_boost_v2")
    parser.add_argument("--initial-capital", type=float, default=10_000_000.0)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "状态.json"
    result_path = output_root / "收益率提升实验结果.csv"
    summary_path = output_root / "汇总结果.json"

    _写状态(status_path, {"阶段": "加载行情数据", "完成组数": 0})
    stock_data, all_dates = load_price_directory(args.data_dir)
    _写状态(status_path, {"阶段": "预处理因子", "股票数": len(stock_data), "已预处理股票数": 0, "完成组数": 0})
    prepared = _prepare_stock_frames_with_status(stock_data, status_path)

    add_weights, penalty_weights, _ = _load_best_variant_scorecard(
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
        "组合配置",
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

    variants = _方案列表()
    rows: list[dict] = []
    with result_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, variant in enumerate(tqdm(variants, desc="收益率提升实验V2", unit="组"), start=1):
            _写状态(
                status_path,
                {
                    "阶段": "运行收益率提升实验V2",
                    "完成组数": idx - 1,
                    "总组数": len(variants),
                    "评分法": "加权法",
                    "评分方案": "修复型转扣分",
                    "变体名称": variant["变体名称"],
                },
            )
            portfolio = PortfolioConfig(
                initial_capital=args.initial_capital,
                max_positions=int(variant["组合配置"]["最大持仓数"]),
                buy_mode="strict_full",
                replacement_threshold=0.03,
                min_hold_days_for_replace=5,
                max_daily_replacements=1,
                use_trend_start_pool=bool(variant["候选配置"]["趋势候选池启用"]),
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
            portfolio = PortfolioConfig(**{**portfolio.__dict__, "exit_profile": variant["退出配置"]})
            result = _run_model("fixed_days", prepared, all_dates, add_weights, penalty_weights, portfolio)
            row = {
                "评分法": "加权法",
                "评分方案": "修复型转扣分",
                "变体名称": variant["变体名称"],
                "退出配置": json.dumps(variant["退出配置"], ensure_ascii=False),
                "候选配置": json.dumps(variant["候选配置"], ensure_ascii=False),
                "组合配置": json.dumps(variant["组合配置"], ensure_ascii=False),
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

    valid_rows = [row for row in rows if int(row["交易次数"]) > 0]
    rows_sorted = sorted(valid_rows if valid_rows else rows, key=_综合评分, reverse=True)
    summary = {
        "总组数": len(rows),
        "有效组数": len(valid_rows),
        "综合最优": rows_sorted[0] if rows_sorted else None,
        "收益最优": max(valid_rows if valid_rows else rows, key=lambda r: float(r["年化收益"])) if rows else None,
        "回撤最优": min(valid_rows if valid_rows else rows, key=lambda r: float(r["最大回撤"])) if rows else None,
        "全部结果": rows_sorted,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _写状态(
        status_path,
        {
            "阶段": "已完成",
            "完成组数": len(rows),
            "总组数": len(variants),
            "最优变体": summary["综合最优"]["变体名称"] if summary["综合最优"] else None,
        },
    )


if __name__ == "__main__":
    main()
