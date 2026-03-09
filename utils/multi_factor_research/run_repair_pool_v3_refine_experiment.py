from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from utils.multi_factor_research.run_repair_pool_v3_experiment import (
    build_binary_score_spec,
    build_weighted_score_spec,
)
from utils.multi_factor_research.run_weighted_portfolio_backtest import (
    PortfolioConfig,
    build_portfolio_result,
    build_prepared_stock_data,
    compute_round_trip_metrics,
    ensure_date_column,
    run_portfolio_backtest,
)


RESULT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/results/repair_pool_v3_refine_v1")
STATE_PATH = RESULT_DIR / "状态.json"
RESULT_CSV = RESULT_DIR / "修复候选池3.0精修结果.csv"
SUMMARY_JSON = RESULT_DIR / "汇总结果.json"
DATA_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/data/20260226")


def write_state(**kwargs: Any) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(kwargs, ensure_ascii=False, indent=2), encoding="utf-8")


def score_summary(metrics: dict[str, Any]) -> float:
    trades = max(1, int(metrics.get("完整轮次交易数", 0)))
    annual = float(metrics.get("年化收益", 0.0))
    drawdown = float(metrics.get("最大回撤", 1.0))
    sharpe = float(metrics.get("夏普比率", 0.0))
    avg_trade = float(metrics.get("平均持有期间收益率", 0.0))
    trade_gate = min(1.0, trades / 40.0)
    return (annual * 0.45 - drawdown * 0.25 + sharpe * 0.20 + avg_trade * 0.10) * trade_gate


def build_variants() -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for hold_days in (8, 10, 12):
        for stop_loss in (0.95, 0.96):
            for fail_days, fail_gain in ((3, 0.02), (4, 0.02), (5, 0.03)):
                variants.append(
                    {
                        "变体名称": f"持有{hold_days}天_止损{int(stop_loss*100)}_失败{fail_days}天{int(fail_gain*100)}pct",
                        "持有天数": hold_days,
                        "止损系数": stop_loss,
                        "启动失败天数": fail_days,
                        "启动失败涨幅": fail_gain,
                    }
                )
    return variants


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    variants = build_variants()
    write_state(
        阶段="加载行情数据",
        股票数=0,
        完成组数=0,
        总组数=len(variants),
        评分法="加权法",
        方案名称="修复型候选池3.0精修",
        当前变体="",
    )

    prepared = build_prepared_stock_data(DATA_DIR)
    prepared = {code: ensure_date_column(df.copy()) for code, df in prepared.items()}
    write_state(
        阶段="开始回测",
        股票数=len(prepared),
        完成组数=0,
        总组数=len(variants),
        评分法="加权法",
        方案名称="修复型候选池3.0精修",
        当前变体="",
    )

    score_spec = build_weighted_score_spec()
    rows: list[dict[str, Any]] = []

    for idx, variant in enumerate(tqdm(variants, desc="修复型3.0精修"), start=1):
        write_state(
            阶段="正式回测",
            股票数=len(prepared),
            完成组数=idx - 1,
            总组数=len(variants),
            评分法="加权法",
            方案名称="修复型候选池3.0精修",
            当前变体=variant["变体名称"],
        )

        config = PortfolioConfig(
            initial_capital=10_000_000,
            max_positions=10,
            score_method="weighted",
            score_spec=score_spec,
            use_trend_start_pool=True,
            trend_pool_mode="repair_v3",
            rebuilt_min_confirmation_hits=1,
            rebuilt_min_support_hits=1,
            sideways_mode="只降分",
            sideways_score_penalty_scale=4.0,
            fixed_holding_days=variant["持有天数"],
            fixed_days_mode="到期全卖",
            initial_stop_loss_pct=variant["止损系数"],
            startup_fail_days=variant["启动失败天数"],
            startup_fail_min_gain=variant["启动失败涨幅"],
            breakeven_trigger_gain=0.03,
            breakeven_stop_multiple=1.0,
            profit_protection_mode="none",
        )

        result = run_portfolio_backtest(prepared, config)
        metrics = build_portfolio_result(result)
        metrics.update(compute_round_trip_metrics(result["完整轮次交易"]))
        row = {
            "评分法": "加权法",
            "方案名称": "修复型候选池3.0精修",
            **variant,
            **metrics,
        }
        row["综合评分"] = score_summary(metrics)
        rows.append(row)
        pd.DataFrame(rows).to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")

    result_df = pd.DataFrame(rows)
    best_row = result_df.sort_values("综合评分", ascending=False).iloc[0].to_dict() if not result_df.empty else {}
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "总组数": len(variants),
                "最佳变体": best_row,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_state(
        阶段="已完成",
        股票数=len(prepared),
        完成组数=len(variants),
        总组数=len(variants),
        评分法="加权法",
        方案名称="修复型候选池3.0精修",
        当前变体="",
        最优变体=best_row.get("变体名称", ""),
    )


if __name__ == "__main__":
    main()
