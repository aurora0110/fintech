from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data_loader import load_price_directory
from utils.multi_factor_research.run_factor_optimization_experiment import _variant_scorecards
from utils.multi_factor_research.run_weighted_portfolio_backtest import (
    _load_formal_scorecard,
    _prepare_stock_frames,
    _run_model,
    PortfolioConfig,
)


def _select_best_config(summary: dict) -> dict:
    candidates = [
        summary["加权法"]["综合最优"],
        summary["二值法"]["综合最优"],
    ]
    return sorted(
        candidates,
        key=lambda item: (float(item["年化收益"]), float(item["综合评分"]), -float(item["最大回撤"])),
        reverse=True,
    )[0]


def _resolve_scorecard(best: dict, weighted_root: Path, binary_root: Path) -> tuple[str, dict[str, float], dict[str, float]]:
    method_name = str(best["评分法"])
    variant_name = str(best["方案名称"])
    if method_name == "加权法":
        add_weights, penalty_weights = _load_formal_scorecard(weighted_root)
    else:
        add_weights, penalty_weights = _load_formal_scorecard(binary_root)
    variants = _variant_scorecards(method_name, add_weights, penalty_weights)
    for item in variants:
        if item["名称"] == variant_name:
            return method_name, dict(item["加分"]), dict(item["扣分"])
    raise RuntimeError(f"未找到评分方案: {method_name} / {variant_name}")


def _drawdown_window(equity_curve: pd.Series) -> dict:
    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1.0
    trough_date = drawdown.idxmin()
    trough_value = float(drawdown.loc[trough_date])
    peak_slice = equity_curve.loc[:trough_date]
    peak_date = peak_slice.idxmax()
    peak_value = float(equity_curve.loc[peak_date])
    trough_equity = float(equity_curve.loc[trough_date])

    recovery_date = None
    after_trough = equity_curve.loc[trough_date:]
    recovered = after_trough[after_trough >= peak_value]
    if not recovered.empty:
        recovery_date = recovered.index[0]

    return {
        "回撤起点": pd.Timestamp(peak_date),
        "回撤低点": pd.Timestamp(trough_date),
        "回撤恢复日": pd.Timestamp(recovery_date) if recovery_date is not None else None,
        "起点净值": peak_value,
        "低点净值": trough_equity,
        "最大回撤": abs(trough_value),
        "回撤天数": int((pd.Timestamp(trough_date) - pd.Timestamp(peak_date)).days),
    }


def _longest_losing_streak(round_df: pd.DataFrame) -> dict:
    max_streak = 0
    current = 0
    worst_sum = 0.0
    current_sum = 0.0
    start = None
    end = None
    current_start = None
    for row in round_df.itertuples(index=False):
        ret = float(row.完整持有收益率)
        if ret <= 0:
            if current == 0:
                current_start = pd.Timestamp(row.最终卖出日期)
                current_sum = 0.0
            current += 1
            current_sum += ret
            if current > max_streak or (current == max_streak and current_sum < worst_sum):
                max_streak = current
                worst_sum = current_sum
                start = current_start
                end = pd.Timestamp(row.最终卖出日期)
        else:
            current = 0
            current_sum = 0.0
            current_start = None
    return {
        "最大连续亏损轮次": max_streak,
        "最大连续亏损累计收益率": worst_sum,
        "最大连续亏损开始": start,
        "最大连续亏损结束": end,
    }


def _classify_drawdown(drawdown_rounds: pd.DataFrame) -> str:
    if drawdown_rounds.empty:
        return "回撤区间无完整轮次，无法归因"
    losses = drawdown_rounds[drawdown_rounds["完整持有收益率"] <= 0].copy()
    if losses.empty:
        return "回撤区间内完整轮次整体不亏，回撤更多来自持仓浮亏"
    top5_loss_share = abs(losses.nsmallest(5, "完整持有收益率")["完整持有收益率"].sum()) / abs(losses["完整持有收益率"].sum())
    stop_like = losses["最终卖出原因"].isin(["止损卖出", "启动失败卖出", "滴滴止损", "两根跌破多空线止损"])
    stop_like_ratio = float(stop_like.mean())
    avg_loss = float(losses["完整持有收益率"].mean())
    if top5_loss_share >= 0.55:
        return "最大回撤更像由少数几笔大亏主导"
    if stop_like_ratio >= 0.6 and avg_loss > -0.08:
        return "最大回撤更像由连续止损/连续弱交易慢慢磨损导致"
    if stop_like_ratio >= 0.6:
        return "最大回撤由连续止损与中等幅度亏损共同造成"
    return "最大回撤更像由混合因素造成，既有连续弱交易，也有少数较大亏损"


def main() -> None:
    parser = argparse.ArgumentParser(description="对当前最优趋势配置做大回撤归因")
    parser.add_argument("--data-dir", default="/Users/lidongyang/Desktop/Qstrategy/data/20260226")
    parser.add_argument("--factor-opt-summary", default="/Users/lidongyang/Desktop/Qstrategy/results/factor_optimization_experiment_v1/汇总结果.json")
    parser.add_argument("--weighted-scorecard-root", default="/Users/lidongyang/Desktop/Qstrategy/results/formal_scorecard_trend_v2_pool_weighted")
    parser.add_argument("--binary-scorecard-root", default="/Users/lidongyang/Desktop/Qstrategy/results/formal_scorecard_trend_v2_pool_binary")
    parser.add_argument("--output-root", default="/Users/lidongyang/Desktop/Qstrategy/results/drawdown_attribution_v1")
    parser.add_argument("--initial-capital", type=float, default=10_000_000.0)
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--buy-mode", choices=["staged", "full", "strict_full"], default="full")
    parser.add_argument("--replacement-threshold", type=float, default=0.03)
    parser.add_argument("--min-hold-days-for-replace", type=int, default=5)
    parser.add_argument("--max-daily-replacements", type=int, default=1)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary = json.loads(Path(args.factor_opt_summary).read_text(encoding="utf-8"))
    best = _select_best_config(summary)
    method_name, add_weights, penalty_weights = _resolve_scorecard(
        best,
        Path(args.weighted_scorecard_root),
        Path(args.binary_scorecard_root),
    )

    stock_data, all_dates = load_price_directory(args.data_dir)
    prepared = _prepare_stock_frames(stock_data)

    config = PortfolioConfig(
        initial_capital=args.initial_capital,
        max_positions=args.max_positions,
        replacement_threshold=args.replacement_threshold,
        min_hold_days_for_replace=args.min_hold_days_for_replace,
        max_daily_replacements=args.max_daily_replacements,
        buy_mode=args.buy_mode,
        use_trend_start_pool=True,
        min_confirmation_hits=1,
        min_support_hits=1,
        sideways_mode=str(best["横盘模式"]),
        sideways_filter_threshold=float(best["横盘过滤阈值"]),
        sideways_score_penalty_scale=float(best["横盘降分系数"]),
        exit_profile=dict(best["参数"]),
    )

    result = _run_model("fixed_days", prepared, all_dates, add_weights, penalty_weights, config)
    equity_curve = result["equity_curve"]
    drawdown_meta = _drawdown_window(equity_curve)

    round_df = pd.DataFrame(result["round_trips"])
    if not round_df.empty:
        round_df["首次买入日期"] = pd.to_datetime(round_df["首次买入日期"])
        round_df["最终卖出日期"] = pd.to_datetime(round_df["最终卖出日期"])
        drawdown_rounds = round_df[
            (round_df["最终卖出日期"] >= drawdown_meta["回撤起点"])
            & (round_df["最终卖出日期"] <= drawdown_meta["回撤低点"])
        ].copy()
    else:
        drawdown_rounds = round_df.copy()

    longest_losing = _longest_losing_streak(round_df) if not round_df.empty else {
        "最大连续亏损轮次": 0,
        "最大连续亏损累计收益率": 0.0,
        "最大连续亏损开始": None,
        "最大连续亏损结束": None,
    }

    sell_reason_overall = round_df["最终卖出原因"].value_counts().to_dict() if not round_df.empty else {}
    sell_reason_drawdown = drawdown_rounds["最终卖出原因"].value_counts().to_dict() if not drawdown_rounds.empty else {}

    worst_rounds = (
        drawdown_rounds.nsmallest(20, "完整持有收益率")[
            ["股票代码", "首次买入日期", "最终卖出日期", "完整持有收益率", "持有天数", "最终卖出原因"]
        ]
        if not drawdown_rounds.empty
        else pd.DataFrame(columns=["股票代码", "首次买入日期", "最终卖出日期", "完整持有收益率", "持有天数", "最终卖出原因"])
    )

    summary_payload = {
        "当前最优配置": {
            "评分法": method_name,
            "方案名称": best["方案名称"],
            "参数名称": best["参数名称"],
            "参数": best["参数"],
            "横盘模式": best["横盘模式"],
            "横盘过滤阈值": best["横盘过滤阈值"],
            "横盘降分系数": best["横盘降分系数"],
        },
        "整体指标": {
            "年化收益": float(result["metrics"]["annual_return"]),
            "最大回撤": float(result["metrics"]["max_drawdown"]),
            "夏普比率": float(result["metrics"]["sharpe"]),
            "资金倍数": float(result["metrics"]["final_multiple"]),
            "最终资金": float(result["final_equity"]),
            "交易次数": int(result["trade_count"]),
            "完整轮次交易数": int(result["完整轮次交易数"]),
            "平均持有期间收益率": float(result["平均持有期间收益率"]),
        },
        "最大回撤区间": {
            "回撤起点": drawdown_meta["回撤起点"].strftime("%Y-%m-%d"),
            "回撤低点": drawdown_meta["回撤低点"].strftime("%Y-%m-%d"),
            "回撤恢复日": drawdown_meta["回撤恢复日"].strftime("%Y-%m-%d") if drawdown_meta["回撤恢复日"] is not None else None,
            "起点净值": float(drawdown_meta["起点净值"]),
            "低点净值": float(drawdown_meta["低点净值"]),
            "最大回撤": float(drawdown_meta["最大回撤"]),
            "回撤天数": int(drawdown_meta["回撤天数"]),
        },
        "回撤区间统计": {
            "回撤区间完整轮次数量": int(len(drawdown_rounds)),
            "回撤区间平均完整收益率": float(drawdown_rounds["完整持有收益率"].mean()) if not drawdown_rounds.empty else 0.0,
            "回撤区间中位完整收益率": float(drawdown_rounds["完整持有收益率"].median()) if not drawdown_rounds.empty else 0.0,
            "整体卖出原因分布": sell_reason_overall,
            "回撤区间卖出原因分布": sell_reason_drawdown,
        },
        "连续亏损归因": {
            "最大连续亏损轮次": int(longest_losing["最大连续亏损轮次"]),
            "最大连续亏损累计收益率": float(longest_losing["最大连续亏损累计收益率"]),
            "最大连续亏损开始": longest_losing["最大连续亏损开始"].strftime("%Y-%m-%d") if longest_losing["最大连续亏损开始"] is not None else None,
            "最大连续亏损结束": longest_losing["最大连续亏损结束"].strftime("%Y-%m-%d") if longest_losing["最大连续亏损结束"] is not None else None,
        },
        "回撤成因判断": _classify_drawdown(drawdown_rounds),
    }

    (output_root / "汇总结果.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    equity_curve.rename("equity").to_csv(output_root / "equity_curve.csv", encoding="utf-8")
    pd.DataFrame(result["trades"]).to_csv(output_root / "逐笔交易.csv", index=False, encoding="utf-8-sig")
    round_df.to_csv(output_root / "完整轮次交易.csv", index=False, encoding="utf-8-sig")
    worst_rounds.to_csv(output_root / "回撤区间最差轮次.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
