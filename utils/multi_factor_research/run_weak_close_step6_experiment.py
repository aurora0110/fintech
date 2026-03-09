from __future__ import annotations

import argparse
import ast
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


止损类原因 = ["止损卖出", "启动失败卖出", "滴滴止损", "两根跌破多空线止损"]


def _写状态(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
    if method_name == "加权法":
        add_weights, penalty_weights = _load_formal_scorecard(weighted_root)
    else:
        add_weights, penalty_weights = _load_formal_scorecard(binary_root)
    variants = _variant_scorecards(method_name, add_weights, penalty_weights)
    for item in variants:
        if item["名称"] == str(best["方案名称"]):
            return method_name, dict(item["加分"]), dict(item["扣分"])
    raise RuntimeError(f"未找到评分方案: {method_name} / {best['方案名称']}")


def _drawdown_window(equity_curve: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1.0
    trough_date = pd.Timestamp(drawdown.idxmin())
    peak_date = pd.Timestamp(equity_curve.loc[:trough_date].idxmax())
    return peak_date, trough_date


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


def _candidate_cfg_from_best(best: dict) -> dict:
    return {
        "use_trend_start_pool": True,
        "min_confirmation_hits": 1,
        "min_support_hits": 1,
        "sideways_mode": str(best["横盘模式"]),
        "sideways_filter_threshold": float(best["横盘过滤阈值"]),
        "sideways_score_penalty_scale": float(best["横盘降分系数"]),
    }


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


def _future_follow_stats(
    trades_df: pd.DataFrame,
    prepared: dict[str, pd.DataFrame],
    windows: list[int],
    drawdown_start: pd.Timestamp | None = None,
    drawdown_end: pd.Timestamp | None = None,
) -> list[dict]:
    rows: list[dict] = []
    if trades_df.empty:
        return rows
    for trade in trades_df.itertuples(index=False):
        code = str(trade.code)
        if code not in prepared:
            continue
        df = prepared[code]
        exit_date = pd.Timestamp(trade.exit_date)
        if exit_date not in df.index:
            continue
        idx = df.index.get_loc(exit_date)
        if idx + 1 >= len(df.index):
            continue
        for window in windows:
            end_idx = min(idx + window, len(df.index) - 1)
            if end_idx <= idx:
                continue
            future = df.iloc[idx + 1:end_idx + 1]
            if future.empty:
                continue
            exit_price = float(trade.exit_price)
            rows.append(
                {
                    "卖出原因": str(trade.reason),
                    "窗口天数": window,
                    "是否在最大回撤区间": bool(
                        drawdown_start is not None
                        and drawdown_end is not None
                        and drawdown_start <= exit_date <= drawdown_end
                    ),
                    "样本数": 1,
                    "窗口终点继续下跌": float(future.iloc[-1]["close"]) < exit_price,
                    "窗口内继续创新低": float(future["low"].min()) < exit_price,
                    "窗口内最大反弹达到5pct": float(future["high"].max()) >= exit_price * 1.05,
                }
            )
    return rows


def _aggregate_follow_stats(detail_rows: list[dict]) -> pd.DataFrame:
    if not detail_rows:
        return pd.DataFrame(
            columns=[
                "分组",
                "卖出原因",
                "窗口天数",
                "样本数",
                "窗口终点继续下跌概率",
                "窗口内继续创新低概率",
                "窗口内最大反弹达到5pct概率",
            ]
        )
    detail = pd.DataFrame(detail_rows)
    result_rows: list[dict] = []
    for group_name, frame in [
        ("总体", detail),
        ("最大回撤区间", detail[detail["是否在最大回撤区间"]]),
        ("非最大回撤区间", detail[~detail["是否在最大回撤区间"]]),
    ]:
        if frame.empty:
            continue
        for reason, sub in list(frame.groupby("卖出原因")) + [("全部止损类", frame)]:
            for window, win_sub in sub.groupby("窗口天数"):
                result_rows.append(
                    {
                        "分组": group_name,
                        "卖出原因": reason,
                        "窗口天数": int(window),
                        "样本数": int(len(win_sub)),
                        "窗口终点继续下跌概率": float(win_sub["窗口终点继续下跌"].mean()),
                        "窗口内继续创新低概率": float(win_sub["窗口内继续创新低"].mean()),
                        "窗口内最大反弹达到5pct概率": float(win_sub["窗口内最大反弹达到5pct"].mean()),
                    }
                )
    return pd.DataFrame(result_rows)


def _strategy_result_row(
    label: str,
    variant_name: str,
    result: dict,
    prepared: dict[str, pd.DataFrame],
    windows: list[int],
) -> dict:
    trades = pd.DataFrame(result["trades"])
    metrics = result["metrics"]
    weak_close_count = int((trades["reason"] == "三天连续弱收盘卖出").sum()) if not trades.empty else 0
    weak_close_stats = _aggregate_follow_stats(
        _future_follow_stats(trades[trades["reason"] == "三天连续弱收盘卖出"], prepared, windows)
    )
    row = {
        "策略": label,
        "版本": variant_name,
        "年化收益": float(metrics["annual_return"]),
        "最大回撤": float(metrics["max_drawdown"]),
        "夏普比率": float(metrics["sharpe"]),
        "资金倍数": float(metrics["final_multiple"]),
        "最终资金": float(result["final_equity"]),
        "交易次数": int(result["trade_count"]),
        "完整轮次交易数": int(result["完整轮次交易数"]),
        "平均持有期间收益率": float(result["平均持有期间收益率"]),
        "盈利轮次占比": float(result["盈利轮次占比"]),
        "三天连续弱收盘触发次数": weak_close_count,
    }
    for window in windows:
        sub = weak_close_stats[(weak_close_stats["分组"] == "总体") & (weak_close_stats["卖出原因"] == "全部止损类") & (weak_close_stats["窗口天数"] == window)]
        if sub.empty:
            row[f"三天连续弱收盘卖出后{window}天继续下跌概率"] = 0.0
            row[f"三天连续弱收盘卖出后{window}天继续创新低概率"] = 0.0
        else:
            row[f"三天连续弱收盘卖出后{window}天继续下跌概率"] = float(sub.iloc[0]["窗口终点继续下跌概率"])
            row[f"三天连续弱收盘卖出后{window}天继续创新低概率"] = float(sub.iloc[0]["窗口内继续创新低概率"])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="第6步实验：三天连续弱收盘条件评估")
    parser.add_argument("--data-dir", default="/Users/lidongyang/Desktop/Qstrategy/data/20260226")
    parser.add_argument("--factor-opt-summary", default="/Users/lidongyang/Desktop/Qstrategy/results/factor_optimization_experiment_v1/汇总结果.json")
    parser.add_argument("--problem-attr-csv", default="/Users/lidongyang/Desktop/Qstrategy/results/problem_attribution_experiment_v1/问题归因实验结果.csv")
    parser.add_argument("--weighted-scorecard-root", default="/Users/lidongyang/Desktop/Qstrategy/results/formal_scorecard_trend_v2_pool_weighted")
    parser.add_argument("--binary-scorecard-root", default="/Users/lidongyang/Desktop/Qstrategy/results/formal_scorecard_trend_v2_pool_binary")
    parser.add_argument("--output-root", default="/Users/lidongyang/Desktop/Qstrategy/results/weak_close_step6_v1")
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
    windows = [5, 10, 20]

    _写状态(status_path, {"阶段": "读取最优配置", "完成组数": 0})
    factor_opt_summary = json.loads(Path(args.factor_opt_summary).read_text(encoding="utf-8"))
    best = _select_best_config(factor_opt_summary)
    method_name, add_weights, penalty_weights = _resolve_scorecard(
        best,
        Path(args.weighted_scorecard_root),
        Path(args.binary_scorecard_root),
    )
    candidate_cfg = _candidate_cfg_from_best(best)
    strategy_profiles = _parse_strategy_profiles(Path(args.problem_attr_csv), method_name)

    _写状态(status_path, {"阶段": "加载行情数据", "完成组数": 0, "评分法": method_name, "方案名称": best["方案名称"]})
    stock_data, all_dates = load_price_directory(args.data_dir)
    prepared = _prepare_stock_frames(stock_data)

    _写状态(status_path, {"阶段": "运行观测实验", "完成组数": 0})
    baseline_profile = dict(strategy_profiles["fixed_days"]["参数"])
    baseline_config = _make_config(
        baseline_profile,
        candidate_cfg,
        args.initial_capital,
        args.max_positions,
        args.buy_mode,
        args.replacement_threshold,
        args.min_hold_days_for_replace,
        args.max_daily_replacements,
    )
    baseline_result = _run_model("fixed_days", prepared, all_dates, add_weights, penalty_weights, baseline_config)
    drawdown_start, drawdown_end = _drawdown_window(baseline_result["equity_curve"])
    baseline_trades = pd.DataFrame(baseline_result["trades"])
    stop_trades = baseline_trades[baseline_trades["reason"].isin(止损类原因)].copy() if not baseline_trades.empty else pd.DataFrame()
    follow_detail = _future_follow_stats(stop_trades, prepared, windows, drawdown_start, drawdown_end)
    follow_summary = _aggregate_follow_stats(follow_detail)
    follow_summary.to_csv(output_root / "止损后继续下跌概率.csv", index=False, encoding="utf-8-sig")

    tasks = []
    for label in ["fixed_take_profit", "fixed_days", "tiered"]:
        tasks.append((label, "原始退出", False))
        tasks.append((label, "加入三天连续弱收盘", True))

    result_rows: list[dict] = []
    total = len(tasks)
    for done, (label, variant_name, enable_weak_close) in enumerate(tqdm(tasks, desc="第6步策略对照", unit="组"), start=1):
        profile = dict(strategy_profiles[label]["参数"])
        profile["启用三天连续弱收盘退出"] = enable_weak_close
        _写状态(
            status_path,
            {
                "阶段": "运行策略对照实验",
                "完成组数": done - 1,
                "总组数": total,
                "评分法": method_name,
                "策略": label,
                "版本": variant_name,
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
        result = _run_model(label, prepared, all_dates, add_weights, penalty_weights, config)
        row = _strategy_result_row(label, variant_name, result, prepared, windows)
        row["评分法"] = method_name
        row["方案名称"] = best["方案名称"]
        row["参数名称"] = strategy_profiles[label]["参数名称"]
        result_rows.append(row)

    strategy_df = pd.DataFrame(result_rows)
    strategy_df.to_csv(output_root / "策略对照实验结果.csv", index=False, encoding="utf-8-sig")

    summary = {
        "当前最优趋势配置": {
            "评分法": method_name,
            "方案名称": best["方案名称"],
            "候选池": "趋势池V1",
            "横盘模式": candidate_cfg["sideways_mode"],
            "横盘过滤阈值": candidate_cfg["sideways_filter_threshold"],
            "横盘降分系数": candidate_cfg["sideways_score_penalty_scale"],
        },
        "观测实验基线参数": strategy_profiles["fixed_days"]["参数"],
        "最大回撤区间": {
            "回撤起点": drawdown_start.strftime("%Y-%m-%d"),
            "回撤低点": drawdown_end.strftime("%Y-%m-%d"),
        },
        "止损后继续下跌统计样本数": int(len(stop_trades)),
    }
    (output_root / "汇总结果.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _写状态(status_path, {"阶段": "已完成", "完成组数": total, "总组数": total, "评分法": method_name})


if __name__ == "__main__":
    main()
