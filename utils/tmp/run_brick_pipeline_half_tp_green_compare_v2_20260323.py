from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.backtest_pipeline.candidate_pools.base import CandidatePoolContext
from utils.backtest_pipeline.candidate_pools.brick import brick_base, _load_feature_map
from utils.backtest_pipeline.catalog import register_builtin_modules
from utils.backtest_pipeline.rankers.base import RankerContext
from utils.backtest_pipeline.registry import CANDIDATE_POOL_REGISTRY, RANKER_REGISTRY
from utils.backtest_pipeline.runner import load_pipeline_config


SMOKE_CONFIG = Path("/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/configs/brick_formal_best_pipeline_smoke.json")
FULL_CONFIG = Path("/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/configs/brick_formal_best_pipeline.json")
RESULT_ROOT = Path("/Users/lidongyang/Desktop/Qstrategy/results")
EPS = 1e-12


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def update_progress(result_dir: Path, stage: str, **extra) -> None:
    payload = {"stage": stage}
    payload.update(extra)
    write_json(result_dir / "progress.json", payload)


def load_candidates_from_pipeline(config_path: Path) -> pd.DataFrame:
    register_builtin_modules()
    config = load_pipeline_config(str(config_path))
    candidate_cls = CANDIDATE_POOL_REGISTRY.get(config.strategy.candidate_pool).builder
    candidate_pool = candidate_cls()
    candidate_df = candidate_pool.generate(
        CandidatePoolContext(
            data_dir=config.data.data_dir,
            stock_data={},
            all_dates=[],
            strategy_config=config.strategy,
        )
    )
    ranker_cls = RANKER_REGISTRY.get(config.ranker.name).builder
    ranker = ranker_cls()
    candidate_df = ranker.score(RankerContext(candidate_df=candidate_df, params=config.ranker.params))
    candidate_df = candidate_df.sort_values(["signal_date", "rank_score", "code"], ascending=[True, False, True]).copy()
    return candidate_df.groupby("signal_date", group_keys=False).head(config.ranker.top_n).reset_index(drop=True)


def calc_profit_factor(ret_series: pd.Series) -> float:
    pos = float(ret_series[ret_series > 0].sum())
    neg = float(ret_series[ret_series < 0].sum())
    if abs(neg) < EPS:
        return np.nan
    return pos / abs(neg)


def _baseline_time_exit(df: pd.DataFrame, entry_idx: int) -> tuple[int, float, str]:
    n = len(df)
    exit_idx = min(entry_idx + 3 + 1, n - 1)
    exit_price = float(df.at[exit_idx, "open"])
    return exit_idx, exit_price, "time_exit_next_open"


def simulate_half_tp_then_green(df: pd.DataFrame, signal_idx: int, first_take_profit_pct: float = 0.035) -> Optional[dict]:
    n = len(df)
    entry_idx = signal_idx + 1
    if entry_idx >= n:
        return None

    entry_price = float(df.at[entry_idx, "open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    signal_low = float(df.at[signal_idx, "low"])
    stop_price = brick_base.stop_loss_price(signal_low, "entry_low_x_0.99")
    tp_price = entry_price * (1.0 + first_take_profit_pct)

    half_sold = False
    half_exit_idx = None
    half_exit_price = None
    final_exit_idx = None
    final_exit_price = None
    exit_reason = "end_of_data_close"

    for j in range(entry_idx + 1, n):
        low_j = float(df.at[j, "low"])
        high_j = float(df.at[j, "high"])
        green_j = bool(df.at[j, "brick_green"])

        if stop_price is not None and np.isfinite(low_j) and low_j <= stop_price:
            next_idx = min(j + 1, n - 1)
            next_open = float(df.at[next_idx, "open"])
            if np.isfinite(next_open) and next_open > 0:
                final_exit_idx = next_idx
                final_exit_price = next_open
                exit_reason = "stop_before_partial_next_open" if not half_sold else "stop_after_partial_next_open"
                break

        if not half_sold and np.isfinite(high_j) and high_j >= tp_price:
            next_idx = min(j + 1, n - 1)
            next_open = float(df.at[next_idx, "open"])
            if np.isfinite(next_open) and next_open > 0:
                half_sold = True
                half_exit_idx = next_idx
                half_exit_price = next_open
                if green_j:
                    final_exit_idx = next_idx
                    final_exit_price = next_open
                    exit_reason = "tp_half_and_green_same_day_next_open"
                    break
                continue

        if green_j:
            next_idx = min(j + 1, n - 1)
            next_open = float(df.at[next_idx, "open"])
            if np.isfinite(next_open) and next_open > 0:
                final_exit_idx = next_idx
                final_exit_price = next_open
                exit_reason = "green_next_open_before_partial" if not half_sold else "green_next_open_after_partial"
                break

    if final_exit_idx is None:
        final_exit_idx = n - 1
        final_exit_price = float(df.at[final_exit_idx, "close"])
        exit_reason = "end_of_data_close"

    if not np.isfinite(final_exit_price) or final_exit_price <= 0:
        return None

    if half_sold and half_exit_price is not None:
        ret = 0.5 * (half_exit_price / entry_price - 1.0) + 0.5 * (final_exit_price / entry_price - 1.0)
        half_exit_date = df.at[half_exit_idx, "date"] if half_exit_idx is not None else None
    else:
        ret = final_exit_price / entry_price - 1.0
        half_exit_date = None

    return {
        "signal_date": df.at[signal_idx, "date"],
        "entry_date": df.at[entry_idx, "date"],
        "half_exit_date": half_exit_date,
        "exit_date": df.at[final_exit_idx, "date"],
        "entry_price": entry_price,
        "half_exit_price": half_exit_price,
        "exit_price": final_exit_price,
        "ret": ret,
        "holding_days": int(final_exit_idx - entry_idx),
        "success": ret > 0,
        "exit_reason": exit_reason,
        "signal_low": signal_low,
    }


def simulate_half_tp_then_green_profit_only(
    df: pd.DataFrame,
    signal_idx: int,
    first_take_profit_pct: float = 0.035,
    baseline_take_profit_pct: float = 0.03,
) -> Optional[dict]:
    n = len(df)
    entry_idx = signal_idx + 1
    if entry_idx >= n:
        return None

    entry_price = float(df.at[entry_idx, "open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    signal_low = float(df.at[signal_idx, "low"])
    stop_price = brick_base.stop_loss_price(signal_low, "entry_low_x_0.99")
    partial_tp_price = entry_price * (1.0 + first_take_profit_pct)
    baseline_tp_price = entry_price * (1.0 + baseline_take_profit_pct)

    # 在半仓止盈前，整笔交易完全沿用原冠军策略：0.99止损、3%止盈、最多持有3天。
    for j in range(entry_idx + 1, min(entry_idx + 3, n - 1) + 1):
        low_j = float(df.at[j, "low"])
        high_j = float(df.at[j, "high"])
        green_j = bool(df.at[j, "brick_green"])

        if stop_price is not None and np.isfinite(low_j) and low_j <= stop_price:
            next_idx = min(j + 1, n - 1)
            next_open = float(df.at[next_idx, "open"])
            if np.isfinite(next_open) and next_open > 0:
                ret = next_open / entry_price - 1.0
                return {
                    "signal_date": df.at[signal_idx, "date"],
                    "entry_date": df.at[entry_idx, "date"],
                    "half_exit_date": None,
                    "exit_date": df.at[next_idx, "date"],
                    "entry_price": entry_price,
                    "half_exit_price": None,
                    "exit_price": next_open,
                    "ret": ret,
                    "holding_days": int(next_idx - entry_idx),
                    "success": ret > 0,
                    "exit_reason": "stop_loss_next_open",
                    "signal_low": signal_low,
                }

        if np.isfinite(high_j) and high_j >= partial_tp_price:
            next_idx = min(j + 1, n - 1)
            next_open = float(df.at[next_idx, "open"])
            if not (np.isfinite(next_open) and next_open > 0):
                return None

            # 半仓止盈后，剩余半仓只在盈利仓路径里等待砖块转绿；止损仍保留。
            if green_j:
                weighted_ret = next_open / entry_price - 1.0
                return {
                    "signal_date": df.at[signal_idx, "date"],
                    "entry_date": df.at[entry_idx, "date"],
                    "half_exit_date": df.at[next_idx, "date"],
                    "exit_date": df.at[next_idx, "date"],
                    "entry_price": entry_price,
                    "half_exit_price": next_open,
                    "exit_price": next_open,
                    "ret": weighted_ret,
                    "holding_days": int(next_idx - entry_idx),
                    "success": weighted_ret > 0,
                    "exit_reason": "tp_half_and_green_same_day_next_open",
                    "signal_low": signal_low,
                }

            final_exit_idx = None
            final_exit_price = None
            exit_reason = "end_of_data_close_after_partial"
            for k in range(j + 1, n):
                low_k = float(df.at[k, "low"])
                green_k = bool(df.at[k, "brick_green"])
                if stop_price is not None and np.isfinite(low_k) and low_k <= stop_price:
                    out_idx = min(k + 1, n - 1)
                    out_open = float(df.at[out_idx, "open"])
                    if np.isfinite(out_open) and out_open > 0:
                        final_exit_idx = out_idx
                        final_exit_price = out_open
                        exit_reason = "stop_after_partial_next_open"
                        break
                if green_k:
                    out_idx = min(k + 1, n - 1)
                    out_open = float(df.at[out_idx, "open"])
                    if np.isfinite(out_open) and out_open > 0:
                        final_exit_idx = out_idx
                        final_exit_price = out_open
                        exit_reason = "green_next_open_after_partial"
                        break

            if final_exit_idx is None:
                final_exit_idx = n - 1
                final_exit_price = float(df.at[final_exit_idx, "close"])

            if not np.isfinite(final_exit_price) or final_exit_price <= 0:
                return None

            weighted_ret = 0.5 * (next_open / entry_price - 1.0) + 0.5 * (final_exit_price / entry_price - 1.0)
            return {
                "signal_date": df.at[signal_idx, "date"],
                "entry_date": df.at[entry_idx, "date"],
                "half_exit_date": df.at[next_idx, "date"],
                "exit_date": df.at[final_exit_idx, "date"],
                "entry_price": entry_price,
                "half_exit_price": next_open,
                "exit_price": final_exit_price,
                "ret": weighted_ret,
                "holding_days": int(final_exit_idx - entry_idx),
                "success": weighted_ret > 0,
                "exit_reason": exit_reason,
                "signal_low": signal_low,
            }

        if np.isfinite(high_j) and high_j >= baseline_tp_price:
            next_idx = min(j + 1, n - 1)
            next_open = float(df.at[next_idx, "open"])
            if np.isfinite(next_open) and next_open > 0:
                ret = next_open / entry_price - 1.0
                return {
                    "signal_date": df.at[signal_idx, "date"],
                    "entry_date": df.at[entry_idx, "date"],
                    "half_exit_date": None,
                    "exit_date": df.at[next_idx, "date"],
                    "entry_price": entry_price,
                    "half_exit_price": None,
                    "exit_price": next_open,
                    "ret": ret,
                    "holding_days": int(next_idx - entry_idx),
                    "success": ret > 0,
                    "exit_reason": "take_profit_next_open",
                    "signal_low": signal_low,
                }

    exit_idx, exit_price, exit_reason = _baseline_time_exit(df, entry_idx)
    if not np.isfinite(exit_price) or exit_price <= 0:
        return None
    ret = exit_price / entry_price - 1.0
    return {
        "signal_date": df.at[signal_idx, "date"],
        "entry_date": df.at[entry_idx, "date"],
        "half_exit_date": None,
        "exit_date": df.at[exit_idx, "date"],
        "entry_price": entry_price,
        "half_exit_price": None,
        "exit_price": exit_price,
        "ret": ret,
        "holding_days": int(exit_idx - entry_idx),
        "success": ret > 0,
        "exit_reason": exit_reason,
        "signal_low": signal_low,
    }


def build_trade_df(feature_map: Dict[str, pd.DataFrame], candidate_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    rows = []
    total = len(candidate_df)
    combo = brick_base.Combo(rebound_threshold=1.2, gain_limit=0.08, take_profit=0.03, stop_mode="entry_low_x_0.99")
    for idx, row in enumerate(candidate_df.itertuples(index=False), 1):
        df = feature_map[row.code]
        if mode == "baseline":
            trade = brick_base.simulate_trade(df, int(row.signal_idx), combo)
        elif mode == "half_tp_green":
            trade = simulate_half_tp_then_green(df, int(row.signal_idx), first_take_profit_pct=0.035)
        elif mode == "half_tp_green_profit_only":
            trade = simulate_half_tp_then_green_profit_only(
                df,
                int(row.signal_idx),
                first_take_profit_pct=0.035,
                baseline_take_profit_pct=combo.take_profit,
            )
        else:
            raise ValueError(mode)
        if trade is None:
            continue
        trade["code"] = row.code
        trade["sort_score"] = float(row.rank_score)
        trade["rebound_ratio"] = float(row.rebound_ratio)
        trade["signal_vs_ma5"] = float(row.signal_vs_ma5)
        rows.append(trade)
        if idx % 500 == 0 or idx == total:
            print(f"BRICK pipeline {mode} 回放进度: {idx}/{total}")
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["signal_date", "code"]).reset_index(drop=True)


def summarize(name: str, trade_df: pd.DataFrame, portfolio_df: pd.DataFrame) -> dict:
    if trade_df.empty:
        return {
            "strategy": name,
            "trade_count": 0,
            "avg_trade_return": np.nan,
            "success_rate": np.nan,
            "avg_holding_days": np.nan,
            "profit_factor": np.nan,
            "max_consecutive_failures": np.nan,
            "annual_return_signal_basket": np.nan,
            "max_drawdown_signal_basket": np.nan,
            "final_equity_signal_basket": np.nan,
            "equity_days_signal_basket": 0,
        }
    metrics = brick_base.compute_equity_metrics(portfolio_df)
    return {
        "strategy": name,
        "trade_count": int(len(trade_df)),
        "avg_trade_return": float(trade_df["ret"].mean()),
        "success_rate": float(trade_df["success"].mean()),
        "avg_holding_days": float(trade_df["holding_days"].mean()),
        "profit_factor": float(calc_profit_factor(trade_df["ret"])),
        "max_consecutive_failures": int(brick_base.max_consecutive_failures(trade_df["success"].tolist())),
        "annual_return_signal_basket": float(metrics["annual_return"]),
        "max_drawdown_signal_basket": float(metrics["max_drawdown"]),
        "final_equity_signal_basket": float(metrics["final_equity"]),
        "equity_days_signal_basket": int(metrics["equity_days"]),
    }


def run_chain(config_path: Path, result_dir: Path) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    update_progress(result_dir, "starting", config=str(config_path))
    candidate_df = load_candidates_from_pipeline(config_path)
    candidate_df.to_csv(result_dir / "pipeline_selected_signals.csv", index=False, encoding="utf-8-sig")
    update_progress(result_dir, "signals_ready", selected_signal_count=int(len(candidate_df)))

    config = load_pipeline_config(str(config_path))
    file_limit = int(config.strategy.params.get("file_limit", 0) or 0)
    feature_map = _load_feature_map(config.data.data_dir, file_limit=file_limit)
    update_progress(result_dir, "features_ready", stock_count=len(feature_map))

    summaries = {}
    comparison_rows = []
    mode_specs = [
        ("baseline", "baseline_formal_best_tp3_sl099_hold3"),
        ("half_tp_green", "half_tp35_then_green_with_sl099"),
        ("half_tp_green_profit_only", "half_tp35_profit_only_then_green_else_baseline"),
    ]
    for mode, label in mode_specs:
        trade_df = build_trade_df(feature_map, candidate_df, mode)
        trade_df.to_csv(result_dir / f"{mode}_trades.csv", index=False, encoding="utf-8-sig")
        portfolio_df = brick_base.build_portfolio_curve(trade_df)
        portfolio_df.to_csv(result_dir / f"{mode}_portfolio.csv", index=False, encoding="utf-8-sig")
        update_progress(result_dir, f"{mode}_ready", trade_count=int(len(trade_df)))
        summary = summarize(label, trade_df, portfolio_df)
        summaries[mode] = summary
        comparison_rows.append(summary)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(result_dir / "comparison.csv", index=False, encoding="utf-8-sig")

    baseline_summary = summaries["baseline"]
    profit_only_summary = summaries["half_tp_green_profit_only"]

    write_json(
        result_dir / "summary.json",
        {
            "pipeline_config": str(config_path),
            "baseline": baseline_summary,
            "half_tp_green": summaries["half_tp_green"],
            "half_tp_green_profit_only": profit_only_summary,
            "diff_profit_only_minus_baseline": {
                "trade_count": int(profit_only_summary["trade_count"] - baseline_summary["trade_count"]),
                "avg_trade_return": float(profit_only_summary["avg_trade_return"] - baseline_summary["avg_trade_return"]),
                "success_rate": float(profit_only_summary["success_rate"] - baseline_summary["success_rate"]),
                "avg_holding_days": float(profit_only_summary["avg_holding_days"] - baseline_summary["avg_holding_days"]),
                "annual_return_signal_basket": float(
                    profit_only_summary["annual_return_signal_basket"] - baseline_summary["annual_return_signal_basket"]
                ),
                "max_drawdown_signal_basket": float(
                    profit_only_summary["max_drawdown_signal_basket"] - baseline_summary["max_drawdown_signal_basket"]
                ),
                "final_equity_signal_basket": float(
                    profit_only_summary["final_equity_signal_basket"] - baseline_summary["final_equity_signal_basket"]
                ),
            },
            "notes": [
                "这轮直接使用 backtest_pipeline 的 brick.formal_best 候选池，不再使用旧桥接 brick.main。",
                "比较口径只换卖法，买点、排序、前50%过滤、top10 全部保持一致。",
                "profit_only 版本含义：只有先达到3.5%并次日卖出半仓后，剩余半仓才等砖块转绿；否则整笔交易继续按原 3%止盈+0.99止损+3天到期。",
                "止损优先级固定为：止损 > 3.5%半仓止盈 > 其他退出。",
            ],
        },
    )
    update_progress(result_dir, "finished")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--result-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = SMOKE_CONFIG if args.mode == "smoke" else FULL_CONFIG
    if args.result_dir:
        result_dir = Path(args.result_dir)
    else:
        suffix = "smoke" if args.mode == "smoke" else "full"
        result_dir = RESULT_ROOT / f"brick_pipeline_half_tp_green_compare_v2_{suffix}_20260323"
    run_chain(config_path=config_path, result_dir=result_dir)


if __name__ == "__main__":
    main()
