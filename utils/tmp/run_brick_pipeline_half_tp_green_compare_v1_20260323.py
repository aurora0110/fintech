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

    # 优先级固定：止损 > 分批止盈 > 砖块转绿
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
                if not half_sold:
                    final_exit_idx = next_idx
                    final_exit_price = next_open
                    exit_reason = "stop_before_partial_next_open"
                else:
                    final_exit_idx = next_idx
                    final_exit_price = next_open
                    exit_reason = "stop_after_partial_next_open"
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

    baseline_trades = build_trade_df(feature_map, candidate_df, "baseline")
    baseline_trades.to_csv(result_dir / "baseline_trades.csv", index=False, encoding="utf-8-sig")
    baseline_portfolio = brick_base.build_portfolio_curve(baseline_trades)
    baseline_portfolio.to_csv(result_dir / "baseline_portfolio.csv", index=False, encoding="utf-8-sig")
    update_progress(result_dir, "baseline_ready", trade_count=int(len(baseline_trades)))

    half_green_trades = build_trade_df(feature_map, candidate_df, "half_tp_green")
    half_green_trades.to_csv(result_dir / "half_tp_green_trades.csv", index=False, encoding="utf-8-sig")
    half_green_portfolio = brick_base.build_portfolio_curve(half_green_trades)
    half_green_portfolio.to_csv(result_dir / "half_tp_green_portfolio.csv", index=False, encoding="utf-8-sig")
    update_progress(result_dir, "half_green_ready", trade_count=int(len(half_green_trades)))

    baseline_summary = summarize("baseline_formal_best_tp3_sl099_hold3", baseline_trades, baseline_portfolio)
    half_green_summary = summarize("half_tp35_then_green_with_sl099", half_green_trades, half_green_portfolio)

    comparison_df = pd.DataFrame([baseline_summary, half_green_summary])
    comparison_df.to_csv(result_dir / "comparison.csv", index=False, encoding="utf-8-sig")

    write_json(
        result_dir / "summary.json",
        {
            "pipeline_config": str(config_path),
            "baseline": baseline_summary,
            "half_tp_green": half_green_summary,
            "diff_half_tp_green_minus_baseline": {
                "trade_count": int(half_green_summary["trade_count"] - baseline_summary["trade_count"]),
                "avg_trade_return": float(half_green_summary["avg_trade_return"] - baseline_summary["avg_trade_return"]),
                "success_rate": float(half_green_summary["success_rate"] - baseline_summary["success_rate"]),
                "avg_holding_days": float(half_green_summary["avg_holding_days"] - baseline_summary["avg_holding_days"]),
                "annual_return_signal_basket": float(
                    half_green_summary["annual_return_signal_basket"] - baseline_summary["annual_return_signal_basket"]
                ),
                "max_drawdown_signal_basket": float(
                    half_green_summary["max_drawdown_signal_basket"] - baseline_summary["max_drawdown_signal_basket"]
                ),
                "final_equity_signal_basket": float(
                    half_green_summary["final_equity_signal_basket"] - baseline_summary["final_equity_signal_basket"]
                ),
            },
            "notes": [
                "这轮直接使用 backtest_pipeline 的 brick.formal_best 候选池，不再使用旧桥接 brick.main。",
                "比较口径只换卖法，买点、排序、前50%过滤、top10 全部保持一致。",
                "半仓策略假设保留原 0.99 止损；优先级固定为 止损 > 3.5%半仓止盈 > 砖块转绿。",
                "若同一天既达到 3.5% 又砖块转绿，则两半仓都按次日开盘卖出。",
                "若直到样本结束都未触发剩余半仓卖点，则最后一个交易日收盘平仓。",
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
        result_dir = RESULT_ROOT / f"brick_pipeline_half_tp_green_compare_v1_{suffix}_20260323"
    run_chain(config_path=config_path, result_dir=result_dir)


if __name__ == "__main__":
    main()
