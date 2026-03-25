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


def simulate_nextday_exec_tp_first(df: pd.DataFrame, signal_idx: int, combo: brick_base.Combo) -> Optional[dict]:
    n = len(df)
    entry_idx = signal_idx + 1
    if entry_idx >= n:
        return None

    entry_price = float(df.at[entry_idx, "open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    signal_low = float(df.at[signal_idx, "low"])
    sl_price = brick_base.stop_loss_price(signal_low, combo.stop_mode)
    tp_price = entry_price * (1.0 + combo.take_profit)

    scheduled_exit_idx = min(entry_idx + 3 + 1, n - 1)
    exit_idx = scheduled_exit_idx
    exit_price = float(df.at[exit_idx, "open"])
    exit_reason = "time_exit_next_open"

    for j in range(entry_idx + 1, min(entry_idx + 3, n - 1) + 1):
        high_j = float(df.at[j, "high"])
        low_j = float(df.at[j, "low"])
        tp_hit = np.isfinite(high_j) and high_j >= tp_price
        sl_hit = sl_price is not None and np.isfinite(low_j) and low_j <= sl_price

        # 用户指定：同日止盈优先
        if tp_hit or sl_hit:
            next_idx = min(j + 1, n - 1)
            if next_idx > entry_idx:
                px = float(df.at[next_idx, "open"])
                if np.isfinite(px) and px > 0:
                    exit_idx = next_idx
                    exit_price = px
                    exit_reason = "take_profit_next_open" if tp_hit else "stop_loss_next_open"
                    break

    ret = exit_price / entry_price - 1.0
    return {
        "signal_date": df.at[signal_idx, "date"],
        "entry_date": df.at[entry_idx, "date"],
        "exit_date": df.at[exit_idx, "date"],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "ret": ret,
        "holding_days": int(exit_idx - entry_idx),
        "success": ret > 0,
        "exit_reason": exit_reason,
        "signal_low": signal_low,
    }


def simulate_intraday_exec_tp_first(df: pd.DataFrame, signal_idx: int, combo: brick_base.Combo) -> Optional[dict]:
    n = len(df)
    entry_idx = signal_idx + 1
    if entry_idx >= n:
        return None

    entry_price = float(df.at[entry_idx, "open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    signal_low = float(df.at[signal_idx, "low"])
    sl_price = brick_base.stop_loss_price(signal_low, combo.stop_mode)
    tp_price = entry_price * (1.0 + combo.take_profit)

    exit_idx = min(entry_idx + 3 + 1, n - 1)
    exit_price = float(df.at[exit_idx, "open"])
    exit_reason = "time_exit_next_open"

    for j in range(entry_idx + 1, min(entry_idx + 3, n - 1) + 1):
        high_j = float(df.at[j, "high"])
        low_j = float(df.at[j, "low"])
        tp_hit = np.isfinite(high_j) and high_j >= tp_price
        sl_hit = sl_price is not None and np.isfinite(low_j) and low_j <= sl_price

        if tp_hit or sl_hit:
            exit_idx = j
            if tp_hit:
                exit_price = tp_price
                exit_reason = "take_profit_intraday"
            else:
                exit_price = sl_price
                exit_reason = "stop_loss_intraday"
            break

    ret = exit_price / entry_price - 1.0
    return {
        "signal_date": df.at[signal_idx, "date"],
        "entry_date": df.at[entry_idx, "date"],
        "exit_date": df.at[exit_idx, "date"],
        "entry_price": entry_price,
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
        if mode == "nextday_exec":
            trade = simulate_nextday_exec_tp_first(df, int(row.signal_idx), combo)
        elif mode == "intraday_exec":
            trade = simulate_intraday_exec_tp_first(df, int(row.signal_idx), combo)
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
            print(f"BRICK {mode} 回放进度: {idx}/{total}")
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
        ("nextday_exec", "tp_first_nextday_exec"),
        ("intraday_exec", "tp_first_intraday_exec"),
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

    nextday_summary = summaries["nextday_exec"]
    intraday_summary = summaries["intraday_exec"]

    write_json(
        result_dir / "summary.json",
        {
            "pipeline_config": str(config_path),
            "nextday_exec": nextday_summary,
            "intraday_exec": intraday_summary,
            "diff_intraday_minus_nextday": {
                "trade_count": int(intraday_summary["trade_count"] - nextday_summary["trade_count"]),
                "avg_trade_return": float(intraday_summary["avg_trade_return"] - nextday_summary["avg_trade_return"]),
                "success_rate": float(intraday_summary["success_rate"] - nextday_summary["success_rate"]),
                "avg_holding_days": float(intraday_summary["avg_holding_days"] - nextday_summary["avg_holding_days"]),
                "annual_return_signal_basket": float(
                    intraday_summary["annual_return_signal_basket"] - nextday_summary["annual_return_signal_basket"]
                ),
                "max_drawdown_signal_basket": float(
                    intraday_summary["max_drawdown_signal_basket"] - nextday_summary["max_drawdown_signal_basket"]
                ),
                "final_equity_signal_basket": float(
                    intraday_summary["final_equity_signal_basket"] - nextday_summary["final_equity_signal_basket"]
                ),
            },
            "notes": [
                "这轮直接使用 backtest_pipeline 的 brick.formal_best 候选池，买点、排序、前50%过滤、top10 全部保持一致。",
                "比较口径只换退出执行方式：同样是 3% 止盈、entry_low*0.99 止损、最多持有3天。",
                "nextday_exec：当日盘中触发止盈/止损后，次日开盘成交；若同日都触发，按止盈优先。",
                "intraday_exec：当日盘中触发后按触发价成交；若同日都触发，按止盈优先。",
                "买入当日不能卖，首次可退出日为 entry_date 的下一交易日。",
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
        result_dir = RESULT_ROOT / f"brick_intraday_vs_nextday_compare_v1_{suffix}_20260324"
    run_chain(config_path=config_path, result_dir=result_dir)


if __name__ == "__main__":
    main()
